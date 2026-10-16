"""Agent Loop — the core execution engine.

Inspired by OpenCode's server-first architecture and OpenClaw's lane queue pattern.
Each user gets a serial execution queue (no concurrent agent turns per user).

Flow:
1. Message arrives from Telegram
2. Intent Router classifies → right agent type
3. Model Router picks → right model for the task
4. Soul Prompt assembled with memory context
5. Agent executes with tool calling
6. Result sent back to Telegram
"""

import json
import logging
from dataclasses import dataclass, field
from uuid import UUID

from app.services.agent.router import Intent, classify_intent, get_model_for_intent
from app.services.agent.soul import build_soul_prompt
from app.services.generation_service import (
    EmptyGenerationError,
    create_generation_response,
)

logger = logging.getLogger(__name__)
MAX_TURNS = 10  # Max tool-calling turns per interaction


@dataclass
class AgentMessage:
    role: str  # "user" or "assistant"
    content: str


@dataclass
class AgentContext:
    user_id: UUID
    chat_id: int  # Telegram chat ID
    user_name: str | None = None
    user_language: str = "en"
    timezone: str = "UTC"
    connected_services: list[str] = field(default_factory=list)
    identity_memories: list[str] = field(default_factory=list)
    working_context: list[str] = field(default_factory=list)
    recalled_memories: list[str] = field(default_factory=list)
    conversation_history: list[AgentMessage] = field(default_factory=list)
    has_voice: bool = False
    voice_transcript: str | None = None


@dataclass
class AgentResult:
    response: str
    intent: Intent
    model_used: str
    input_tokens: int = 0
    output_tokens: int = 0
    tool_calls: int = 0


# Provider-neutral function definitions. They are converted to Responses API tools
# at the model boundary.
NON_DATA_TOOLS = [
    {
        "name": "get_digest",
        "description": "Get AI-generated summary of user's Telegram activity for a specific date.",
        "input_schema": {
            "type": "object",
            "properties": {
                "date": {
                    "type": "string",
                    "description": "Date in YYYY-MM-DD format. Defaults to yesterday.",
                },
            },
        },
    },
    {
        "name": "track_commitment",
        "description": "Track a promise or commitment detected in conversation.",
        "input_schema": {
            "type": "object",
            "properties": {
                "who": {
                    "type": "string",
                    "description": "Who made the promise (person name)",
                },
                "what": {
                    "type": "string",
                    "description": "What was promised",
                },
                "deadline": {
                    "type": "string",
                    "description": "When it should be done (YYYY-MM-DD or description)",
                },
                "direction": {
                    "type": "string",
                    "enum": ["i_promised", "they_promised"],
                    "description": "Whether user promised or someone else promised",
                },
            },
            "required": ["who", "what", "direction"],
        },
    },
    {
        "name": "extract_entities",
        "description": "Extract people, topics, decisions, dates, and amounts from text. Use when the user shares meeting notes, voice transcripts, or complex messages.",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to extract entities from",
                },
            },
            "required": ["text"],
        },
    },
    {
        "name": "list_commitments",
        "description": "List open commitments/promises. Shows what the user promised others and what others promised the user.",
        "input_schema": {
            "type": "object",
            "properties": {
                "direction": {
                    "type": "string",
                    "enum": ["all", "i_promised", "they_promised"],
                    "description": "Filter by direction. Default: all",
                },
            },
        },
    },
    {
        "name": "search_web",
        "description": "Search the internet for current information.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query",
                },
            },
            "required": ["query"],
        },
    },
]


def _data_tools() -> list[dict]:
    from app.services.tool_registry import TOOL_DEFINITIONS, required_scope

    return [
        {
            "name": definition.name,
            "description": definition.description,
            "input_schema": definition.parameters,
        }
        for definition in TOOL_DEFINITIONS
        if required_scope(definition.name) != "write"
    ]


TOOLS = _data_tools() + NON_DATA_TOOLS


def _responses_tools() -> list[dict]:
    return [
        {
            "type": "function",
            "name": tool["name"],
            "description": tool["description"],
            "parameters": tool["input_schema"],
        }
        for tool in TOOLS
    ]


async def execute_tool(tool_name: str, tool_input: dict, context: AgentContext) -> str:
    """Execute a tool call and return the result as a string.

    This is the central dispatch for all agent tools.
    Each tool is sandboxed and validates that the user owns the resource.
    """
    logger.info(f"Executing tool: {tool_name} for user {context.user_id}")

    from app.services.tool_registry import (
        _DEFINITION_BY_NAME,
        execute_data_tool,
        required_scope,
    )

    if tool_name in _DEFINITION_BY_NAME:
        if required_scope(tool_name) == "write":
            return (
                "Sending is available only through the explicitly approved MCP workflow"
            )
        from app.core.database import get_db_context

        async with get_db_context() as db:
            result = await execute_data_tool(db, context.user_id, tool_name, tool_input)
        return json.dumps(result, ensure_ascii=False, default=str)
    if tool_name == "get_digest":
        return await _tool_get_digest(tool_input, context)
    elif tool_name == "track_commitment":
        return await _tool_track_commitment(tool_input, context)
    elif tool_name == "extract_entities":
        return _tool_extract_entities(tool_input)
    elif tool_name == "list_commitments":
        return _tool_list_commitments(tool_input, context)
    elif tool_name == "search_web":
        from app.services.agent.web_search import search_web

        query = tool_input.get("query", "")
        return await search_web(query)
    else:
        return f"Unknown tool: {tool_name}"


async def _tool_search_messages(tool_input: dict, context: AgentContext) -> str:
    """Search user's message history via the existing search service."""
    from app.core.database import async_session_factory
    from app.schemas.search import SearchRequest
    from app.services.search_service import semantic_search

    query = tool_input.get("query", "")
    request = SearchRequest(
        query=query,
        limit=10,
    )

    async with async_session_factory() as db:
        results = await semantic_search(db, context.user_id, request)

    if not results.results:
        return f"No messages found matching: {query}"

    lines = []
    for r in results.results:
        sender = r.sender_name or "Unknown"
        chat = r.chat_title or "Unknown chat"
        date = r.sent_at.strftime("%Y-%m-%d %H:%M") if r.sent_at else ""
        text = (r.text or "")[:300]
        lines.append(f"[{chat}] {sender} ({date}): {text}")

    return "\n\n".join(lines)


async def _tool_get_digest(tool_input: dict, context: AgentContext) -> str:
    """Get digest for a specific date."""
    from datetime import date

    from app.core.database import async_session_factory
    from app.services.digest_service import generate_digest

    date_str = tool_input.get("date")
    if date_str:
        try:
            digest_date = date.fromisoformat(date_str)
        except ValueError:
            digest_date = None
    else:
        digest_date = None

    async with async_session_factory() as db:
        digest = await generate_digest(db, context.user_id, digest_date)
        return digest.content or "No digest available for this date."


async def _tool_track_commitment(tool_input: dict, context: AgentContext) -> str:
    """Track a commitment using the real commitment store."""
    from app.services.agent.commitments import (
        Commitment,
        CommitmentDirection,
        save_commitment,
    )

    who = tool_input.get("who", "Unknown")
    what = tool_input.get("what", "")
    deadline = tool_input.get("deadline")
    direction_str = tool_input.get("direction", "they_promised")

    direction = (
        CommitmentDirection.I_PROMISED
        if direction_str == "i_promised"
        else CommitmentDirection.THEY_PROMISED
    )

    commitment = Commitment(
        who=who,
        what=what,
        direction=direction,
        deadline=deadline,
    )
    save_commitment(commitment, context.user_id)

    if direction == CommitmentDirection.I_PROMISED:
        deadline_text = f" by {deadline}" if deadline else ""
        return f"✅ Tracked: You promised {who} to {what}{deadline_text}"
    else:
        deadline_text = f" by {deadline}" if deadline else ""
        return f"✅ Tracked: {who} promised to {what}{deadline_text}"


def _tool_extract_entities(tool_input: dict) -> str:
    """Extract entities from text using fast pattern matching."""
    from app.services.agent.entities import (
        extract_entities_fast,
        format_entities_for_display,
    )

    text = tool_input.get("text", "")
    if not text:
        return "No text provided for entity extraction."

    entities = extract_entities_fast(text)
    return format_entities_for_display(entities)


def _tool_list_commitments(tool_input: dict, context: AgentContext) -> str:
    """List user's open commitments."""
    from app.services.agent.commitments import (
        CommitmentDirection,
        format_commitments_for_display,
        get_user_commitments,
    )

    direction_str = tool_input.get("direction", "all")

    if direction_str == "i_promised":
        direction = CommitmentDirection.I_PROMISED
    elif direction_str == "they_promised":
        direction = CommitmentDirection.THEY_PROMISED
    else:
        direction = None

    commitments = get_user_commitments(context.user_id, direction=direction)
    return format_commitments_for_display(commitments)


async def run_agent(context: AgentContext, message: str) -> AgentResult:
    """Run the agent loop: classify → route → execute → respond.

    This is the main entry point for all user interactions.
    """
    from app.services.agent.metrics import increment

    # 1. Classify intent
    increment("agent_requests_total")
    intent = await classify_intent(message, has_voice=context.has_voice)
    model = get_model_for_intent(intent)
    increment(f"agent_intent_{intent.value}")

    logger.info(f"Agent: intent={intent.value}, model={model}, user={context.user_id}")

    # 2. Build soul prompt
    system_prompt = build_soul_prompt(
        user_name=context.user_name,
        user_language=context.user_language,
        timezone=context.timezone,
        connected_services=context.connected_services,
        identity_memories=context.identity_memories,
        working_context=context.working_context,
        recalled_memories=context.recalled_memories,
    )

    # 3. Build message history
    messages = []
    for msg in context.conversation_history[-20:]:  # Last 20 messages
        messages.append({"role": msg.role, "content": msg.content})

    # Add the current message
    user_content = message
    if context.voice_transcript:
        user_content = (
            f"[Voice message transcript]: {context.voice_transcript}\n\nUser's text: {message}"
            if message
            else f"[Voice message transcript]: {context.voice_transcript}"
        )

    messages.append({"role": "user", "content": user_content})

    # 4. Responses API agent loop with function calling
    input_items: list = messages
    total_input_tokens = 0
    total_output_tokens = 0
    tool_call_count = 0

    for _turn in range(MAX_TURNS):
        response = await create_generation_response(
            input_items,
            max_output_tokens=4096,
            instructions=system_prompt,
            tools=_responses_tools(),
        )

        if response.usage is not None:
            total_input_tokens += response.usage.input_tokens
            total_output_tokens += response.usage.output_tokens

        function_calls = [
            item for item in response.output if item.type == "function_call"
        ]
        if function_calls:
            input_items.extend(response.output)
            for call in function_calls:
                tool_call_count += 1
                tool_input = json.loads(call.arguments)
                if not isinstance(tool_input, dict):
                    raise ValueError(f"Tool arguments must be an object: {call.name}")
                try:
                    result = await execute_tool(call.name, tool_input, context)
                except Exception as exc:
                    logger.error("Tool %s failed: %s", call.name, exc)
                    result = f"Error executing {call.name}: {exc}"
                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": call.call_id,
                        "output": result,
                    }
                )
            continue

        final_response = response.output_text.strip()
        if not final_response:
            raise EmptyGenerationError("Agent response contained no text")

        increment("agent_tokens_input", total_input_tokens)
        increment("agent_tokens_output", total_output_tokens)
        increment("agent_tool_calls", tool_call_count)
        return AgentResult(
            response=final_response,
            intent=intent,
            model_used=model,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            tool_calls=tool_call_count,
        )

    raise RuntimeError(f"Agent exceeded the {MAX_TURNS}-turn tool-call limit")
