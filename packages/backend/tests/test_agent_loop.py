"""Tests for the Responses API agent loop and its tool boundary."""

import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock, patch
from uuid import uuid4

import pytest

from app.services.agent.loop import (
    MAX_TURNS,
    TOOLS,
    AgentContext,
    AgentMessage,
    AgentResult,
    _responses_tools,
    execute_tool,
    run_agent,
)
from app.services.agent.router import Intent
from app.services.generation_service import EmptyGenerationError


def _context(**overrides) -> AgentContext:
    values = {
        "user_id": uuid4(),
        "chat_id": 12345,
        "user_name": "TestUser",
        "user_language": "en",
        "timezone": "UTC",
    }
    values.update(overrides)
    return AgentContext(**values)


def _response(
    text: str = "",
    *,
    calls: list | None = None,
    input_tokens: int = 10,
    output_tokens: int = 20,
):
    return SimpleNamespace(
        output=calls or [],
        output_text=text,
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        ),
    )


def _tool_call(name: str, arguments: dict | str, call_id: str = "call_1"):
    return SimpleNamespace(
        type="function_call",
        name=name,
        arguments=(
            json.dumps(arguments, ensure_ascii=False)
            if isinstance(arguments, dict)
            else arguments
        ),
        call_id=call_id,
    )


def _agent_patches(response_mock: AsyncMock, intent: Intent = Intent.CHAT):
    return (
        patch("app.services.agent.loop.create_generation_response", response_mock),
        patch(
            "app.services.agent.loop.classify_intent",
            new_callable=AsyncMock,
            return_value=intent,
        ),
        patch(
            "app.services.agent.loop.get_model_for_intent",
            return_value="gpt-5.6-luna",
        ),
        patch(
            "app.services.agent.loop.build_soul_prompt",
            return_value="You are Wai.",
        ),
    )


def test_data_classes_and_defaults():
    message = AgentMessage(role="user", content="hello")
    context = _context()
    result = AgentResult("ok", Intent.CHAT, "gpt-5.6-luna")

    assert message.content == "hello"
    assert context.connected_services == []
    assert context.voice_transcript is None
    assert result.input_tokens == 0
    assert result.model_used == "gpt-5.6-luna"


def test_tool_definitions_convert_to_responses_api_schema():
    assert {tool["name"] for tool in TOOLS} == {
        "get_inbox",
        "clear_draft",
        "search_messages",
        "get_files",
        "get_message",
        "list_drafts",
        "save_draft",
        "prepare_media",
        "download_media",
        "get_message_content",
        "get_transcript_segments",
        "get_data_status",
        "get_digest",
        "track_commitment",
        "extract_entities",
        "list_commitments",
        "search_web",
    }
    converted = _responses_tools()
    assert all(tool["type"] == "function" for tool in converted)
    assert all(tool["parameters"]["type"] == "object" for tool in converted)


@pytest.mark.asyncio
async def test_execute_tool_dispatches_and_reports_unknown_tools():
    context = _context()
    db = AsyncMock()

    @asynccontextmanager
    async def db_context():
        yield db

    with (
        patch("app.core.database.get_db_context", side_effect=db_context),
        patch(
            "app.services.tool_registry.execute_data_tool",
            new_callable=AsyncMock,
            return_value={"results": [], "has_more": False},
        ) as execute,
    ):
        result = await execute_tool("search_messages", {"query": "test"}, context)

    assert json.loads(result) == {"results": [], "has_more": False}
    execute.assert_awaited_once_with(
        db, context.user_id, "search_messages", {"query": "test"}
    )

    assert "Unknown tool" in await execute_tool("missing", {}, context)


@pytest.mark.asyncio
async def test_plain_response_uses_luna_profile_and_reports_usage():
    generation = AsyncMock(return_value=_response("Hello!"))
    p1, p2, p3, p4 = _agent_patches(generation)
    with p1, p2, p3, p4:
        result = await run_agent(_context(), "Hello")

    assert result.response == "Hello!"
    assert result.model_used == "gpt-5.6-luna"
    assert result.input_tokens == 10
    assert result.output_tokens == 20
    kwargs = generation.await_args.kwargs
    assert kwargs["instructions"] == "You are Wai."
    assert kwargs["max_output_tokens"] == 4096
    assert len(kwargs["tools"]) == len(TOOLS)


@pytest.mark.asyncio
async def test_history_and_voice_transcript_are_passed_as_responses_input():
    generation = AsyncMock(return_value=_response("Summary"))
    p1, p2, p3, p4 = _agent_patches(generation, Intent.VOICE_SUMMARY)
    context = _context(
        conversation_history=[
            AgentMessage(role="user", content="Earlier"),
            AgentMessage(role="assistant", content="Noted"),
        ],
        has_voice=True,
        voice_transcript="Discussion about Q2 budget",
    )
    with p1, p2, p3, p4:
        await run_agent(context, "summarize this")

    input_items = generation.await_args.args[0]
    assert input_items[:2] == [
        {"role": "user", "content": "Earlier"},
        {"role": "assistant", "content": "Noted"},
    ]
    assert "Q2 budget" in input_items[-1]["content"]
    assert "summarize this" in input_items[-1]["content"]


@pytest.mark.asyncio
async def test_function_call_result_is_replayed_to_the_model():
    call = _tool_call("search_messages", {"query": "pricing"})
    generation = AsyncMock(
        side_effect=[
            _response(calls=[call], input_tokens=15, output_tokens=25),
            _response("Alex discussed pricing.", input_tokens=30, output_tokens=40),
        ]
    )
    p1, p2, p3, p4 = _agent_patches(generation, Intent.SEARCH)
    with (
        p1,
        p2,
        p3,
        p4,
        patch(
            "app.services.agent.loop.execute_tool",
            new_callable=AsyncMock,
            return_value="Found the message",
        ) as execute,
    ):
        result = await run_agent(_context(), "Find pricing")

    assert result.response == "Alex discussed pricing."
    assert result.tool_calls == 1
    assert result.input_tokens == 45
    assert result.output_tokens == 65
    execute.assert_awaited_once_with("search_messages", {"query": "pricing"}, ANY)
    second_input = generation.await_args_list[1].args[0]
    assert call in second_input
    assert second_input[-1] == {
        "type": "function_call_output",
        "call_id": "call_1",
        "output": "Found the message",
    }


@pytest.mark.asyncio
async def test_multiple_function_calls_in_one_response():
    generation = AsyncMock(
        side_effect=[
            _response(
                calls=[
                    _tool_call("search_messages", {"query": "budget"}, "call_1"),
                    _tool_call("get_digest", {"date": "2026-08-03"}, "call_2"),
                ]
            ),
            _response("Combined answer"),
        ]
    )
    p1, p2, p3, p4 = _agent_patches(generation, Intent.SEARCH)
    with (
        p1,
        p2,
        p3,
        p4,
        patch(
            "app.services.agent.loop.execute_tool",
            new_callable=AsyncMock,
            side_effect=["budget", "digest"],
        ),
    ):
        result = await run_agent(_context(), "Combine both")

    assert result.tool_calls == 2
    assert result.response == "Combined answer"


@pytest.mark.asyncio
async def test_invalid_tool_json_and_empty_generation_are_surfaced():
    invalid = AsyncMock(return_value=_response(calls=[_tool_call("search_web", "{")]))
    p1, p2, p3, p4 = _agent_patches(invalid)
    with p1, p2, p3, p4, pytest.raises(json.JSONDecodeError):
        await run_agent(_context(), "test")

    empty = AsyncMock(return_value=_response())
    p1, p2, p3, p4 = _agent_patches(empty)
    with p1, p2, p3, p4, pytest.raises(EmptyGenerationError):
        await run_agent(_context(), "test")


@pytest.mark.asyncio
async def test_turn_limit_is_surfaced():
    generation = AsyncMock(
        return_value=_response(calls=[_tool_call("search_web", {"query": "loop"})])
    )
    p1, p2, p3, p4 = _agent_patches(generation)
    with (
        p1,
        p2,
        p3,
        p4,
        patch(
            "app.services.agent.loop.execute_tool",
            new_callable=AsyncMock,
            return_value="result",
        ),
        pytest.raises(RuntimeError, match=str(MAX_TURNS)),
    ):
        await run_agent(_context(), "loop")

    assert generation.await_count == MAX_TURNS


@pytest.mark.asyncio
async def test_metrics_record_aggregated_token_usage():
    generation = AsyncMock(
        return_value=_response("Digest", input_tokens=50, output_tokens=100)
    )
    p1, p2, p3, p4 = _agent_patches(generation, Intent.DIGEST)
    calls = []
    with (
        p1,
        p2,
        p3,
        p4,
        patch(
            "app.services.agent.metrics.increment",
            side_effect=lambda metric, value=1: calls.append((metric, value)),
        ),
    ):
        await run_agent(_context(), "/digest")

    assert ("agent_tokens_input", 50) in calls
    assert ("agent_tokens_output", 100) in calls
