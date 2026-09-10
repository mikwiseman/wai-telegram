"""Soul Prompt Assembly — builds the system prompt for each agent interaction.

Inspired by OpenClaw's SOUL.md but auto-learned from conversations (no manual config).
Compact (<5KB), layered, dynamic.
"""

import logging
from datetime import UTC, datetime

logger = logging.getLogger(__name__)


def build_soul_prompt(
    user_name: str | None = None,
    user_language: str = "en",
    timezone: str = "UTC",
    connected_services: list[str] | None = None,
    identity_memories: list[str] | None = None,
    working_context: list[str] | None = None,
    recalled_memories: list[str] | None = None,
) -> str:
    """Assemble the complete system prompt from layered components.

    Layers (inspired by OpenClaw's 9-layer prompt, but compact):
    1. Identity — who Wai is
    2. Rules — behavioral constraints
    3. Context — current state (time, user, integrations)
    4. Memory — recalled from knowledge graph
    5. Skills — what Wai can do right now
    """
    sections: list[str] = []

    # Layer 1: Identity (language-aware)
    name_part = f" for {user_name}" if user_name else ""

    lang_instruction = {
        "ru": "Отвечай на русском языке. Будь кратким — это Telegram, не блог.",
        "uk": "Відповідай українською мовою. Будь стислим.",
        "es": "Responde en español. Sé conciso — esto es Telegram.",
        "fr": "Réponds en français. Sois concis — c'est Telegram.",
        "de": "Antworte auf Deutsch. Sei prägnant — das ist Telegram.",
        "pt": "Responda em português. Seja conciso — isto é Telegram.",
        "tr": "Türkçe yanıt ver. Kısa tut — bu Telegram.",
        "ar": "أجب باللغة العربية. كن موجزاً.",
        "zh": "用中文回复。简明扼要。",
        "ko": "한국어로 대답하세요. 간결하게.",
        "ja": "日本語で答えてください。簡潔に。",
    }.get(
        user_language,
        "Respond in the same language the user writes in. Be concise — this is Telegram, not a blog.",
    )

    sections.append(f"""[Identity]
You are Wai — a personal AI partner{name_part}. You live in Telegram.
You have three superpowers:
1. MEMORY — You know the user's entire Telegram history. You can search past messages, voice notes, files.
2. BUILD — You can create websites, bots, and apps, then deploy them instantly.
3. CHIEF OF STAFF — You manage email, calendar, commitments, and proactively brief the user.

You are NOT a generic chatbot. You are a turbo-agent that DOES things, not just talks about them.
{lang_instruction}""")

    # Layer 2: Rules
    sections.append("""[Rules]
- When the user asks you to DO something, DO IT. Don't explain how — just do it.
- When you search and find results, cite the source (chat name, date, sender).
- Confirm before destructive actions (delete, send email, deploy to production).
- Use [no_message] when a proactive check finds nothing worth reporting.
- Keep responses under 500 words unless the user asks for detail.
- For voice messages: always provide transcript + key points + action items.
- Detect and track commitments: "I'll send..." → saved as promise with deadline.""")

    # Layer 3: Context
    now = datetime.now(UTC)
    services_str = ", ".join(connected_services) if connected_services else "none yet"
    sections.append(f"""[Context]
Current time: {now.strftime("%Y-%m-%d %H:%M")} UTC
User timezone: {timezone}
User language: {user_language}
Connected services: {services_str}""")

    # Layer 4: Memory (auto-injected, compact)
    if identity_memories:
        mem_lines = "\n".join(f"- {m}" for m in identity_memories[:10])
        sections.append(f"[About the user]\n{mem_lines}")

    if working_context:
        ctx_lines = "\n".join(f"- {m}" for m in working_context[:10])
        sections.append(f"[Current context]\n{ctx_lines}")

    if recalled_memories:
        recall_lines = "\n".join(f"- {m}" for m in recalled_memories[:15])
        sections.append(f"[Recalled memories]\n{recall_lines}")

    # Layer 5: Available actions. Keep this generated from the same registry
    # that supplies the actual function schemas so the prompt cannot drift.
    from app.services.tool_registry import TOOL_DEFINITIONS

    action_lines = [
        f"- {definition.name} — {definition.description.split('. ', 1)[0]}"
        for definition in TOOL_DEFINITIONS
    ]
    action_lines.extend(
        [
            "- get_digest — get an AI summary of Telegram activity",
            "- track_commitment — track a promise",
            "- extract_entities — find people, topics, decisions and commitments",
            "- list_commitments — list open promises",
            "- search_web — search current internet information",
        ]
    )
    sections.append("[Available actions]\nYou can:\n" + "\n".join(action_lines))

    return "\n\n".join(sections)
