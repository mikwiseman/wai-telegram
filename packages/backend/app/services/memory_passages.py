"""Bounded, verbatim passages. Originals and access rights stay in Telegram."""

import base64
import hashlib
import json
import math
import struct
from dataclasses import dataclass
from functools import lru_cache
from uuid import UUID

import tiktoken

MODEL = "text-embedding-3-large"
DIMENSIONS = 3072
MAX_TOKENS = 1400
VERSION = 1


@lru_cache(maxsize=1)
def tokenizer():
    return tiktoken.get_encoding("cl100k_base")


def token_count(value: str) -> int:
    return len(tokenizer().encode(value, disallowed_special=()))


@dataclass(frozen=True)
class Passage:
    id: str
    text: str
    source_text: str
    message_ids: list[UUID]
    tokens: int


def _pieces(value: str, budget: int):
    """Split at character boundaries, keeping whitespace and Unicode intact."""
    if budget < 1:
        raise ValueError("Passage token budget must be positive")
    while value:
        high = min(len(value), budget * 12)
        if token_count(value[:high]) <= budget:
            yield value[:high]
            value = value[high:]
            continue
        low = 1
        while low < high:
            middle = (low + high + 1) // 2
            if token_count(value[:middle]) <= budget:
                low = middle
            else:
                high = middle - 1
        boundary = max(value.rfind("\n", 0, low), value.rfind(" ", 0, low)) + 1
        end = boundary if boundary >= low // 2 else low
        yield value[:end]
        value = value[end:]


def _passage(text: str, sources: list[str], ids: list[UUID]) -> Passage:
    unique_ids = list(dict.fromkeys(ids))
    identity = json.dumps(
        [VERSION, MODEL, DIMENSIONS, list(map(str, unique_ids)), text],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return Passage(
        hashlib.sha256(identity.encode()).hexdigest(),
        text,
        "".join(sources),
        unique_ids,
        token_count(text),
    )


def build_passages(chat_title: str, messages: list[dict]) -> list[Passage]:
    result: list[Passage] = []
    groups: dict[int | None, list[dict]] = {}
    for message in messages:
        if not message.get("deleted_at"):
            groups.setdefault(message.get("thread_id"), []).append(message)
    for thread, rows in groups.items():
        title = f"Chat: {chat_title[:160]}" + (f" · thread {thread}" if thread else "")
        text, sources, ids = title, [], []

        def flush():
            nonlocal text, sources, ids
            if ids:
                result.append(_passage(text, sources, ids))
            text, sources, ids = title, [], []

        for row in rows:
            stamp = row["sent_at"].isoformat(timespec="minutes")
            name = (row.get("sender_name") or "Unknown speaker")[:120]
            prefix = f"\n[{stamp}] {name}: "
            body = (row.get("text") or "").strip()
            if not body:
                body = (
                    row.get("content_summary") or row.get("media_file_name") or ""
                ).strip()
            if body:
                budget = MAX_TOKENS - token_count(title + prefix) - 8
                for piece in _pieces(body, budget):
                    line = prefix + piece
                    if token_count(text + line) > MAX_TOKENS:
                        flush()
                    text += line
                    sources.append(piece)
                    ids.append(row["id"])
            content = row.get("content_text")
            if content and content.strip() and content != body:
                flush()
                header = (
                    title
                    + prefix
                    + f"[extracted {row.get('media_file_name') or 'content'}]\n"
                )
                # Filenames are data too, but cannot consume the full context.
                header = next(_pieces(header, MAX_TOKENS // 3))
                for piece in _pieces(content, MAX_TOKENS - token_count(header) - 8):
                    result.append(_passage(header + piece, [piece], [row["id"]]))
        flush()
    return result


def decode_embedding(value: str | list[float]) -> list[float]:
    if isinstance(value, str):
        raw = base64.b64decode(value, validate=True)
        if len(raw) != DIMENSIONS * 4:
            raise ValueError("Embedding dimension mismatch")
        vector = list(struct.unpack(f"<{DIMENSIONS}f", raw))
    else:
        vector = value
    if len(vector) != DIMENSIONS:
        raise ValueError("Embedding dimension mismatch")
    if not all(math.isfinite(x) for x in vector):
        raise ValueError("Embedding values must be finite")
    return vector
