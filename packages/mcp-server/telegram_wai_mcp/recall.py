"""recall: the first round of looking something up in the archive, in one call.

A question comes in; a small pack of dated, sourced evidence goes out, newest
first. Plain code over the backend's existing read calls: no model, and nothing
is ever sent, saved or synced.
"""

import asyncio
import base64
import json
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from typing import Any, Awaitable, Callable

from mcp.types import Tool, ToolAnnotations

from telegram_wai_mcp.client import TelegramAIClient
from telegram_wai_mcp.links import message_urls
from telegram_wai_mcp.recall_understand import (
    Understood,
    fold,
    is_person,
    name_key,
    stem,
    understand,
    words,
)

logger = logging.getLogger(__name__)

# How long one backend call may take before recall goes on without it.
PART_SECONDS = 5.0
MAX_ITEM_CHARS = 700
MAX_TOTAL_CHARS = 7000
SUMMARY_CHARS = 300
NEIGHBOUR_CHARS = 200
SEARCH_LIMIT = 30
BROWSE_LIMIT = 200
FILE_LIMIT = 10
PER_CHAT = 2
MEANING_ONLY = 3
# One word out of many is not the subject: the words found must carry this share
# of the question, where a word the archive rarely holds counts double.
WORDING_SHARE = 0.3
OPEN_PAGE = 7
VOICE_TYPES = ("voice", "video_note", "audio")
FILE_TYPES = ("document", "video", "audio")
# Kinds of thing that are recognised by their shape, so the named chats are read
# directly instead of hoping the message also contains the word for it.
SHAPE_HINTS = frozenset({"link", "address", "phone", "amount"})

_ADDRESS = re.compile(
    r"\bул\.|\bулиц|\bпросп|\bпр-т|\bпер\.|\bпереул|\bшоссе|\bнаб\.|\bнабережн|\bбульвар"
    r"|\bплощад|\bд\.\s?\d|\bдом\s\d|\bкв\.\s?\d|\bквартир|\bподъезд|\bэтаж|\bофис\s?\d"
    r"|\bstreet\b|\bst\.|\bavenue\b|\bave\b|\broad\b|\bblvd\b|\bapt\b|\bsuite\b|\bfloor\b"
    r"|maps\.google|google\.[a-z.]+/maps|maps\.app\.goo\.gl|yandex\.[a-z]+/maps|2gis\.|maps\.apple"
)
_PHONE = re.compile(r"(?<!\d)\+?\d(?:[\s\-()]*\d){9,13}(?!\d)")
_AMOUNT = re.compile(
    r"[$€₽£]\s?\d|\d[\d\s.,]*\s?(?:₽|руб|р\b|тыс|млн|[kк]\b|\$|€|£|usd|eur|rub|долл|евро)"
)

RECALL_TOOL = Tool(
    name="recall",
    description=(
        "Start here for any question about what was written, sent or agreed in Telegram: "
        "what someone wrote about a subject, where an address, a file, a link or a voice "
        "message is, what was decided in a chat. Ask in plain Russian or English, as spoken. "
        'Dates in the question ("вчера", "last week", "с 12 по 18 октября") and names of '
        "people and chats are understood, with allowance for dictation and transliteration. "
        "One call searches message text, voice transcripts, files and links, and returns up "
        "to limit pieces of evidence, newest first, each with its chat, sender, date, a "
        "Telegram link when one exists, and the tool call that opens it in its chat. At most "
        "2 pieces come from one chat unless the question names that chat or person. "
        "Read-only: it never sends, saves or syncs anything. Use the other tools to read "
        "further around a piece of evidence, or when recall finds nothing."
    ),
    inputSchema={
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "minLength": 2,
                "maxLength": 1000,
                "description": "What to look up, in Russian or English; dictated speech is fine",
            },
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 20,
                "default": 10,
                "description": "Most pieces of evidence to return",
            },
            "since": {
                "type": "string",
                "format": "date",
                "description": "YYYY-MM-DD, first UTC day to search; replaces dates in the question",
            },
            "until": {
                "type": "string",
                "format": "date",
                "description": "YYYY-MM-DD, last UTC day to search; replaces dates in the question",
            },
        },
        "required": ["question"],
        "additionalProperties": False,
    },
    annotations=ToolAnnotations(readOnlyHint=True),
)


_FIRST_ID = "00000000-0000-0000-0000-000000000000"


@dataclass
class _Found:
    """One archived message, whichever backend call returned it."""

    id: str
    chat_id: str
    chat_title: str
    number: int
    sent_at: str
    when: datetime
    sender: str
    text: str
    media: str | None
    file_name: str | None
    summary: str
    said: str  # the transcript or extracted text the search matched
    urls: list[str]
    url: str | None
    words_hit: set[str] = field(default_factory=set)  # words an exact search found in it
    phrase_hit: bool = False  # the exact search for the phrase found it
    similarity: float = 0.0  # how high the search by meaning placed it

    def absorb(self, other: "_Found") -> None:
        self.words_hit |= other.words_hit
        self.phrase_hit = self.phrase_hit or other.phrase_hit
        self.text = self.text or other.text
        self.similarity = max(self.similarity, other.similarity)
        self.said = self.said or other.said
        self.summary = self.summary or other.summary
        self.urls = self.urls or other.urls
        self.url = self.url or other.url


@dataclass
class _Run:
    api: TelegramAIClient
    pool: dict[str, _Found] = field(default_factory=dict)
    ran: set[str] = field(default_factory=set)
    failed: set[str] = field(default_factory=set)
    rare: set[str] = field(default_factory=set)

    async def part(self, name: str, call: Awaitable[Any]) -> Any:
        """One backend call. A failure or a slow answer costs that part, not the recall."""
        self.ran.add(name)
        try:
            return await asyncio.wait_for(call, PART_SECONDS)
        except Exception as error:
            self.failed.add(name)
            # The error class only: a backend message can quote the question.
            logger.warning("recall: %s did not answer (%s)", name, type(error).__name__)
            return None

    def keep(self, found: _Found | None) -> None:
        if found is None:
            return
        known = self.pool.setdefault(found.id, found)
        if known is not found:
            known.absorb(found)

    def state(self, name: str) -> bool | str:
        return "failed" if name in self.failed else name in self.ran

    def state_of_text_search(self) -> bool | str:
        searches = {"exact", "meaning"} & self.ran
        if not searches:
            return False
        return "failed" if searches <= self.failed else True


def _found(raw: Any, chat: dict[str, Any] | None = None, **marks: Any) -> _Found | None:
    if not isinstance(raw, dict) or raw.get("deleted_at"):
        return None
    chat = chat or {}
    message_id = raw.get("id") or raw.get("message_id")
    chat_id = raw.get("chat_id") or chat.get("id")
    number = raw.get("telegram_message_id")
    sent_at = raw.get("sent_at")
    if not (message_id and chat_id and isinstance(number, int) and isinstance(sent_at, str)):
        return None
    try:
        when = datetime.fromisoformat(sent_at.replace("Z", "+00:00"))
    except ValueError:
        return None
    media = raw.get("media_type") or None
    return _Found(
        id=str(message_id),
        chat_id=str(chat_id),
        chat_title=str(raw.get("chat_title") or chat.get("title") or "Unknown"),
        number=number,
        sent_at=sent_at,
        when=when.astimezone(UTC) if when.tzinfo else when.replace(tzinfo=UTC),
        sender=str(raw.get("sender_name") or ("You" if raw.get("is_outgoing") else "Unknown")),
        text=str(raw.get("text") or raw.get("caption") or "").strip(),
        media=None if media == "other" else media,
        file_name=raw.get("media_file_name") or None,
        summary=str(raw.get("content_summary") or "").strip(),
        # For a plain message the backend may put a whole day of talk here; only a
        # recording or a document has content of its own.
        said=str(raw.get("content_preview") or "").strip() if media else "",
        urls=[url for url, _host in message_urls(raw)],
        url=raw.get("telegram_message_url") or None,
        **marks,
    )


def _cursor(sent_at: str, number: int, message_id: str = _FIRST_ID) -> str:
    """A message cursor in the form the backend's chat messages endpoint reads.

    It mirrors _encode_message_cursor in the backend's api/v1/chats.py. If that
    form changes, reading around a message fails and is reported as failed.
    """
    payload = {"id": message_id, "sent_at": sent_at, "telegram_message_id": number}
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _day_start(day: date) -> str:
    return datetime.combine(day, time.min, tzinfo=UTC).isoformat()


def _bounds(understood: Understood) -> dict[str, str]:
    bounds = {}
    if understood.since:
        bounds["date_from"] = _day_start(understood.since)
    if understood.until:
        bounds["date_to"] = datetime.combine(understood.until, time.max, tzinfo=UTC).isoformat()
    return bounds


def _exact_queries(understood: Understood) -> list[tuple[str, str | None]]:
    """The wording to look for as written: the phrase, then the longest words.

    The index keeps words as they were typed, so a word is also tried without
    its case ending - the form it most often has in a message.
    """
    queries: dict[str, str | None] = {}
    if understood.phrase:
        queries[understood.phrase] = None
    longest = sorted(understood.terms, key=len, reverse=True)[:3]
    for term in longest:
        queries.setdefault(term, term)
    for term in longest:
        root = stem(fold(term))
        if len(queries) < 5 and len(root) >= 4:
            queries.setdefault(root, term)
    return list(queries.items())


async def _search(
    run: _Run,
    understood: Understood,
    mode: str,
    query: str,
    chat_ids: list[str] | None,
    term: str | None = None,
) -> None:
    arguments: dict[str, Any] = {
        "query": query,
        "mode": mode,
        "limit": SEARCH_LIMIT,
        **_bounds(understood),
    }
    if chat_ids:
        arguments["chat_ids"] = chat_ids
    exact = mode == "exact"
    result = await run.part(
        "exact" if exact else "meaning",
        run.api.execute_data_tool("search_messages", arguments),
    )
    rows = (result or {}).get("results") or []
    for raw in rows:
        if exact:
            # The match may sit deep in a transcript, past the part the search returns.
            marks: dict[str, Any] = {"words_hit": {term}} if term else {"phrase_hit": True}
        else:
            marks = {"similarity": float((raw or {}).get("similarity") or 0)}
        run.keep(_found(raw, **marks))
    # A word the whole archive holds only a few times says more than a common one.
    if exact and term and not chat_ids and rows and not result.get("has_more"):
        run.rare.add(term)


async def _browse(run: _Run, understood: Understood, chat: dict[str, Any]) -> None:
    """Read the newest messages of a named chat inside the window, as they are."""
    before = None
    if understood.until:
        before = _cursor(_day_start(understood.until + timedelta(days=1)), 0)
    page = await run.part(
        "browse",
        run.api.get_messages(chat_id=str(chat["id"]), limit=BROWSE_LIMIT, before=before),
    )
    for raw in (page or {}).get("messages") or []:
        found = _found(raw, chat)
        if found and not (understood.since and found.when.date() < understood.since):
            run.keep(found)


async def _files(run: _Run, understood: Understood, part: str, media_types: list[str]) -> None:
    arguments: dict[str, Any] = {
        "media_types": media_types,
        "limit": FILE_LIMIT,
        **_bounds(understood),
    }
    if understood.who:
        arguments["chat_ids"] = [str(chat["id"]) for chat in understood.who]
    query = " ".join(understood.terms)
    if len(query) >= 2:
        arguments["query"] = query
    result = await run.part(part, run.api.execute_data_tool("get_files", arguments))
    for raw in (result or {}).get("files") or []:
        run.keep(_found(raw))


def _kind(found: _Found, hints: frozenset[str]) -> str:
    if found.media in VOICE_TYPES:
        return "voice"
    if found.media:
        return "file"
    return "link" if found.urls and "link" in hints else "message"


def _is_asked_kind(found: _Found, hints: frozenset[str]) -> bool:
    text = fold(f"{found.text} {found.said if found.media in VOICE_TYPES else ''}")
    return bool(
        ("voice" in hints and found.media in VOICE_TYPES)
        or ("photo" in hints and found.media == "photo")
        or ("file" in hints and found.media and found.media not in ("photo", "voice", "video_note"))
        or ("link" in hints and found.urls)
        or ("address" in hints and _ADDRESS.search(f"{text} {' '.join(found.urls)}"))
        or ("phone" in hints and _PHONE.search(text))
        or ("amount" in hints and _AMOUNT.search(text))
    )


@dataclass
class _Judged:
    found: _Found
    supported: bool  # the wording, the kind of thing or the named place fits
    near: bool  # in a named chat, or sent by a named person
    score: float
    by_summary: bool  # a file that fits only through its long summary


def _judge(
    found: _Found,
    understood: Understood,
    rare: set[str],
    named: set[str],
    people: list[set[str]],
) -> _Judged:
    own = words(f"{found.text} {found.file_name or ''} {found.sender}")
    whole = own + words(f"{found.summary} {found.said}") if found.media else own
    own_stems = {stem(word) for word in own}
    whole_stems = {stem(word) for word in whole}
    phrase = " ".join(words(understood.phrase or ""))
    has_phrase = found.phrase_hit or (bool(phrase) and phrase in " ".join(whole))
    weight = {term: 2 if term in rare else 1 for term in understood.terms}
    matched = [
        term
        for term in understood.terms
        if term in found.words_hit or stem(fold(term)) in whole_stems
    ]
    share = sum(weight[term] for term in matched) / sum(weight.values()) if weight else 0.0
    asked_kind = _is_asked_kind(found, understood.hints)
    has_subject = bool(understood.terms or understood.phrase)
    sender = {name_key(word) for word in words(found.sender)}
    near = found.chat_id in named or any(person <= sender for person in people)
    if has_subject:
        supported = has_phrase or share >= WORDING_SHARE or asked_kind
    else:
        # No subject to search for. A question about a kind of thing ("the link
        # from Марта") is answered by things of that kind; any other, by what the
        # named chat or the window holds.
        in_scope = near or bool(understood.since or understood.until)
        supported = asked_kind if understood.hints else in_scope
    in_own_words = any(stem(fold(term)) in own_stems for term in understood.terms) or (
        bool(phrase) and phrase in " ".join(own)
    )
    return _Judged(
        found=found,
        supported=supported,
        near=near,
        score=3 * has_phrase + 2 * share + 2 * asked_kind + min(found.similarity, 1.0),
        by_summary=bool(
            found.media
            and found.media not in VOICE_TYPES
            and has_subject
            and not in_own_words
            and not understood.hints & {"file", "photo"}
        ),
    )


def _select(run: _Run, understood: Understood, limit: int) -> list[_Judged]:
    named = {str(chat["id"]) for chat in understood.who}
    # A named person is also followed into group chats, by the name on the message.
    people = [
        person
        for chat in understood.who
        if is_person(chat)
        and (
            person := {
                name_key(word) for word in words(str(chat.get("title") or "")) if len(word) >= 3
            }
        )
    ]
    judged = [_judge(found, understood, run.rare, named, people) for found in run.pool.values()]
    # The wording fits before only the meaning does; a named place before the rest
    # of the archive; a better fit before a worse; and then the newest.
    judged.sort(
        key=lambda item: (
            item.supported,
            item.near,
            int(item.score) if item.supported else item.found.similarity,
            item.found.when,
        ),
        reverse=True,
    )
    found_near = any(item.supported and item.near for item in judged)
    chosen: list[_Judged] = []
    per_chat: Counter[str] = Counter()
    by_meaning = by_summary = 0
    for item in judged:
        if len(chosen) == limit:
            break
        found = item.found
        if not item.supported and (found.similarity <= 0 or by_meaning == MEANING_ONLY):
            continue
        # The rest of the archive answers a question about a named chat or person
        # only when that chat and that person's messages hold nothing.
        if named and found_near and not item.near:
            continue
        if found.chat_id not in named and per_chat[found.chat_id] == PER_CHAT:
            continue
        if item.by_summary and by_summary:
            continue
        per_chat[found.chat_id] += 1
        by_meaning += not item.supported
        by_summary += item.by_summary
        chosen.append(item)
    chosen.sort(key=lambda item: item.found.when, reverse=True)
    return chosen


def _clip(text: str, limit: int) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _around(text: str, understood: Understood, limit: int) -> str:
    """A long text cut to the part where the question's words are, not just its start."""
    text = text.strip()
    if len(text) <= limit:
        return text
    low = fold(text)
    sought = [fold(understood.phrase or ""), *(stem(fold(term)) for term in understood.terms)]
    spots = [spot for word in sought if word and (spot := low.find(word)) >= 0]
    start = max(0, min(spots) - limit // 4) if spots else 0
    return ("…" if start else "") + _clip(text[start:], limit - bool(start))


def _own_text(found: _Found, kind: str, understood: Understood, cap: int) -> str:
    if kind == "voice":
        head = " ".join(part for part in ("[voice]", found.text) if part)
        said = found.said or found.summary or "(no transcript yet)"
        return f"{head} {_around(said, understood, max(cap - len(head) - 1, 100))}"
    if kind == "file":
        label = f"[file: {found.file_name}]" if found.file_name else f"[{found.media}]"
        detail = _around(found.summary or found.said, understood, SUMMARY_CHARS)
        if detail and found.text:
            detail = f"— {detail}"
        return " ".join(part for part in (label, found.text, detail) if part)
    hidden = [url for url in found.urls if url not in found.text]
    return _around(
        "\n".join(part for part in (found.text, " ".join(hidden)) if part), understood, cap
    )


def _neighbour_line(raw: Any, found: _Found, shown: set[str]) -> str | None:
    """A message next to the evidence, when it is short and part of the same exchange."""
    near = _found(raw, {"id": found.chat_id})
    if near is None or near.id in shown or not near.text or len(near.text) > NEIGHBOUR_CHARS:
        return None
    if abs(near.when - found.when) > timedelta(days=1):
        return None
    return f"{near.sender}: {near.text}"


async def _write(
    run: _Run, item: _Judged, number: int, cap: int, understood: Understood, shown: set[str]
) -> dict[str, Any]:
    """One piece of evidence: the message, its short neighbours, and how to open it."""
    found = item.found
    kind = _kind(found, understood.hints)
    if item.by_summary:
        cap = min(SUMMARY_CHARS, cap)
    text = _clip(_own_text(found, kind, understood, cap), cap)
    open_at = _cursor(found.sent_at, found.number + 1)
    marked = f"» {found.sender}: {text}"
    if len(marked) + 20 < cap:
        here = _cursor(found.sent_at, found.number, found.id)
        earlier, later = await asyncio.gather(
            run.part("context", run.api.get_messages(chat_id=found.chat_id, limit=1, before=here)),
            run.part("context", run.api.get_messages(chat_id=found.chat_id, limit=1, after=here)),
        )
        before = next(iter((earlier or {}).get("messages") or []), None)
        after = next(iter((later or {}).get("messages") or []), None)
        if isinstance(after, dict) and isinstance(after.get("telegram_message_id"), int):
            # Start the page one message later, so the reply is on it too.
            open_at = _cursor(str(after.get("sent_at")), after["telegram_message_id"] + 1)
        lines = [marked]
        room = cap - len(marked)
        said_before = _neighbour_line(before, found, shown)
        if said_before and len(said_before) + 1 <= room:
            lines.insert(0, said_before)
            room -= len(said_before) + 1
        said_after = _neighbour_line(after, found, shown)
        if said_after and len(said_after) + 1 <= room:
            lines.append(said_after)
        if len(lines) > 1:
            text = "\n".join(lines)
    return {
        "n": number,
        "kind": kind,
        "layer": "raw",
        "title": f"{found.chat_title} · {found.sender}",
        "date": found.when.date().isoformat(),
        "text": text,
        "id": found.id,
        "url": found.url,
        "open": {
            "tool": "get_chat_messages",
            "args": {"chat_id": found.chat_id, "before": open_at, "limit": OPEN_PAGE},
        },
    }


_PLAIN_PART = {
    "names": "the list of chats",
    "exact": "the search by wording",
    "meaning": "the search by meaning",
    "browse": "reading the named chats",
    "files": "the list of files",
    "voice": "the list of voice messages",
    "context": "the neighbouring messages",
}
_SEARCHED_HOW = {
    "exact": "by wording",
    "meaning": "by meaning",
    "browse": "by reading the named chats",
    "files": "in the list of files",
    "voice": "among voice messages",
}


def _plural(count: int, noun: str) -> str:
    return f"{count:,} {noun}" if count == 1 else f"{count:,} {noun}s"


def _listed(items: list[str]) -> str:
    return items[0] if len(items) == 1 else f"{', '.join(items[:-1])} and {items[-1]}"


def _note(
    run: _Run, understood: Understood, chosen: list[_Judged], chats: int, messages: int
) -> str:
    """One plain sentence: what was found, or what was searched to find nothing."""
    if understood.since and understood.until:
        window = f" from {understood.since} to {understood.until}"
    elif understood.since:
        window = f" since {understood.since}"
    elif understood.until:
        window = f" until {understood.until}"
    else:
        window = ""
    silent = [_PLAIN_PART[name] for name in _PLAIN_PART if name in run.failed]
    how = [text for name, text in _SEARCHED_HOW.items() if run.state(name) is True]
    if chosen:
        places = len({item.found.chat_id for item in chosen})
        note = (
            f"Found {_plural(len(chosen), 'message')} in {_plural(places, 'chat')}{window}, "
            "newest first"
        )
        by_meaning = sum(1 for item in chosen if not item.supported)
        if by_meaning:
            note += f"; {by_meaning} of them fit by meaning only, not by wording"
        if understood.who and not any(item.near for item in chosen):
            note += "; none is from the chat or person the question names"
    elif not how:
        return f"The archive could not be searched{window}: {_listed(silent)} did not answer."
    else:
        note = (
            f"Nothing in the archive matches: searched {_plural(messages, 'message')} in "
            f"{_plural(chats, 'chat')}{window} {_listed(how)}"
        )
    if silent:
        note += f"; {_listed(silent)} did not answer"
    return note + "."


async def recall(
    api: TelegramAIClient,
    *,
    question: str,
    inventory: Callable[[TelegramAIClient], Awaitable[dict[str, Any]]],
    limit: int = 10,
    since: date | None = None,
    until: date | None = None,
    today: date | None = None,
) -> dict[str, Any]:
    """Understand the question, retrieve from every index at once, assemble the pack."""
    run = _Run(api)
    listing = await run.part("names", inventory(api))
    chats = [chat for chat in (listing or {}).get("chats") or [] if isinstance(chat, dict)]
    understood = understand(
        question,
        chats,
        today=today or datetime.now(UTC).date(),
        since=since,
        until=until,
    )

    named = [str(chat["id"]) for chat in understood.who]
    has_subject = bool(understood.terms or understood.phrase)
    wording = _exact_queries(understood)
    meaning = " ".join(understood.terms) or understood.phrase or question
    whole_archive = has_subject or not named
    parts: list[Awaitable[None]] = []
    if named and has_subject:
        parts.append(_search(run, understood, "hybrid", meaning, named))
        parts += [_search(run, understood, "exact", query, named, term) for query, term in wording]
    if named and (not has_subject or understood.hints & SHAPE_HINTS):
        parts += [_browse(run, understood, chat) for chat in understood.who]
    if whole_archive:
        parts.append(_search(run, understood, "hybrid", meaning, None))
        parts += [_search(run, understood, "exact", query, None, term) for query, term in wording]
    wanted_files = [
        *(FILE_TYPES if "file" in understood.hints else ()),
        *(("photo",) if "photo" in understood.hints else ()),
    ]
    if wanted_files:
        parts.append(_files(run, understood, "files", wanted_files))
    if "voice" in understood.hints:
        parts.append(_files(run, understood, "voice", list(VOICE_TYPES)))
    await asyncio.gather(*parts)

    chosen = _select(run, understood, limit)
    cap = min(MAX_ITEM_CHARS, MAX_TOTAL_CHARS // max(len(chosen), 1))
    shown = {item.found.id for item in chosen}
    evidence = await asyncio.gather(
        *(
            _write(run, item, number, cap, understood, shown)
            for number, item in enumerate(chosen, 1)
        )
    )

    if chats:
        scope = (
            [chat for chat in chats if chat.get("indexing_enabled", True)]
            if whole_archive
            else understood.who
        )
        chat_count = len(scope)
        message_count = sum(int(chat.get("total_messages_synced") or 0) for chat in scope)
    else:
        # Without the list of chats, count what the searches actually returned.
        chat_count = len({found.chat_id for found in run.pool.values()})
        message_count = len(run.pool)
    text_search = run.state_of_text_search()
    return {
        "understood": {
            "from": understood.since.isoformat() if understood.since else None,
            "to": understood.until.isoformat() if understood.until else None,
            "who": [
                {
                    "kind": "person" if is_person(chat) else "chat",
                    "id": str(chat["id"]),
                    "name": str(chat.get("title") or ""),
                }
                for chat in understood.who
            ],
            "about": understood.about,
        },
        "evidence": list(evidence),
        "searched": {
            "chats": chat_count,
            "messages": message_count,
            # Transcripts are part of the message index; a question that asks for
            # a voice message also reads the list of voice messages.
            "voice": run.state("voice") if "voice" in run.ran else text_search,
            "names": run.state("names"),
            "exact": run.state("exact"),
            "meaning": run.state("meaning"),
            "browse": run.state("browse"),
            "files": run.state("files"),
            "links": (
                False
                if "link" not in understood.hints
                else run.state("browse")
                if "browse" in run.ran
                else text_search
            ),
            "context": run.state("context"),
        },
        "note": _note(run, understood, chosen, chat_count, message_count),
    }
