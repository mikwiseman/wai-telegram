"""recall: the first round of looking something up in the archive, in one call.

A question comes in; a small pack of dated, sourced evidence goes out, newest
first. Plain code over the backend's existing read calls: no model, and nothing
is ever sent, saved or synced.

What the backend can do in a few seconds decides how recall looks (measured on
an archive of 5,000 chats and 2.8 million messages):

- The list of chats takes seconds to page through, so it is kept per API key
  and refreshed behind the calls.
- A search inside one chat or one week is quick. A search for one common word
  across the whole archive is not (tens of seconds), and a call that is cut off
  keeps running on the backend. So words are searched as written only inside
  the named chats, inside the question's window, or inside the last 90 days.
- The search by meaning returns almost nothing from a small chat or a single
  day. So a named chat is read directly, newest first, and for a window of a
  few days the chats that were active then are read; the question's words are
  then looked for in what was read, in any of their forms.
- A long chat cannot be read back far in that time. So where a search found a
  piece that no reading reached, the talk around it is read in a second, short
  round: a screenshot or a link sent there has no words of its own to be found by.
"""

import asyncio
import base64
import json
import logging
import math
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import Any, Awaitable, Callable

from mcp.types import Tool, ToolAnnotations

from telegram_wai_mcp.client import TelegramAIClient
from telegram_wai_mcp.links import message_urls
from telegram_wai_mcp.recall_understand import (
    Understood,
    fold,
    is_person,
    name_key,
    same_word,
    stem,
    understand,
    words,
)

logger = logging.getLogger(__name__)

# How long one part of the retrieval may take before recall goes on without it,
# and how long reading around one piece of evidence may take. With the list of
# chats in memory a whole recall stays under about eight seconds.
PART_SECONDS = 5.0
CONTEXT_SECONDS = 2.0
CALLS_AT_ONCE = 8
# The list of chats: how long a copy serves, and how long the first call waits for it.
CHAT_LIST_FRESH_SECONDS = 300.0
CHAT_LIST_FIRST_LOAD_SECONDS = 20.0
CHAT_LIST_PAGE = 1000

MAX_ITEM_CHARS = 700
MAX_TOTAL_CHARS = 7000
SUMMARY_CHARS = 300
NEIGHBOUR_CHARS = 200
PER_CHAT = 2
MEANING_ONLY = 3
# Across the whole archive one word out of many is not the subject: the words
# found must carry this share of the weight of the question's words, where a
# word weighs more the fewer messages hold it.
WORDING_SHARE = 0.3
MAX_TERMS = 12
# A thing sent within this time of talk that carries half the subject belongs to it.
TALK_SPAN = timedelta(minutes=30)
TALK_LEAD = 1.0
OPEN_PAGE = 7
# The talk around a piece found only by a search: this many pieces, this many
# messages on each side of each.
AROUND_READS = 3
AROUND_PAGE = 40

SEARCH_LIMIT = 30
# Inside a chat the search by meaning is asked for few: three times as many take twice as long.
MEANING_LIMIT = 9
# Across the archive too: thirty took two to five seconds alone and often ran
# out of time beside the other parts; fifteen take about a second less.
ARCHIVE_MEANING_LIMIT = 15
FILE_LIMIT = 30
# A named chat is read newest first: this many messages a page, this many pages.
PAGE = 500
PAGES_ONE_CHAT = 4
PAGES_IN_ALL = 6
# Reading stops here and keeps what came back: under the time of a part.
READ_SECONDS = 4.0
# A window this short is read chat by chat: the chats active then, a page each.
NARROW_DAYS = 7
ACTIVE_CHATS = 60
ACTIVE_PAGE = 100
# With no window in the question, how far back a search is still quick.
MEANING_DAYS = 365
WORDING_DAYS = 90

VOICE_TYPES = ("voice", "video_note", "audio")
FILE_TYPES = ("document", "video", "audio")

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

_LIST_LINE = re.compile(r"(?m)^\s*(?:\d{1,2}[.)]|[-•–—*])\s+\S")
_LIST_INLINE = re.compile(r"(?s)\b1[.)]\s.+\b2[.)]\s.+\b3[.)]\s")
_APP_LINK = re.compile(
    r"apps\.apple\.com|testflight\.apple\.com|play\.google\.com|appgallery\.huawei|rustore\.ru"
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
    outgoing: bool
    text: str
    media: str | None
    file_name: str | None
    summary: str
    said: str  # the transcript, or the text read out of a file or a picture
    urls: list[str]
    url: str | None
    chat_url: str | None = None
    # The stretch of its chat it was read in, so its neighbours here are its
    # neighbours there; 0 when only a search found it.
    read: int = 0
    words_hit: set[str] = field(default_factory=set)  # words an exact search found in it
    phrase_hit: bool = False  # an exact search for a phrase found it
    similarity: float = 0.0  # how high the search by meaning placed it

    def absorb(self, other: "_Found") -> None:
        self.read = self.read or other.read
        self.words_hit |= other.words_hit
        self.phrase_hit = self.phrase_hit or other.phrase_hit
        self.text = self.text or other.text
        self.similarity = max(self.similarity, other.similarity)
        self.said = self.said or other.said
        self.summary = self.summary or other.summary
        self.urls = self.urls or other.urls
        self.url = self.url or other.url
        self.chat_url = self.chat_url or other.chat_url


def _moment(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return moment.astimezone(UTC) if moment.tzinfo else moment.replace(tzinfo=UTC)


def _found(raw: Any, chat: dict[str, Any] | None = None, **marks: Any) -> _Found | None:
    if not isinstance(raw, dict) or raw.get("deleted_at"):
        return None
    chat = chat or {}
    message_id = raw.get("id") or raw.get("message_id")
    chat_id = raw.get("chat_id") or chat.get("id")
    number = raw.get("telegram_message_id")
    when = _moment(raw.get("sent_at"))
    if not (message_id and chat_id and isinstance(number, int) and when):
        return None
    media = raw.get("media_type") or None
    return _Found(
        id=str(message_id),
        chat_id=str(chat_id),
        chat_title=str(raw.get("chat_title") or chat.get("title") or "Unknown"),
        number=number,
        sent_at=raw["sent_at"],
        when=when,
        sender=str(raw.get("sender_name") or ("You" if raw.get("is_outgoing") else "Unknown")),
        outgoing=bool(raw.get("is_outgoing")),
        text=str(raw.get("text") or raw.get("caption") or "").strip(),
        # "other" is a link preview, a poll or a place: a message, not a file.
        media=None if media == "other" else media,
        file_name=raw.get("media_file_name") or None,
        summary=str(raw.get("content_summary") or "").strip(),
        # For a plain message the backend may put a whole day of talk here; only a
        # recording, a document or a picture has content of its own.
        said=str(raw.get("content_preview") or "").strip() if media else "",
        urls=[url for url, _host in message_urls(raw)],
        url=raw.get("telegram_message_url") or None,
        chat_url=raw.get("telegram_chat_url") or None,
        **marks,
    )


def _cursor(sent_at: str, number: int, message_id: str = _FIRST_ID) -> str:
    """A message cursor in the form the backend's chat messages endpoint reads.

    It mirrors _encode_message_cursor in the backend's api/v1/chats.py. If that
    form changes, reading a chat fails and is reported as failed.
    """
    payload = {"id": message_id, "sent_at": sent_at, "telegram_message_id": number}
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _day_start(day: date) -> datetime:
    return datetime(day.year, day.month, day.day, tzinfo=UTC)


def _bounds(since: date | None, until: date | None) -> dict[str, str]:
    bounds = {}
    if since:
        bounds["date_from"] = _day_start(since).isoformat()
    if until:
        last_moment = _day_start(until) + timedelta(days=1) - timedelta(microseconds=1)
        bounds["date_to"] = last_moment.isoformat()
    return bounds


# --- the list of chats -----------------------------------------------------------

_CHAT_FIELDS = (
    "id",
    "title",
    "chat_type",
    "username",
    "last_activity_at",
    "total_messages_synced",
    "indexing_enabled",
)


@dataclass
class _ChatList:
    chats: list[dict[str, Any]] = field(default_factory=list)
    loaded: float | None = None  # when the last full load ended, on the monotonic clock
    loading: asyncio.Task[None] | None = None


# One list per backend and API key, for the life of the server process.
_chat_lists: dict[tuple[str, str], _ChatList] = {}


async def _load_chat_list(entry: _ChatList, open_client: Callable[[], TelegramAIClient]) -> None:
    # Its own client: the call that started the load may be long over when it ends.
    api = open_client()
    try:
        chats: list[dict[str, Any]] = []
        cursor = None
        while True:
            page = await api.list_chats(limit=CHAT_LIST_PAGE, cursor=cursor)
            chats += [
                {key: chat[key] for key in _CHAT_FIELDS if key in chat}
                for chat in page.get("chats") or []
                if isinstance(chat, dict)
            ]
            cursor = page.get("next_cursor")
            if not cursor:
                break
        entry.chats, entry.loaded = chats, time.monotonic()
    except Exception as error:
        logger.warning("recall: the list of chats did not load (%s)", type(error).__name__)
    finally:
        await api.close()


async def chat_list(
    api: TelegramAIClient, open_client: Callable[[], TelegramAIClient]
) -> list[dict[str, Any]] | None:
    """Every chat of this API key, from memory.

    Paging through thousands of chats takes longer than a recall may, so the
    list is loaded once and then refreshed behind the calls: a call takes the
    copy that is there, even a stale one, and never waits for the refresh.
    Only the very first call waits, and a load it gives up on still finishes.
    """
    key = (str(getattr(api, "base_url", "")), str(getattr(api, "api_key", "")))
    entry = _chat_lists.setdefault(key, _ChatList())
    fresh = entry.loaded is not None and time.monotonic() - entry.loaded < CHAT_LIST_FRESH_SECONDS
    if not fresh and (entry.loading is None or entry.loading.done()):
        entry.loading = asyncio.create_task(_load_chat_list(entry, open_client))
    if entry.loaded is None and entry.loading is not None:
        try:
            await asyncio.wait_for(asyncio.shield(entry.loading), CHAT_LIST_FIRST_LOAD_SECONDS)
        except TimeoutError:
            pass
    return entry.chats if entry.loaded is not None else None


# --- retrieval ---------------------------------------------------------------------


@dataclass
class _Run:
    api: TelegramAIClient
    pool: dict[str, _Found] = field(default_factory=dict)
    ran: set[str] = field(default_factory=set)
    failed: set[str] = field(default_factory=set)
    gate: asyncio.Semaphore = field(default_factory=lambda: asyncio.Semaphore(CALLS_AT_ONCE))
    stretches: int = 0

    def stretch(self) -> int:
        """A new number for one stretch of a chat read in order."""
        self.stretches += 1
        return self.stretches

    async def part(self, name: str, work: Awaitable[Any], seconds: float | None = None) -> Any:
        """One part of the work. A failure or a slow answer costs that part, not the recall."""
        self.ran.add(name)
        try:
            return await asyncio.wait_for(work, PART_SECONDS if seconds is None else seconds)
        except Exception as error:
            self.failed.add(name)
            # The error class only: a backend message can quote the question.
            logger.warning("recall: %s did not answer (%s)", name, type(error).__name__)
            return None

    async def call(self, method: Callable[..., Awaitable[Any]], *args: Any, **kwargs: Any) -> Any:
        """One backend call, with only a few running at a time."""
        async with self.gate:
            return await method(*args, **kwargs)

    def keep(self, found: _Found | None) -> None:
        if found is None:
            return
        known = self.pool.setdefault(found.id, found)
        if known is not found:
            known.absorb(found)

    def state(self, name: str) -> bool | str:
        return "failed" if name in self.failed else name in self.ran

    def state_of_text_search(self) -> bool | str:
        searches = {"exact", "meaning", "browse"} & self.ran
        if not searches:
            return False
        return "failed" if searches <= self.failed else True


def _inside(found: _Found, since: date | None, until: date | None) -> bool:
    day = found.when.date()
    return not (since and day < since) and not (until and day > until)


async def _search(
    run: _Run,
    mode: str,
    query: str,
    *,
    since: date | None,
    until: date | None,
    chat_ids: list[str] | None = None,
    term: str | None = None,
    phrase: bool = False,
    limit: int = SEARCH_LIMIT,
    keep_since: date | None = None,
    keep_until: date | None = None,
    seconds: float | None = None,
) -> dict[str, Any] | None:
    """One backend search. keep_since and keep_until drop rows after the fact,
    for the searches that are only quick when the backend is not asked to filter."""
    exact = mode == "exact"
    arguments: dict[str, Any] = {"query": query, "mode": mode, "limit": limit}
    arguments.update(_bounds(since, until))
    if chat_ids:
        arguments["chat_ids"] = chat_ids
    result = await run.part(
        "exact" if exact else "meaning",
        run.call(run.api.execute_data_tool, "search_messages", arguments),
        seconds,
    )
    for raw in (result or {}).get("results") or []:
        # A match may sit deep in a transcript, past the part the search returns,
        # so what was searched for is noted on what was found.
        marks: dict[str, Any] = {}
        if term:
            marks["words_hit"] = {term}
        elif phrase:
            marks["phrase_hit"] = True
        elif not exact:
            marks["similarity"] = float((raw or {}).get("similarity") or 0)
        found = _found(raw, **marks)
        if found and _inside(found, keep_since, keep_until):
            run.keep(found)
    return result


async def _search_word(
    run: _Run, query: str, term: str, *, recent: date, until: date | None
) -> None:
    """A word as written, across the whole archive, without the slow case.

    A common word takes the backend tens of seconds to find everywhere, and a
    call that is cut off keeps running there. So the last months go first; only
    a word that is rare there is then looked for in all the years, which is
    quick because it is rare.
    """
    started = time.monotonic()
    lately = await _search(run, "exact", query, since=recent, until=until, term=term)
    left = PART_SECONDS - (time.monotonic() - started)
    if lately is not None and not lately.get("has_more") and left > 1:
        await _search(run, "exact", query, since=None, until=until, term=term, seconds=left)


async def _read_chat(run: _Run, understood: Understood, chat: dict[str, Any], pages: int) -> None:
    """Read a named chat as it is, newest first, from the end of the window back."""
    before = None
    if understood.until:
        before = _cursor((_day_start(understood.until) + timedelta(days=1)).isoformat(), 0)
    deadline = time.monotonic() + READ_SECONDS
    stretch = run.stretch()
    for number in range(pages):
        call = run.call(run.api.get_messages, chat_id=str(chat["id"]), limit=PAGE, before=before)
        if number == 0:
            page = await call
        else:
            # A later page that comes too late is left unread; what is read stays.
            try:
                page = await asyncio.wait_for(call, deadline - time.monotonic())
            except TimeoutError:
                break
        past_the_window = False
        for raw in page.get("messages") or []:
            found = _found(raw, chat, read=stretch)
            if found is None:
                continue
            if understood.since and found.when.date() < understood.since:
                past_the_window = True
            else:
                run.keep(found)
        before = page.get("next_cursor")
        if past_the_window or not page.get("has_more") or not before:
            break


def _active_chats(chats: list[dict[str, Any]], since: date, until: date) -> list[dict[str, Any]]:
    """The chats that had a message in the window or after it, the surest first.

    The list of chats knows only each chat's last message. A chat whose last
    message falls inside the window was active then for certain; one that went on
    afterwards may have been, the sooner it stopped the likelier.
    """
    start, end = _day_start(since), _day_start(until) + timedelta(days=1)
    active = []
    for chat in chats:
        last = _moment(chat.get("last_activity_at"))
        # A channel is a broadcast, not a conversation; it is read only when named.
        if last is None or last < start or chat.get("chat_type") == "channel":
            continue
        active.append((max(last - end, timedelta(0)), -last.timestamp(), str(chat.get("id")), chat))
    return [chat for *_, chat in sorted(active, key=lambda item: item[:3])[:ACTIVE_CHATS]]


async def _read_active_chats(
    run: _Run, understood: Understood, chats: list[dict[str, Any]]
) -> None:
    """Read what each of those chats holds inside the window: a page each."""
    assert understood.since and understood.until
    before = _cursor((_day_start(understood.until) + timedelta(days=1)).isoformat(), 0)
    deadline = time.monotonic() + READ_SECONDS

    async def read(chat: dict[str, Any]) -> None:
        # A chat that answers too late is left unread; the reading fails only if none answers.
        page = await asyncio.wait_for(
            run.call(
                run.api.get_messages, chat_id=str(chat["id"]), limit=ACTIVE_PAGE, before=before
            ),
            deadline - time.monotonic(),
        )
        stretch = run.stretch()
        for raw in page.get("messages") or []:
            found = _found(raw, chat, read=stretch)
            if found and found.when.date() >= understood.since:
                run.keep(found)

    outcomes = await asyncio.gather(*(read(chat) for chat in chats), return_exceptions=True)
    errors = [outcome for outcome in outcomes if isinstance(outcome, Exception)]
    if errors and len(errors) == len(outcomes):
        raise errors[0]


async def _files(
    run: _Run, understood: Understood, part: str, media_types: list[str], named: list[str]
) -> None:
    arguments: dict[str, Any] = {"media_types": media_types, "limit": FILE_LIMIT}
    arguments.update(_bounds(understood.since, understood.until))
    query = " ".join(understood.terms)
    if named:
        # The newest of the named chats; the question's words are looked for here.
        arguments["chat_ids"] = named
    elif len(query) >= 2:
        arguments["query"] = query
    result = await run.part(part, run.call(run.api.execute_data_tool, "get_files", arguments))
    for raw in (result or {}).get("files") or []:
        run.keep(_found(raw))


def _exact_queries(understood: Understood, most: int) -> list[tuple[str, str]]:
    """The words to look for as written: the longest, then the same without endings.

    The index keeps words as they were typed, so a word is also tried without
    its case ending, the form it most often has in a message.
    """
    queries: dict[str, str] = {}
    longest = sorted(understood.terms, key=len, reverse=True)[:3]
    for term in longest:
        queries.setdefault(term, term)
    for term in longest:
        root = stem(fold(term))
        if len(root) >= 4:
            queries.setdefault(root, term)
    return list(queries.items())[:most]


def _pages(who: list[dict[str, Any]]) -> dict[str, int]:
    """How many pages of each named chat to read: what it holds, within what all may take."""
    pages = {
        str(chat["id"]): min(
            PAGES_ONE_CHAT, max(1, math.ceil(int(chat.get("total_messages_synced") or 0) / PAGE))
        )
        for chat in who
    }
    while sum(pages.values()) > PAGES_IN_ALL:
        longest = max(pages, key=lambda chat_id: pages[chat_id])
        pages[longest] -= 1
    return pages


def _plan(
    run: _Run,
    understood: Understood,
    chats: list[dict[str, Any]],
    today: date,
) -> tuple[list[Awaitable[None]], bool]:
    """Every part of the retrieval, to run at once, and whether the whole archive is searched."""
    since, until = understood.since, understood.until
    named = [str(chat["id"]) for chat in understood.who]
    has_subject = bool(understood.terms or understood.phrases)
    days = (until - since).days + 1 if since and until else None
    narrow = days is not None and days <= NARROW_DAYS
    meaning = " ".join(understood.terms) if len(understood.terms) > 1 else understood.asked
    parts: list[Awaitable[None]] = []

    if named:
        pages = _pages(understood.who)
        parts += [
            run.part("browse", _read_chat(run, understood, chat, pages[str(chat["id"])]))
            for chat in understood.who
        ]
        # A chat too long to read whole is also searched: by meaning inside a year
        # (quick with a window, slow and empty-handed without), and by wording.
        deep = [
            str(chat["id"])
            for chat in understood.who
            if int(chat.get("total_messages_synced") or 0) > pages[str(chat["id"])] * PAGE
        ]
        if has_subject and deep:
            floor = since or (until or today) - timedelta(days=MEANING_DAYS)
            parts.append(
                _search(
                    run,
                    "hybrid",
                    meaning,
                    since=floor,
                    until=until,
                    chat_ids=deep,
                    limit=MEANING_LIMIT,
                )
            )
            parts += [
                _search(run, "exact", query, since=since, until=until, chat_ids=deep, term=term)
                for query, term in _exact_queries(understood, 4)
            ]
        parts += [
            _search(run, "exact", phrase, since=since, until=until, chat_ids=named, phrase=True)
            for phrase in understood.phrases
        ]
        if has_subject:
            # A person also writes in group chats. Their name is on every message
            # there, and the search by wording reads it: the newest of the last months.
            recent = since or (until or today) - timedelta(days=WORDING_DAYS)
            for chat in [chat for chat in understood.who if is_person(chat)][:2]:
                name = next(
                    (word for word in words(str(chat.get("title") or "")) if len(word) >= 3), None
                )
                if name:
                    parts.append(_search(run, "exact", name, since=recent, until=until))
    elif narrow:
        active = _active_chats(chats, since, until)
        if active:
            parts.append(run.part("browse", _read_active_chats(run, understood, active)))

    whole_archive = has_subject or not named
    if whole_archive:
        # A window with a start and no end runs to today. Asked for with its
        # start alone, a month took the search by meaning six seconds, every time.
        span = ((until or today) - since).days + 1 if since else None
        if span is None or span <= NARROW_DAYS:
            # Unfiltered, or filtered to a few days: both quick.
            parts.append(
                _search(
                    run, "hybrid", meaning, since=since, until=until, limit=ARCHIVE_MEANING_LIMIT
                )
            )
        else:
            # A window of weeks makes the search by meaning crawl. Ask without
            # it and keep what falls inside.
            parts.append(
                _search(
                    run,
                    "hybrid",
                    meaning,
                    since=None,
                    until=None,
                    limit=ARCHIVE_MEANING_LIMIT,
                    keep_since=since,
                    keep_until=until,
                )
            )
        # One common word across the whole archive takes the backend tens of
        # seconds, so words are searched inside the window or the last months.
        floor = since or (until or today) - timedelta(days=WORDING_DAYS)
        # With a chat named, the rest of the archive is a fallback: fewer words.
        wording = _exact_queries(understood, 2 if named else 3)
        if since is None:
            parts += [
                _search_word(run, query, term, recent=floor, until=until) for query, term in wording
            ]
        elif days is None or days <= WORDING_DAYS:
            parts += [
                _search(run, "exact", query, since=since, until=until, term=term)
                for query, term in wording
            ]
        if not named:
            parts += [
                # A quotation is the person's own exact words: it is looked for everywhere.
                _search(
                    run,
                    "exact",
                    phrase,
                    since=since if understood.quoted and place == 0 else floor,
                    until=until,
                    phrase=True,
                )
                for place, phrase in enumerate(understood.phrases)
            ]

    wanted_files = [
        *(FILE_TYPES if "file" in understood.hints else ()),
        *(("photo",) if "photo" in understood.hints else ()),
    ]
    if wanted_files:
        parts.append(_files(run, understood, "files", wanted_files, named))
    if "voice" in understood.hints:
        parts.append(_files(run, understood, "voice", list(VOICE_TYPES), named))
    return parts, whole_archive


# --- judging what was found --------------------------------------------------------


def _kind(found: _Found, hints: frozenset[str]) -> str:
    if found.media in VOICE_TYPES:
        return "voice"
    if found.media:
        return "file"
    return "link" if found.urls and "link" in hints else "message"


def _has_asked_shape(found: _Found, hints: frozenset[str]) -> bool:
    """Whether the message looks like the thing asked for: a list, an address, a number.

    A shape is a strong lead: a list is a list whether or not it says "список".
    """
    if not hints:
        return False
    text = fold(f"{found.text} {found.said if found.media in VOICE_TYPES else ''}")
    links = " ".join(found.urls)
    return bool(
        ("app" in hints and _APP_LINK.search(f"{text} {links}"))
        or (
            "list" in hints
            and (len(_LIST_LINE.findall(found.text)) >= 3 or _LIST_INLINE.search(found.text))
        )
        or ("address" in hints and _ADDRESS.search(f"{text} {links}"))
        or ("phone" in hints and _PHONE.search(text))
        or ("amount" in hints and _AMOUNT.search(text))
    )


def _is_asked_container(found: _Found, hints: frozenset[str]) -> bool:
    """Whether the message is the kind of carrier asked for: a file, a photo, a link, a voice.

    A weak lead when the question also has a subject: a chat holds many links,
    and "the link to that message" may not even mean a link inside it.
    """
    return bool(
        ("voice" in hints and found.media in VOICE_TYPES)
        or ("photo" in hints and found.media == "photo")
        or ("file" in hints and found.media and found.media not in ("photo", "voice", "video_note"))
        or ("link" in hints and found.urls)
    )


@dataclass
class _Judged:
    found: _Found
    supported: bool  # the wording, the kind of thing or the named place fits
    strong: bool  # by its own words or its shape, not only by what kind of carrier it is
    near: bool  # in a named chat, or sent by a named person
    score: float
    by_summary: bool  # a document that fits only through its long summary


def _is_the_question(asked: set[str], stems: set[str]) -> bool:
    """A message that repeats the question is the question: sent to an assistant in
    this same archive, typed or dictated, or quoted back by it."""
    return len(asked) >= 6 and len(asked & stems) >= 0.8 * len(asked)


def _judge(run: _Run, understood: Understood, narrow: bool, question: str) -> list[_Judged]:
    """Weigh everything found against the question.

    A word counts in any of its forms, and counts for more the fewer of the
    found messages hold it: in a chat about apps, "приложение" says little and
    "бизнес-ланчи" says a lot.
    """
    asked = {stem(word) for word in words(question) if len(word) >= 3}
    founds = [
        found
        for found in run.pool.values()
        if not _is_the_question(asked, {stem(word) for word in words(f"{found.text} {found.said}")})
    ]
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
    terms = understood.terms[:MAX_TERMS]
    families = [stem(fold(term)) for term in terms]
    phrases = [joined for phrase in understood.phrases if (joined := " ".join(words(phrase)))]
    has_subject = bool(terms or phrases)

    own_words = {
        found.id: words(f"{found.text} {found.file_name or ''} {found.sender}") for found in founds
    }
    whole_words = {
        found.id: own_words[found.id] + words(f"{found.summary} {found.said}")
        if found.media
        else own_words[found.id]
        for found in founds
    }
    whole_stems = {key: {stem(word) for word in value} for key, value in whole_words.items()}
    vocabulary = set().union(*whole_stems.values()) if whole_stems else set()
    forms = [{known for known in vocabulary if same_word(family, known)} for family in families]
    matched = {
        found.id: {
            index
            for index, term in enumerate(terms)
            if term in found.words_hit or not forms[index].isdisjoint(whole_stems[found.id])
        }
        for found in founds
    }
    held_by = Counter(index for indexes in matched.values() for index in indexes)
    weight = [
        math.log(1 + len(founds) / (1 + held_by[index])) if held_by[index] else 0.0
        for index in range(len(terms))
    ]
    all_weight = sum(weight)

    # How strongly each message speaks of the subject, by its own words.
    whole_text = {found.id: " ".join(whole_words[found.id]) for found in founds}
    has_phrase = {
        found.id: found.phrase_hit or any(phrase in whole_text[found.id] for phrase in phrases)
        for found in founds
    }
    wording = {
        found.id: sum(weight[index] for index in matched[found.id]) / all_weight
        if all_weight
        else 0.0
        for found in founds
    }
    lead = {found.id: 3 * has_phrase[found.id] + 2 * wording[found.id] for found in founds}
    around = _talk_around(founds, lead)

    judged = []
    for found in founds:
        mine = matched[found.id]
        shape = _has_asked_shape(found, understood.hints)
        container = _is_asked_container(found, understood.hints)
        sender = {name_key(word) for word in words(found.sender)}
        near = found.chat_id in named or any(person <= sender for person in people)
        right_side = understood.direction is not None and (
            found.outgoing == (understood.direction == "out")
        )
        if not has_subject:
            # No subject to search for. A question about a kind of thing ("the link
            # from Марта") is answered by things of that kind; any other, by what the
            # named chat or the window holds.
            in_scope = near or bool(understood.since or understood.until)
            strong = (shape or container) if understood.hints else in_scope
        elif near or narrow:
            # Inside a named chat or a few days, one word of the question is a lead.
            strong = has_phrase[found.id] or shape or bool(mine)
        else:
            strong = has_phrase[found.id] or shape or wording[found.id] >= WORDING_SHARE
        # A file, a picture or a link seldom names its subject; the talk around it
        # does. So a thing sent in the middle of talk about the subject fits as well
        # as its own words would, and by as much as the talk does.
        talk = around.get(found.id, 0.0)
        in_the_talk = (
            has_subject
            and talk >= TALK_LEAD
            and bool(found.media or found.urls)
            and (understood.direction is None or right_side)
        )
        own = " ".join(own_words[found.id])
        own_stems = {stem(word) for word in own_words[found.id]}
        in_own_words = any(phrase in own for phrase in phrases) or any(
            not forms[index].isdisjoint(own_stems) for index in mine
        )
        judged.append(
            _Judged(
                found=found,
                supported=strong or container or in_the_talk,
                strong=strong or in_the_talk,
                near=near,
                score=(
                    max(lead[found.id], talk if in_the_talk else 0.0)
                    + 2 * shape
                    + container
                    + right_side
                    + min(found.similarity, 1.0)
                    + 0.5 * talk
                ),
                by_summary=bool(
                    found.media == "document"
                    and has_subject
                    and not in_own_words
                    and "file" not in understood.hints
                ),
            )
        )
    return judged


def _talk_around(founds: list[_Found], lead: dict[str, float]) -> dict[str, float]:
    """For each message read in its chat, the strongest lead among the messages
    sent within half an hour of it: how much the talk around it is on the subject."""
    threads: dict[str, list[_Found]] = {}
    for found in founds:
        if found.read:
            threads.setdefault(found.chat_id, []).append(found)
    around: dict[str, float] = {}
    for thread in threads.values():
        thread.sort(key=lambda found: found.when)
        first = 0
        for place, found in enumerate(thread):
            while found.when - thread[first].when > TALK_SPAN:
                first += 1
            last = place
            while last + 1 < len(thread) and thread[last + 1].when - found.when <= TALK_SPAN:
                last += 1
            others = [lead[other.id] for other in thread[first : last + 1] if other is not found]
            around[found.id] = max(others, default=0.0)
    return around


def _select(judged: list[_Judged], understood: Understood, limit: int) -> list[_Judged]:
    named = {str(chat["id"]) for chat in understood.who}
    # The wording fits before only the meaning does; its own words or shape before
    # the mere kind of carrier; a named place before the rest of the archive; a better
    # fit before a worse, in quarter-point steps; then the newest.
    judged.sort(
        key=lambda item: (
            item.supported,
            item.strong,
            item.near,
            int(item.score * 4) if item.supported else item.found.similarity,
            item.found.when,
        ),
        reverse=True,
    )
    found_near = any(item.strong and item.near for item in judged)
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


async def _read_around(run: _Run, understood: Understood, chosen: list[_Judged]) -> None:
    """Read the talk around the best few pieces that a search found and no reading reached.

    A search finds a message by its words. The screenshot, the link or the file
    sent in the same minutes seldom has them, and in a long chat that minute may
    be months behind what was read. Read there, they are judged with their talk.
    """
    picked: list[_Found] = []
    for item in sorted(chosen, key=lambda item: (item.strong, item.score), reverse=True):
        found = item.found
        if found.read or any(
            other.chat_id == found.chat_id and abs(other.when - found.when) <= TALK_SPAN
            for other in picked
        ):
            continue
        picked.append(found)
        if len(picked) == AROUND_READS:
            break

    async def read(found: _Found) -> None:
        here = _cursor(found.sent_at, found.number, found.id)
        chat = {"id": found.chat_id, "title": found.chat_title}
        pages = await asyncio.gather(
            *(
                run.call(
                    run.api.get_messages,
                    chat_id=found.chat_id,
                    limit=AROUND_PAGE,
                    **{side: here},
                )
                for side in ("before", "after")
            )
        )
        stretch = run.stretch()
        found.read = stretch
        for page in pages:
            for raw in page.get("messages") or []:
                near = _found(raw, chat, read=stretch)
                if near and _inside(near, understood.since, understood.until):
                    run.keep(near)
                    # One stretch from end to end, even where it meets one read before.
                    run.pool[near.id].read = stretch

    await asyncio.gather(*(read(found) for found in picked))


# --- writing the evidence ----------------------------------------------------------


def _clip(text: str, limit: int) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _around(text: str, understood: Understood, limit: int) -> str:
    """A long text cut to the part where the question's words are, not just its start."""
    text = text.strip()
    if len(text) <= limit:
        return text
    low = fold(text)
    sought = [*(fold(phrase) for phrase in understood.phrases)]
    sought += [stem(fold(term)) for term in understood.terms]
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


def _neighbour_line(
    near: _Found | None, found: _Found, shown: set[str], understood: Understood
) -> str | None:
    """A message next to the evidence, when it is short and part of the same exchange.

    A voice message next to it is given by the part of its transcript where the
    question's words are: a spoken question is what a two-word answer answers.
    """
    if near is None or near.id in shown or abs(near.when - found.when) > timedelta(days=1):
        return None
    if near.media in VOICE_TYPES and (near.said or near.summary):
        return f"{near.sender}: [voice] {_around(near.said or near.summary, understood, NEIGHBOUR_CHARS)}"
    if not near.text or len(near.text) > NEIGHBOUR_CHARS:
        return None
    return f"{near.sender}: {near.text}"


def _threads(run: _Run) -> dict[tuple[str, int], list[_Found]]:
    """The stretches of chat that were read, in order, so neighbours need no call."""
    threads: dict[tuple[str, int], list[_Found]] = {}
    for found in run.pool.values():
        if found.read:
            threads.setdefault((found.chat_id, found.read), []).append(found)
    for thread in threads.values():
        thread.sort(key=lambda found: (found.when, found.number))
    return threads


async def _neighbours(
    run: _Run, found: _Found, threads: dict[tuple[str, int], list[_Found]]
) -> tuple[_Found | None, _Found | None]:
    thread = threads.get((found.chat_id, found.read)) or []
    if found.read and found in thread:
        place = thread.index(found)
        before = thread[place - 1] if place else None
        after = thread[place + 1] if place + 1 < len(thread) else None
        return before, after
    here = _cursor(found.sent_at, found.number, found.id)
    chat = {"id": found.chat_id, "title": found.chat_title}
    earlier, later = await asyncio.gather(
        run.part(
            "context",
            run.call(run.api.get_messages, chat_id=found.chat_id, limit=1, before=here),
            CONTEXT_SECONDS,
        ),
        run.part(
            "context",
            run.call(run.api.get_messages, chat_id=found.chat_id, limit=1, after=here),
            CONTEXT_SECONDS,
        ),
    )
    return (
        _found(next(iter((earlier or {}).get("messages") or []), None), chat),
        _found(next(iter((later or {}).get("messages") or []), None), chat),
    )


async def _write(
    run: _Run,
    item: _Judged,
    number: int,
    cap: int,
    understood: Understood,
    shown: set[str],
    threads: dict[tuple[str, int], list[_Found]],
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
        before, after = await _neighbours(run, found, threads)
        if after is not None:
            # Start the page one message later, so the reply is on it too.
            open_at = _cursor(after.sent_at, after.number + 1)
        lines = [marked]
        room = cap - len(marked)
        said_before = _neighbour_line(before, found, shown, understood)
        if said_before and len(said_before) + 1 <= room:
            lines.insert(0, said_before)
            room -= len(said_before) + 1
        said_after = _neighbour_line(after, found, shown, understood)
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
        **({"chat_url": found.chat_url} if found.chat_url else {}),
        "open": {
            "tool": "get_chat_messages",
            "args": {"chat_id": found.chat_id, "before": open_at, "limit": OPEN_PAGE},
        },
    }


_PLAIN_PART = {
    "names": "the list of chats",
    "exact": "the search by wording",
    "meaning": "the search by meaning",
    "browse": "reading the chats",
    "files": "the list of files",
    "voice": "the list of voice messages",
    "context": "the neighbouring messages",
}
_SEARCHED_HOW = {
    "exact": "by wording",
    "meaning": "by meaning",
    "browse": "by reading the chats",
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
    open_client: Callable[[], TelegramAIClient],
    limit: int = 10,
    since: date | None = None,
    until: date | None = None,
    today: date | None = None,
) -> dict[str, Any]:
    """Understand the question, retrieve from every index at once, assemble the pack.

    open_client makes one more client for the same backend and key: the list of
    chats is loaded with it, behind this call and after it.
    """
    run = _Run(api)
    today = today or datetime.now(UTC).date()
    run.ran.add("names")
    chats = await chat_list(api, open_client)
    if chats is None:
        run.failed.add("names")
        chats = []
    understood = understand(question, chats, today=today, since=since, until=until)

    parts, whole_archive = _plan(run, understood, chats, today)
    await asyncio.gather(*parts)

    days = (
        (understood.until - understood.since).days + 1
        if understood.since and understood.until
        else None
    )
    narrow = days is not None and days <= NARROW_DAYS
    chosen = _select(_judge(run, understood, narrow, question), understood, limit)
    if any(not item.found.read for item in chosen):
        await run.part("context", _read_around(run, understood, chosen), CONTEXT_SECONDS)
        chosen = _select(_judge(run, understood, narrow, question), understood, limit)
    cap = min(MAX_ITEM_CHARS, MAX_TOTAL_CHARS // max(len(chosen), 1))
    shown = {item.found.id for item in chosen}
    threads = _threads(run)
    evidence = await asyncio.gather(
        *(
            _write(run, item, number, cap, understood, shown, threads)
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
            # Transcripts are part of the message index and of every chat that is
            # read; a question that asks for a voice message also reads their list.
            "voice": run.state("voice") if "voice" in run.ran else text_search,
            "names": run.state("names"),
            "exact": run.state("exact"),
            "meaning": run.state("meaning"),
            "browse": run.state("browse"),
            "files": run.state("files"),
            "links": False if "link" not in understood.hints else text_search,
            "context": run.state("context"),
        },
        "note": _note(run, understood, chosen, chat_count, message_count),
    }
