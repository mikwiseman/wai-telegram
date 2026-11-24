"""recall: one call that turns a question into a small pack of dated, sourced evidence.

Every chat, person and message here is invented.
"""

import asyncio
import base64
import json
import re
from datetime import date, datetime
from unittest.mock import AsyncMock, patch

import pytest
from mcp.types import CallToolResult
from telegram_wai_mcp import recall as recall_module
from telegram_wai_mcp import server
from telegram_wai_mcp.recall import recall

TODAY = date(2026, 11, 24)
VOICE = {"voice", "video_note", "audio"}


def _chat(chat_id, title, chat_type="private", username=None, synced=1000):
    return {
        "id": chat_id,
        "title": title,
        "chat_type": chat_type,
        "username": username,
        "telegram_chat_id": -1001234567890 if chat_type == "supergroup" else 4242,
        "last_activity_at": "2026-11-20T00:00:00Z",
        "total_messages_synced": synced,
        "indexing_enabled": True,
    }


def _msg(chat_id, number, day, text, sender="Марта Кораблёва", **extra):
    return {
        "id": f"m-{chat_id}-{number}",
        "chat_id": chat_id,
        "telegram_message_id": number,
        "sent_at": f"{day}T10:00:00Z" if len(day) == 10 else day,
        "text": text,
        "sender_name": sender,
        "is_outgoing": False,
        "has_media": bool(extra.get("media_type")),
        "media_type": None,
        "content_summary": None,
        "content_text": None,
        "media_file_name": None,
        "visible_urls": [],
        "hidden_urls": [],
        "deleted_at": None,
        **extra,
    }


def _when(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _cursor(value):
    return json.loads(base64.urlsafe_b64decode(value + "=" * (-len(value) % 4)))


class FakeArchive:
    """The read side of the backend, and nothing else: a write would raise."""

    base_url = "https://archive.test"

    def __init__(self, chats, messages):
        self.chats = {chat["id"]: chat for chat in chats}
        self.messages = messages
        self.calls = []
        self.by_meaning = []  # ids a search by meaning returns whatever the words are
        self.broken = set()  # "exact", "hybrid", "get_files", "get_messages", "list_chats"
        self.stuck = set()

    async def _trouble(self, part):
        if part in self.stuck:
            await asyncio.sleep(30)
        if part in self.broken:
            raise RuntimeError(f"Backend returned HTTP 500 for {part}")

    async def close(self):
        return None

    async def list_chats(self, limit=100, cursor=None, **_):
        self.calls.append(("list_chats", {}))
        await self._trouble("list_chats")
        chats = list(self.chats.values())
        return {"chats": chats, "total": len(chats), "has_more": False, "next_cursor": None}

    def _url(self, message):
        chat = self.chats[message["chat_id"]]
        if chat.get("username"):
            return f"https://t.me/{chat['username']}/{message['telegram_message_id']}"
        if chat["chat_type"] == "supergroup":
            return f"https://t.me/c/1234567890/{message['telegram_message_id']}"
        return None

    def _row(self, message, similarity=1.0):
        chat = self.chats[message["chat_id"]]
        return {
            **{key: value for key, value in message.items() if key != "content_text"},
            "chat_title": chat["title"],
            "chat_type": chat["chat_type"],
            "chat_username": chat.get("username"),
            "similarity": similarity,
            "content_preview": (message.get("content_text") or "")[:1200] or None,
            "telegram_message_url": self._url(message),
        }

    def _in_scope(self, message, arguments):
        if message.get("deleted_at"):
            return False
        if arguments.get("chat_ids") and message["chat_id"] not in arguments["chat_ids"]:
            return False
        sent = _when(message["sent_at"])
        if arguments.get("date_from") and sent < _when(arguments["date_from"]):
            return False
        return not (arguments.get("date_to") and sent > _when(arguments["date_to"]))

    async def execute_data_tool(self, name, arguments=None):
        arguments = dict(arguments or {})
        self.calls.append((name, arguments))
        if name == "search_messages":
            mode = arguments.get("mode", "hybrid")
            await self._trouble(mode)
            return self._search(arguments, mode)
        if name == "get_files":
            await self._trouble("get_files")
            return self._files(arguments)
        raise AssertionError(f"recall must not call {name}")

    def _search(self, arguments, mode):
        query = arguments["query"].casefold()
        rows = []
        for message in self.messages:
            if not self._in_scope(message, arguments):
                continue
            haystack = " ".join(
                str(message.get(field) or "")
                for field in ("text", "content_text", "content_summary", "sender_name")
            ).casefold()
            said = set(re.findall(r"\w+", haystack))
            wanted = re.findall(r"\w+", query)
            # Like the backend: whole words as they were typed, no word forms.
            if mode == "exact":
                if query in haystack and all(word in said for word in wanted):
                    rows.append(self._row(message, 1.0))
            elif all(word in said for word in wanted):
                rows.append(self._row(message, 0.9))
            elif message["id"] in self.by_meaning:
                rows.append(self._row(message, 0.45))
        rows.sort(key=lambda row: (row["similarity"], row["sent_at"]), reverse=True)
        limit = arguments.get("limit", 20)
        return {
            "results": rows[:limit],
            "query": arguments["query"],
            "total": len(rows[:limit]),
            "has_more": len(rows) > limit,
            "next_cursor": None,
        }

    def _files(self, arguments):
        types = arguments.get("media_types") or ["document", "photo", "video", "voice"]
        entries = []
        for message in self.messages:
            if message.get("media_type") not in types or not self._in_scope(message, arguments):
                continue
            chat = self.chats[message["chat_id"]]
            entries.append(
                {
                    "message_id": message["id"],
                    "chat_id": message["chat_id"],
                    "chat_title": chat["title"],
                    "chat_type": chat["chat_type"],
                    "telegram_message_id": message["telegram_message_id"],
                    "sent_at": message["sent_at"],
                    "sender_name": message["sender_name"],
                    "is_outgoing": message["is_outgoing"],
                    "media_type": message["media_type"],
                    "media_file_name": message.get("media_file_name"),
                    "caption": message.get("text"),
                    "content_summary": message.get("content_summary"),
                    "telegram_message_url": self._url(message),
                }
            )
        entries.sort(key=lambda entry: entry["sent_at"], reverse=True)
        return {"files": entries[: arguments.get("limit", 20)], "total": len(entries)}

    async def get_messages(self, chat_id, limit=50, before=None, after=None):
        self.calls.append(("get_messages", {"chat_id": chat_id, "limit": limit}))
        await self._trouble("get_messages")
        rows = sorted(
            (message for message in self.messages if message["chat_id"] == chat_id),
            key=lambda message: (_when(message["sent_at"]), message["telegram_message_id"]),
        )

        def position(message):
            return (_when(message["sent_at"]), message["telegram_message_id"], message["id"])

        def edge(cursor):
            point = _cursor(cursor)
            return (_when(point["sent_at"]), point["telegram_message_id"], point["id"])

        if after:
            page = [message for message in rows if position(message) > edge(after)][:limit]
            page.reverse()
        else:
            if before:
                rows = [message for message in rows if position(message) < edge(before)]
            page = rows[::-1][:limit]
        return {
            "messages": [
                {
                    **{key: value for key, value in message.items() if key != "content_text"},
                    "content_preview": (message.get("content_text") or "")[:1200] or None,
                    "telegram_message_url": self._url(message),
                }
                for message in page
            ],
            "has_more": False,
            "next_cursor": None,
        }


CHATS = [
    _chat("c-marta", "Марта Кораблёва"),
    _chat("c-timur", "Тимур Сойкин"),
    _chat("c-dacha", "Дача Сосновка", "group"),
    _chat("c-orion", "Orion Book Club", "supergroup", username="orionbooks"),
    _chat("c-news", "Вести посёлка", "supergroup"),
]


async def _inventory(api):
    return await api.list_chats(limit=200)


async def _recall(api, question, **options):
    options.setdefault("today", TODAY)
    return await recall(api, question=question, inventory=_inventory, **options)


def _fence_archive():
    messages = [
        _msg("c-marta", 10, "2026-03-02", "Забор ставим деревянный, я уже договорилась."),
        _msg("c-marta", 11, "2026-10-05", "Передумали: забор будет из сетки, деревянный дорого."),
        _msg("c-marta", 12, "2026-10-05T10:01:00Z", "Ок, сетка так сетка.", sender=None),
        _msg("c-marta", 13, "2026-11-01", "Привет! Как дела?"),
        _msg(
            "c-dacha", 20, "2026-09-10", "Кто едет в субботу красить забор?", sender="Лев Ордынцев"
        ),
        _msg("c-dacha", 21, "2026-09-11", "Забор покрасили, фото позже.", sender="Лев Ордынцев"),
        _msg("c-dacha", 22, "2026-09-12", "Про забор: краски хватило.", sender="Зоя Вершинина"),
        _msg(
            "c-news", 30, "2026-08-01", "В посёлке меняют забор у въезда.", sender="Вести посёлка"
        ),
    ]
    messages[2]["is_outgoing"] = True
    return FakeArchive(CHATS, messages)


EVIDENCE_KEYS = {"n", "kind", "layer", "title", "date", "text", "id", "url", "open"}


async def test_evidence_is_newest_first_with_a_source_for_every_item():
    api = _fence_archive()
    result = await _recall(api, "что мы решили про забор")

    assert set(result) == {"understood", "evidence", "searched", "note"}
    assert result["understood"] == {"from": None, "to": None, "who": [], "about": ["забор"]}
    evidence = result["evidence"]
    assert [item["n"] for item in evidence] == list(range(1, len(evidence) + 1))
    dates = [item["date"] for item in evidence]
    assert dates == sorted(dates, reverse=True)
    assert dates[0] == "2026-10-05"
    for item in evidence:
        assert set(item) == EVIDENCE_KEYS
        assert item["layer"] == "raw"
        assert item["kind"] in {"message", "file", "link", "voice"}
        assert item["id"].startswith("m-")
        assert item["open"]["tool"] == "get_chat_messages"
        assert set(item["open"]["args"]) == {"chat_id", "before", "limit"}
    newest = evidence[0]
    assert newest["title"] == "Марта Кораблёва · Марта Кораблёва"
    assert "забор будет из сетки" in newest["text"]
    assert newest["url"] is None  # Telegram has no link to a message in a private chat
    assert result["searched"]["chats"] == 5
    assert result["searched"]["messages"] == 5000
    assert result["searched"]["voice"] is True
    assert "newest first" in result["note"]


async def test_a_better_fit_is_chosen_first_but_shown_in_date_order():
    messages = [
        _msg(
            "c-news",
            1,
            "2026-05-01",
            "Решено: забор из сетки ставим в мае.",
            sender="Вести посёлка",
        ),
        *(
            _msg(
                f"c-{index}",
                1,
                f"2026-10-{index + 1:02d}",
                "Забор опять обсуждали.",
                sender="Сосед",
            )
            for index in range(4)
        ),
    ]
    chats = [*CHATS, *(_chat(f"c-{index}", f"Участок {index}", "group") for index in range(4))]
    api = FakeArchive(chats, messages)

    # Only two fit in: the one with the exact phrase wins a place over newer ones...
    result = await _recall(api, "кто писал «забор из сетки»", limit=2)
    assert [item["id"] for item in result["evidence"]] == ["m-c-3-1", "m-c-news-1"]
    # ...and is still read after them, because it is older.
    assert [item["date"] for item in result["evidence"]] == ["2026-10-04", "2026-05-01"]


async def test_one_word_out_of_many_is_not_the_subject_unless_it_is_a_rare_one():
    messages = [
        _msg("c-dacha", 1, "2026-10-01", "Профнастила нет на складе.", sender="Лев Ордынцев"),
        *(
            _msg("c-news", number, f"2026-09-{number - 9:02d}", "Ремонта дороги не будет.")
            for number in range(10, 39)
        ),
        _msg("c-orion", 50, "2026-10-02", "Ремонта клуба ждём.", sender="Зоя Вершинина"),
        _msg("c-orion", 51, "2026-10-03", "Ремонта всё нет.", sender="Зоя Вершинина"),
    ]
    api = FakeArchive(CHATS, messages)
    result = await _recall(api, "что писали про поставку профнастила для ремонта забора")
    assert result["understood"]["about"] == ["поставку", "профнастила", "ремонта", "забора"]
    # Thirty-one messages hold the common word and nothing else of the question;
    # one holds the word the archive has only once.
    assert [item["id"] for item in result["evidence"]] == ["m-c-dacha-1"]


async def test_the_open_call_shows_the_message_in_its_chat():
    api = _fence_archive()
    result = await _recall(api, "что мы решили про забор")
    item = next(item for item in result["evidence"] if item["id"] == "m-c-marta-11")
    page = await api.get_messages(**item["open"]["args"])
    shown = [message["telegram_message_id"] for message in page["messages"]]
    assert 11 in shown and 12 in shown and 10 in shown


async def test_at_most_two_items_from_a_chat_nobody_named():
    result = await _recall(_fence_archive(), "что мы решили про забор")
    chats = [item["title"].split(" · ")[0] for item in result["evidence"]]
    assert chats.count("Дача Сосновка") == 2
    assert chats.count("Марта Кораблёва") == 2
    # The two kept from the group are its newest two.
    kept = [item["id"] for item in result["evidence"] if item["id"].startswith("m-c-dacha")]
    assert kept == ["m-c-dacha-22", "m-c-dacha-21"]


async def test_a_named_chat_or_person_gets_up_to_the_limit():
    api = _fence_archive()
    result = await _recall(api, "что писали в чате дача сосновка про забор")
    assert result["understood"]["who"] == [
        {"kind": "chat", "id": "c-dacha", "name": "Дача Сосновка"}
    ]
    assert [item["id"] for item in result["evidence"]] == [
        "m-c-dacha-22",
        "m-c-dacha-21",
        "m-c-dacha-20",
    ]
    assert result["searched"]["chats"] == 5  # the rest of the archive was searched as well
    # Its neighbours are evidence themselves, so they are not repeated inside it.
    assert result["evidence"][1]["text"] == "Забор покрасили, фото позже."

    person = await _recall(api, "что Марта писала про забор")
    assert person["understood"]["who"] == [
        {"kind": "person", "id": "c-marta", "name": "Марта Кораблёва"}
    ]
    assert [item["id"] for item in person["evidence"]] == ["m-c-marta-11", "m-c-marta-10"]

    limited = await _recall(api, "что писали в чате дача сосновка про забор", limit=1)
    assert [item["id"] for item in limited["evidence"]] == ["m-c-dacha-22"]


async def test_a_named_person_is_followed_into_group_chats():
    messages = [
        _msg("c-marta", 1, "2026-10-01", "Забор обсудим завтра."),
        _msg("c-dacha", 2, "2026-10-02", "Забор красим в субботу.", sender="Марта Кораблёва"),
        _msg("c-dacha", 3, "2026-10-03", "Забор уже покрасили.", sender="Лев Ордынцев"),
        _msg("c-dacha", 4, "2026-10-04", "Забор высох.", sender="Зоя Вершинина"),
    ]
    result = await _recall(FakeArchive(CHATS, messages), "что Марта писала про забор")
    assert [item["id"] for item in result["evidence"]] == ["m-c-dacha-2", "m-c-marta-1"]


async def test_the_rest_of_the_archive_answers_when_the_named_chat_holds_nothing():
    messages = [
        _msg("c-marta", 1, "2026-10-01", "Привет! Как дела?"),
        _msg("c-dacha", 3, "2026-10-03", "Забор уже покрасили.", sender="Лев Ордынцев"),
    ]
    result = await _recall(FakeArchive(CHATS, messages), "что Марта писала про забор")
    assert [item["id"] for item in result["evidence"]] == ["m-c-dacha-3"]
    assert "none is from the chat or person the question names" in result["note"]


async def test_short_neighbours_come_with_the_message():
    result = await _recall(_fence_archive(), "что Марта писала про забор")
    newest = result["evidence"][0]
    assert newest["text"] == (
        "» Марта Кораблёва: Передумали: забор будет из сетки, деревянный дорого.\n"
        "You: Ок, сетка так сетка."
    )
    # The message before it was sent seven months earlier: not part of this exchange.
    assert "договорилась" not in newest["text"]


async def test_a_neighbour_from_another_day_is_not_part_of_the_exchange():
    messages = [
        _msg("c-marta", 1, "2026-03-01", "С праздником!"),
        _msg("c-marta", 2, "2026-10-05", "Забор будет из сетки."),
        _msg("c-marta", 3, "2026-10-05T10:02:00Z", "Хорошо.", sender=None, is_outgoing=True),
        _msg("c-marta", 4, "2026-10-05T10:03:00Z", "Длинный ответ. " * 30),
    ]
    result = await _recall(FakeArchive(CHATS, messages), "что Марта писала про забор")
    assert [item["text"] for item in result["evidence"]] == [
        "» Марта Кораблёва: Забор будет из сетки.\nYou: Хорошо."
    ]


async def test_a_question_with_no_subject_reads_the_named_chat_in_the_window():
    api = _fence_archive()
    result = await _recall(api, "что писала Марта в октябре")
    assert result["understood"]["from"] == "2026-10-01"
    assert result["understood"]["to"] == "2026-10-31"
    assert [item["id"] for item in result["evidence"]] == ["m-c-marta-12", "m-c-marta-11"]
    assert result["searched"]["browse"] is True
    assert result["searched"]["chats"] == 1


async def test_the_window_limits_every_search():
    api = _fence_archive()
    result = await _recall(
        api, "что решили про забор", since=date(2026, 9, 1), until=date(2026, 9, 30)
    )
    assert {item["id"] for item in result["evidence"]} == {"m-c-dacha-22", "m-c-dacha-21"}
    searches = [arguments for name, arguments in api.calls if name == "search_messages"]
    assert searches
    for arguments in searches:
        assert arguments["date_from"] == "2026-09-01T00:00:00+00:00"
        assert arguments["date_to"] == "2026-09-30T23:59:59.999999+00:00"


async def test_text_stays_inside_the_budget():
    long_text = "Забор. " + "Очень длинное сообщение про краску и доски. " * 60
    messages = [
        _msg(f"c-{index}", 1, f"2026-10-{index + 1:02d}", long_text, sender=f"Сосед {index}")
        for index in range(20)
    ]
    chats = [_chat(f"c-{index}", f"Участок {index}", "group") for index in range(20)]
    api = FakeArchive(chats, messages)

    ten = await _recall(api, "что писали про забор")
    assert len(ten["evidence"]) == 10
    assert all(len(item["text"]) <= 700 for item in ten["evidence"])

    twenty = await _recall(api, "что писали про забор", limit=20)
    assert len(twenty["evidence"]) == 20
    assert sum(len(item["text"]) for item in twenty["evidence"]) <= 7000


async def test_nothing_matches_says_so_and_says_what_was_searched():
    api = _fence_archive()
    result = await _recall(api, "что писали про аквариум")
    assert result["evidence"] == []
    note = result["note"]
    assert note.startswith("Nothing in the archive matches")
    assert "searched 5,000 messages in 5 chats by wording and by meaning" in note
    assert note.count(".") == 1  # one plain sentence
    assert result["searched"]["exact"] is True
    assert result["searched"]["meaning"] is True


async def test_matches_by_meaning_only_are_few_and_come_after_the_wording():
    messages = [
        _msg("c-dacha", number, f"2026-10-{number:02d}", f"Ограду чинили, день {number}.")
        for number in range(1, 9)
    ]
    api = FakeArchive(CHATS, messages)
    api.by_meaning = [message["id"] for message in messages]
    result = await _recall(api, "что писали в чате дача сосновка про забор")
    assert len(result["evidence"]) == 3
    assert "by meaning" in result["note"]


async def test_a_failed_part_is_marked_and_the_rest_is_returned():
    api = _fence_archive()
    api.broken = {"hybrid"}
    result = await _recall(api, "что мы решили про забор")
    assert result["searched"]["meaning"] == "failed"
    assert result["searched"]["exact"] is True
    assert result["evidence"]
    assert "meaning" in result["note"]


async def test_a_slow_part_is_cut_off_and_marked():
    api = _fence_archive()
    api.stuck = {"hybrid", "get_messages"}
    with patch.object(recall_module, "PART_SECONDS", 0.05):
        result = await asyncio.wait_for(_recall(api, "что мы решили про забор"), timeout=2)
    assert result["searched"]["meaning"] == "failed"
    assert result["searched"]["context"] == "failed"
    assert [item["date"] for item in result["evidence"]][0] == "2026-10-05"
    # Without the neighbours the open call still starts at the message itself.
    assert result["evidence"][0]["open"]["args"]["chat_id"] == "c-marta"


async def test_without_the_list_of_chats_the_archive_is_still_searched():
    api = _fence_archive()
    api.broken = {"list_chats"}
    result = await _recall(api, "что Марта писала про забор")
    assert result["searched"]["names"] == "failed"
    assert result["understood"]["who"] == []
    assert result["evidence"]


async def test_everything_failing_is_not_reported_as_nothing_found():
    api = _fence_archive()
    api.broken = {"hybrid", "exact"}
    result = await _recall(api, "что мы решили про забор")
    assert result["evidence"] == []
    assert not result["note"].startswith("Nothing in the archive matches")
    assert "could not be searched" in result["note"]
    assert result["searched"]["voice"] == "failed"


async def test_a_voice_message_is_found_by_what_was_said_in_it():
    messages = [
        _msg("c-timur", 1, "2026-10-01", "Привет", sender="Тимур Сойкин"),
        _msg(
            "c-timur",
            2,
            "2026-10-02",
            None,
            sender="Тимур Сойкин",
            media_type="voice",
            content_text="Слушай, насчёт забора: столбы привезут в четверг, а сетку в пятницу.",
            content_summary="Тимур сообщает сроки доставки столбов и сетки.",
        ),
        _msg("c-timur", 3, "2026-10-03", "Столбы привезли?", sender="Тимур Сойкин"),
    ]
    api = FakeArchive(CHATS, messages)

    by_words = await _recall(api, "когда привезут столбы")
    voice = next(item for item in by_words["evidence"] if item["id"] == "m-c-timur-2")
    assert voice["kind"] == "voice"
    assert "столбы привезут в четверг" in voice["text"]

    asked_for = await _recall(api, "что Тимур говорил в голосовом")
    assert [item["id"] for item in asked_for["evidence"]] == ["m-c-timur-2"]
    assert asked_for["evidence"][0]["kind"] == "voice"
    files_calls = [arguments for name, arguments in api.calls if name == "get_files"]
    assert files_calls and all(set(call["media_types"]) <= VOICE for call in files_calls)
    assert asked_for["searched"]["voice"] is True

    api.broken = {"get_files"}
    failed = await _recall(api, "что Тимур говорил в голосовом")
    assert failed["searched"]["voice"] == "failed"


async def test_words_deep_in_a_long_transcript_still_count():
    # The search returns only the start of a transcript; the words sit past it.
    long_talk = "Сначала долго говорили о погоде и дороге. " * 40
    messages = [
        _msg(
            "c-timur",
            2,
            "2026-10-02",
            None,
            sender="Тимур Сойкин",
            media_type="voice",
            content_text=long_talk + "Так и есть: столбы привезут в четверг.",
        ),
    ]
    api = FakeArchive(CHATS, messages)
    by_words = await _recall(api, "когда привезут столбы")
    assert [item["id"] for item in by_words["evidence"]] == ["m-c-timur-2"]
    assert by_words["note"] == "Found 1 message in 1 chat, newest first."
    by_phrase = await _recall(api, "кто сказал «так и есть»")
    assert by_phrase["understood"]["about"] == []
    assert [item["id"] for item in by_phrase["evidence"]] == ["m-c-timur-2"]
    assert by_phrase["note"] == "Found 1 message in 1 chat, newest first."


async def test_a_long_text_is_cut_to_where_the_words_are():
    before = "Сначала долго говорили о погоде и дороге. " * 20
    after = " Потом ещё обсудили рассаду и теплицу." * 20
    messages = [
        _msg(
            "c-timur",
            2,
            "2026-10-02",
            None,
            sender="Тимур Сойкин",
            media_type="voice",
            content_text=before + "Столбы привезут в четверг." + after,
        ),
        _msg("c-marta", 3, "2026-10-03", before + "Столбы уже привезли." + after),
    ]
    result = await _recall(FakeArchive(CHATS, messages), "когда привезут столбы")
    voice, message = sorted(result["evidence"], key=lambda item: item["kind"], reverse=True)
    assert voice["text"].startswith("[voice] …")
    assert "Столбы привезут в четверг." in voice["text"]
    assert message["text"].startswith("…")
    assert "Столбы уже привезли." in message["text"]
    assert all(len(item["text"]) <= 700 for item in result["evidence"])


async def test_a_link_and_an_address_are_found_without_their_words():
    messages = [
        _msg("c-marta", 1, "2026-10-01", "Привет, как доехали?"),
        _msg("c-marta", 2, "2026-10-02", "Приезжайте: ул. Сосновая, д. 14, кв. 3, домофон 3."),
        _msg(
            "c-marta",
            3,
            "2026-10-03",
            "Вот таблица",
            hidden_urls=["https://tables.example/fence-budget"],
        ),
        _msg("c-marta", 4, "2026-10-04", "Спасибо!"),
    ]
    api = FakeArchive(CHATS, messages)

    address = await _recall(api, "найди сообщение, где Марта прислала адрес")
    assert [item["id"] for item in address["evidence"]] == ["m-c-marta-2"]

    link = await _recall(api, "ссылка от Марты")
    assert [item["id"] for item in link["evidence"]] == ["m-c-marta-3"]
    assert link["evidence"][0]["kind"] == "link"
    assert "https://tables.example/fence-budget" in link["evidence"][0]["text"]
    assert link["searched"]["links"] is True


async def test_a_file_is_found_and_its_telegram_link_is_kept():
    messages = [
        _msg(
            "c-orion",
            7,
            "2026-10-07",
            "Смета на забор",
            sender="Зоя Вершинина",
            media_type="document",
            media_file_name="smeta-zabor.pdf",
            content_summary="Смета: столбы, сетка, работа.",
        ),
    ]
    api = FakeArchive(CHATS, messages)
    result = await _recall(api, "найди файл со сметой на забор")
    assert [(item["id"], item["kind"]) for item in result["evidence"]] == [("m-c-orion-7", "file")]
    item = result["evidence"][0]
    assert item["url"] == "https://t.me/orionbooks/7"
    assert "smeta-zabor.pdf" in item["text"]
    assert result["searched"]["files"] is True
    for name, arguments in api.calls:
        assert not (name == "get_files" and arguments.get("prepare"))


async def test_long_file_summaries_do_not_crowd_out_messages():
    summary = "Документ описывает забор и ограждения. " * 40
    messages = [
        _msg(
            "c-dacha",
            number,
            f"2026-10-{number:02d}",
            None,
            sender="Лев Ордынцев",
            media_type="document",
            media_file_name=f"scan-{number}.pdf",
            content_summary=summary,
        )
        for number in range(1, 6)
    ]
    messages.append(_msg("c-dacha", 9, "2026-09-01", "Забор ставим в мае.", sender="Зоя Вершинина"))
    api = FakeArchive(CHATS, messages)
    result = await _recall(api, "что писали в чате дача сосновка про забор")
    files = [item for item in result["evidence"] if item["kind"] == "file"]
    assert len(files) == 1
    assert len(files[0]["text"]) <= 320
    assert any(item["id"] == "m-c-dacha-9" for item in result["evidence"])


async def test_recall_only_reads():
    api = _fence_archive()
    for question in (
        "что Марта писала про забор",
        "ссылка и файл от Марты, голосовое тоже",
        "что писали в чате дача сосновка вчера",
    ):
        await _recall(api, question)
    called = {name for name, _ in api.calls}
    assert called <= {"list_chats", "search_messages", "get_files", "get_messages"}
    assert {"search_messages", "get_files", "get_messages"} <= called


class TestTool:
    async def test_recall_is_listed_first_and_read_only(self):
        registry = AsyncMock()
        registry.list_data_tools.return_value = {
            "tools": [
                {"name": name, "description": name, "parameters": {"type": "object"}}
                for name in ("get_inbox", "get_files", "search_messages")
            ]
        }
        with patch.object(server, "get_client", return_value=registry):
            tools = await server.list_tools()
        assert tools[0].name == "recall"
        assert tools[0].annotations.readOnlyHint is True
        assert [tool.name for tool in tools].count("recall") == 1
        schema = tools[0].inputSchema
        assert schema["required"] == ["question"]
        assert schema["properties"]["question"]["minLength"] == 2
        assert schema["properties"]["question"]["maxLength"] == 1000
        limit = schema["properties"]["limit"]
        assert (limit["minimum"], limit["maximum"], limit["default"]) == (1, 20, 10)
        assert set(schema["properties"]) == {"question", "limit", "since", "until"}

    def test_the_server_tells_assistants_to_start_with_recall(self):
        instructions = server.server.create_initialization_options().instructions
        assert "recall" in instructions
        assert instructions.index("recall") < instructions.index("search_messages")

    async def test_the_answer_is_json_and_structured_content(self):
        api = _fence_archive()
        with patch.object(server, "get_client", return_value=api):
            result = await server.call_tool("recall", {"question": "что мы решили про забор"})
        assert isinstance(result, CallToolResult)
        assert result.isError is False
        assert json.loads(result.content[0].text) == result.structuredContent
        assert result.structuredContent["evidence"][0]["date"] == "2026-10-05"

    @pytest.mark.parametrize(
        ("arguments", "message"),
        [
            ({}, '"question" must be a non-empty string'),
            ({"question": "я"}, "2 to 1000 characters"),
            ({"question": "я" * 1001}, "2 to 1000 characters"),
            ({"question": "про забор", "since": "октябрь"}, '"since" must be a YYYY-MM-DD'),
            ({"question": "про забор", "limit": "ten"}, '"limit" must be an integer'),
            (
                {"question": "про забор", "since": "2026-10-02", "until": "2026-10-01"},
                '"since" must not be after "until"',
            ),
        ],
    )
    async def test_bad_input_is_refused_before_anything_is_read(self, arguments, message):
        api = _fence_archive()
        with patch.object(server, "get_client", return_value=api):
            result = await server.call_tool("recall", arguments)
        assert result.isError is True
        assert message in result.content[0].text
        assert api.calls == []
