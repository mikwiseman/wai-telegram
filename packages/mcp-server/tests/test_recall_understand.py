"""How recall reads a question: the dates, the people and chats, the kind of thing.

Every chat, person and message here is invented.
"""

from datetime import date

import pytest
from telegram_wai_mcp.recall_understand import parse_window, understand

TODAY = date(2026, 11, 24)  # a Tuesday


def _chat(chat_id, title, chat_type="private", username=None, active="2026-11-01T00:00:00Z"):
    return {
        "id": chat_id,
        "title": title,
        "chat_type": chat_type,
        "username": username,
        "last_activity_at": active,
        "total_messages_synced": 100,
        "indexing_enabled": True,
    }


CHATS = [
    _chat("c-marta", "Марта Кораблёва"),
    _chat("c-timur", "Тимур Сойкин"),
    _chat("c-ilona", "Ilona Brandt"),
    _chat("c-andres", "Андрес Вильегас"),
    _chat("c-dacha", "Дача Сосновка", "group"),
    _chat("c-orion", "Orion Book Club", "supergroup", username="orionbooks"),
    _chat("c-support", "Чат поддержки", "group"),
]


def _who(question, chats=CHATS):
    return [chat["id"] for chat in understand(question, chats, today=TODAY).who]


@pytest.mark.parametrize(
    ("question", "since", "until"),
    [
        ("что писали вчера", "2026-11-23", "2026-11-23"),
        ("что было позавчера", "2026-11-22", "2026-11-22"),
        ("что прислали сегодня", "2026-11-24", "2026-11-24"),
        ("what was sent yesterday", "2026-11-23", "2026-11-23"),
        ("что обсуждали на прошлой неделе", "2026-11-16", "2026-11-22"),
        ("what did we decide last week", "2026-11-16", "2026-11-22"),
        ("что писали на этой неделе", "2026-11-23", "2026-11-24"),
        ("что решили в октябре", "2026-10-01", "2026-10-31"),
        ("что решили в декабре", "2025-12-01", "2025-12-31"),
        ("what was agreed in October 2025", "2025-10-01", "2025-10-31"),
        ("что писали с 12 по 18 октября", "2026-10-12", "2026-10-18"),
        ("переписка 12–18 октября 2025 года", "2025-10-12", "2025-10-18"),
        ("что было с 28 октября по 3 ноября", "2026-10-28", "2026-11-03"),
        ("what was said from October 12 to 18", "2026-10-12", "2026-10-18"),
        ("что было 3 ноября", "2026-11-03", "2026-11-03"),
        ("что было 30 ноября", "2025-11-30", "2025-11-30"),
        ("what was sent on Oct 5th, 2026", "2026-10-05", "2026-10-05"),
        ("что прислали 2026-10-05", "2026-10-05", "2026-10-05"),
        ("счёт от 05.10.2026", "2026-10-05", "2026-10-05"),
        ("что писали за последние 3 дня", "2026-11-21", "2026-11-24"),
        ("what came in the last 2 weeks", "2026-11-10", "2026-11-24"),
        ("что было в прошлом месяце", "2026-10-01", "2026-10-31"),
        ("что было в прошлом году", "2025-01-01", "2025-12-31"),
        ("смета, которую прислали весной", "2026-03-01", "2026-05-31"),
        ("что присылали зимой", "2025-12-01", "2026-02-28"),
        ("что писали с октября", "2026-10-01", None),
        ("what happened since October", "2026-10-01", None),
        ("что обсуждали в пятницу", "2026-11-20", "2026-11-20"),
        ("что было в 2025 году", "2025-01-01", "2025-12-31"),
        ("что мы решили про забор", None, None),
        ("this may be about the fence", None, None),
        ("what was agreed in May", "2026-05-01", "2026-05-31"),
    ],
)
def test_time_window_from_the_question(question, since, until):
    understood = understand(question, [], today=TODAY)
    assert (
        understood.since and understood.since.isoformat(),
        understood.until and understood.until.isoformat(),
    ) == (since, until)


def test_explicit_dates_override_the_question():
    understood = understand(
        "что писали вчера про забор",
        [],
        today=TODAY,
        since=date(2026, 9, 1),
        until=date(2026, 9, 30),
    )
    assert (understood.since, understood.until) == (date(2026, 9, 1), date(2026, 9, 30))
    assert understood.about == ["забор"]

    only_since = understand("что писали вчера", [], today=TODAY, since=date(2026, 9, 1))
    assert (only_since.since, only_since.until) == (date(2026, 9, 1), None)


def test_a_date_is_not_a_search_word_and_an_impossible_date_is_not_a_date():
    assert understand("что писали 12 октября про забор", [], today=TODAY).about == ["забор"]
    for question in ("что было 31 февраля", "что было 100000000 дней назад"):
        assert parse_window(question, TODAY) == (None, None, None)


def test_a_person_called_like_a_month_is_a_person():
    understood = understand("что писала Марта про забор", CHATS, today=TODAY)
    assert (understood.since, understood.until) == (None, None)
    assert [chat["id"] for chat in understood.who] == ["c-marta"]
    assert understood.about == ["забор"]


@pytest.mark.parametrize(
    "question",
    [
        "что я обещал Марте",
        "что пришло от Марты",
        "о чём говорили с Мартой",
        "найди сообщение от марты кораблевой",
        "what did Marta write",
    ],
)
def test_russian_case_endings_and_latin_spelling_find_the_person(question):
    assert _who(question) == ["c-marta"]


@pytest.mark.parametrize(
    ("question", "chat_id"),
    [
        ("что прислал Timur", "c-timur"),
        ("напомни что писала Илона", "c-ilona"),
        ("что сказал тимор", "c-timur"),  # dictation heard one letter wrong
        ("что написал Тимур Сойкен", "c-timur"),
        ("what did Elona Brant say", "c-ilona"),
        ("что нового в orionbooks", "c-orion"),
        ("что писали в орион бук клаб", "c-orion"),
    ],
)
def test_transliteration_and_dictation_slips_still_find_the_chat(question, chat_id):
    assert _who(question) == [chat_id]


def test_a_bot_is_named_like_a_group_not_like_a_person():
    chats = [
        _chat("c-bot", "Забор Сервис", username="zabor_service_bot"),
        _chat("c-marta", "Марта Кораблёва"),
    ]
    assert _who("что Марта писала про забор", chats) == ["c-marta"]
    assert _who("что пришло от забор сервис", chats) == ["c-bot"]


def test_a_person_and_a_group_are_both_understood():
    understood = understand("что Тимур писал в чате дача сосновка", CHATS, today=TODAY)
    assert [(chat["id"], chat["chat_type"]) for chat in understood.who] == [
        ("c-dacha", "group"),
        ("c-timur", "private"),
    ]
    assert understood.about == []


@pytest.mark.parametrize(
    "question",
    [
        "покажи последние сообщения в чате",  # "чат" names no chat
        "что там с дачей",  # one word of a two-word group title
        "пришли адрес офиса",  # "адрес" is one letter away from a contact's name
        "где ссылка на документ",
        "что мы решили про забор",
    ],
)
def test_a_chat_needs_a_distinctive_word(question):
    assert _who(question) == []


def test_a_full_name_picks_one_person_and_a_first_name_keeps_both():
    chats = [
        _chat("c-timur", "Тимур Сойкин", active="2026-10-01T00:00:00Z"),
        _chat("c-timur-2", "Тимур Ладога", active="2026-11-20T00:00:00Z"),
        _chat("c-timurov", "Пётр Тимуров", active="2026-11-23T00:00:00Z"),
    ]
    assert _who("что писал Тимур", chats) == ["c-timur-2", "c-timur"]
    assert _who("что писал Тимур Сойкин", chats) == ["c-timur"]


@pytest.mark.parametrize(
    ("question", "hints"),
    [
        ("найди файл со сметой", {"file"}),
        ("where is the PDF with the estimate", {"file"}),
        ("фото забора от Тимура", {"photo"}),
        ("пришли ссылку на таблицу", {"link"}),
        ("что было в голосовом про забор", {"voice"}),
        ("the voice message about the fence", {"voice"}),
        ("найди сообщение, где Марта прислала адрес", {"address"}),
        ("номер телефона сантехника", {"phone"}),
        ("сколько стоит забор", {"amount"}),
        ("what was the price of the fence", {"amount"}),
        ("что мы решили про забор", set()),
    ],
)
def test_hints_about_the_kind_of_thing(question, hints):
    assert understand(question, CHATS, today=TODAY).hints == hints


def test_about_keeps_the_subject_and_drops_names_dates_and_small_words():
    understood = understand(
        "Что Марта писала мне вчера про договор аренды и ссылку на смету?", CHATS, today=TODAY
    )
    assert understood.about == ["договор", "аренды", "ссылку", "смету"]
    # "ссылку" says what kind of thing to look for; it is not a word to search for.
    assert understood.terms == ["договор", "аренды", "смету"]
    assert understood.phrase == "договор аренды"


def test_a_quoted_phrase_is_searched_as_written():
    understood = understand("кто писал «забор до мая» в чате", CHATS, today=TODAY)
    assert understood.phrase == "забор до мая"
    assert (understood.since, understood.until) == (None, None)
