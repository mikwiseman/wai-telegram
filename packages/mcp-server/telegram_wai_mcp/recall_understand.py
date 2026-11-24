"""Read a question the way it was asked: when, who, about what, what kind of thing.

Plain rules, no model. The question may be dictated, so names are compared
through a spelling-tolerant key: lower case, ё as е, Cyrillic and Latin brought
to one alphabet, Russian case endings dropped, and a letter or two of slack.
Dates follow the archive's convention: a day is a UTC day.
"""

import calendar
import re
from dataclasses import dataclass
from datetime import date, timedelta
from functools import lru_cache
from typing import Any, Callable

_WORD = re.compile(r"[^\W_]+")
_CYRILLIC = re.compile(r"[а-я]")
_QUOTED = re.compile(r"[«\"“„]([^«»\"“”„]{2,120})[»\"”“]")

# Words that carry no subject: grammar, the asking itself, and the vocabulary of
# messaging ("wrote", "sent", "chat"). What is left is what the question is about.
# Written as they are compared, with е for ё.
_STOP = frozenset(
    """
    и в во на с со к ко у о об обо от до из за по под над при про для без через между не ни да
    нет а но или ли же бы то это эта этот эти этого этой этом эту тот та те там тут здесь где
    куда когда как что чего чему чем кто кого кому кем какой какая какое какие какую каком
    сколько чей чья который которая которое которые которую котором которой я мне меня мной
    мы нам нас нами ты тебе тебя вы вам вас он она оно они его ее их ему ей им ним ней нему
    них мой моя мое мои моей моего наш наша наше наши нашей нашего свой своя свои себя себе
    был была было были есть будет будут уже еще тоже также только очень так такой вот ну
    просто если чтобы потому все всех всем весь вся раз мол типа пожалуйста надо нужно можно
    именно
    вообще либо нибудь может быть мог могла могли хочу хотел хотела
    найди найти найдите покажи показать напомни напомнить вспомни вспомнить скажи расскажи дай
    дать подскажи посмотри глянь поищи ищи узнай
    писал писала писали написал написала написали пишет пишут писать сказал сказала сказали
    говорил говорила говорили говорит прислал прислала прислали присылал присылала присылали
    пришли пришло пришел пришла отправил отправила отправили отправлял отправляла
    скинул скинула скинули кидал кидала кинул кинула переслал переслала переслали упоминал
    упоминала упоминали обсуждал обсуждала обсуждали обсудили ответил ответила ответили
    спрашивал спрашивала спросил спросила спросили решил решила решили договорился
    договорилась договорились согласовали обещал обещала обещали
    сообщение сообщения сообщении сообщений сообщению сообщением переписка переписке переписки
    переписку чат чате чата чату чаты чатах группа группе группы группу канал канале канала
    телеграм телеграме телеграмме диалог диалоге разговор разговоре
    год году года лет неделя неделе неделю недели месяц месяце месяца день дня дней числа
    число вчера сегодня позавчера недавно давно раньше позже потом сейчас назад прошлый
    прошлая прошлой прошлом прошлую этот последний последняя последние последнюю последнее
    последних
    a an the and or but of in on at to for from with by about as is are was were be been do does
    did have has had i me my we us our you your he him his she her it its they them their this
    that these those what which who whom whose when where why how there here not no yes if so
    any some all just also can could would should will please much many
    find show tell give get look search remind recall remember
    message messages chat chats group channel conversation telegram
    wrote write written said say says told sent send sends shared share mentioned mention
    discussed asked replied texted text agreed agree decided decide promised
    today yesterday week weeks month months year years day days ago last past recent recently
    since until before after during
    """.split()
)
# Words a chat title may hold that say nothing about which chat it is.
_PLAIN_TITLE = _STOP | frozenset("чат группа канал команда общий рабочий team work general".split())

# The kind of thing asked for. A pointer ("the file", "the link") says what to
# look for and is not itself a word to search; the rest are also subject words.
_HINTS: dict[str, tuple[re.Pattern[str], bool]] = {
    "file": (
        re.compile(
            r"файл\w*|документ\w*|вложени\w*|pdf|пдф\w*|скан(?:ы|а|у|е|ом|ов)?|files?"
            r"|documents?|attachments?|docx?|xlsx?"
        ),
        True,
    ),
    "photo": (
        re.compile(
            r"фото\w*|фотк\w*|картинк\w*|снимк\w*|снимок|скрин\w*|photos?|pictures?|pics?"
            r"|images?|screenshots?"
        ),
        True,
    ),
    "link": (re.compile(r"ссылк\w*|ссылочк\w*|линк(?:а|у|и|е|ом|ов)?|links?|urls?"), True),
    "voice": (
        re.compile(r"голосов\w*|войс\w*|аудио\w*|кружок|кружоч\w*|voice\w*|audio\w*"),
        True,
    ),
    "address": (re.compile(r"адрес\w*|address\w*|локаци\w*|геолокаци\w*|locations?"), False),
    "phone": (re.compile(r"телефон\w*|phones?|мобильн\w*"), False),
    "amount": (
        re.compile(
            r"сумм\w*|цен[ауые]|ценой|стоимост\w*|стои[тл]\w*|стоят|оплат\w*|заплати\w*"
            r"|price\w*|costs?|amounts?"
        ),
        False,
    ),
}


def fold(text: str) -> str:
    """Lower case with ё as е: how words are compared, never how they are shown."""
    return text.lower().replace("ё", "е")


def words(text: str) -> list[str]:
    return _WORD.findall(fold(text))


# --- names -------------------------------------------------------------------

# Longest first; an ending is dropped only when three letters remain.
_RU_ENDINGS = (
    "иями ями ами ого его ому ему ыми ими ия ие ий ию ии ой ей ою ею ом ем ым им ах ях ую юю "
    "ая яя ое ее ые ых их ам ям а я у ю ы и е о й ь"
).split()
# ъ and ь have no sound of their own: the two double spaces are their empty places.
_TO_LATIN = dict(
    zip(
        "абвгдежзийклмнопрстуфхцчшщъыьэюя",
        "a b v g d e zh z i i k l m n o p r s t u f h ts ch sh sh  i  e iu ia".split(" "),
        strict=True,
    )
)
_LATIN_SPELLINGS = (
    ("shch", "sh"),
    ("sch", "sh"),
    ("kh", "h"),
    ("ph", "f"),
    ("ck", "k"),
    ("th", "t"),
    ("x", "ks"),
    ("w", "v"),
    ("q", "k"),
    ("j", "i"),
    ("y", "i"),
)


def stem(word: str) -> str:
    """A folded word without its Russian case ending or its English plural."""
    if _CYRILLIC.search(word):
        for ending in _RU_ENDINGS:
            if word.endswith(ending) and len(word) - len(ending) >= 3:
                return word[: -len(ending)]
        return word
    return word[:-1] if len(word) > 3 and word.endswith("s") else word


@lru_cache(maxsize=20000)
def name_key(word: str) -> str:
    """One spelling for a folded name however it was typed, heard or declined."""
    if _CYRILLIC.search(word):
        key = "".join(_TO_LATIN.get(letter, letter) for letter in stem(word))
    else:
        key = word
        for spelling, plain in _LATIN_SPELLINGS:
            key = key.replace(spelling, plain)
        key = re.sub(r"c(?=[ei])", "s", key)
        key = re.sub(r"c(?!h)", "k", key)
        if key.startswith("ie"):
            key = key[1:]
    bare = key.rstrip("aeiou")
    return bare if len(bare) >= 2 else key


def _distance(a: str, b: str) -> int:
    previous = list(range(len(b) + 1))
    for row, letter_a in enumerate(a, 1):
        current = [row]
        for column, letter_b in enumerate(b, 1):
            current.append(
                min(
                    previous[column] + 1,
                    current[column - 1] + 1,
                    previous[column - 1] + (letter_a != letter_b),
                )
            )
        previous = current
    return previous[-1]


def same_name(asked: str, known: str) -> int:
    """How surely two folded words are one name: 3 as written, 2 same key, 1 close, 0 no."""
    if asked == known:
        return 3
    if len(asked) < 3 or len(known) < 3:
        return 0
    a, b = name_key(asked), name_key(known)
    if a == b:
        return 2
    shortest = min(len(a), len(b))
    slack = 0 if shortest < 4 else 1 if shortest <= 6 else 2
    if slack and a[0] == b[0] and abs(len(a) - len(b)) <= slack and _distance(a, b) <= slack:
        return 1
    return 0


def is_person(chat: dict[str, Any]) -> bool:
    """A private chat with a person. A bot is private too, and its username says so."""
    return chat.get("chat_type") == "private" and not fold(
        str(chat.get("username") or "")
    ).endswith("bot")


@dataclass(frozen=True)
class _Match:
    chat: dict[str, Any]
    score: tuple[int, int]  # title words found, and how surely
    words: frozenset[int]  # the question words that named it


def _chat_match(
    chat: dict[str, Any], asked: dict[int, str], pairs: dict[tuple[int, int], str]
) -> _Match | None:
    title = words(str(chat.get("title") or ""))
    telling = [word for word in title if word not in _PLAIN_TITLE and not word.isdigit()]
    found = 0
    surely = 0
    naming: set[int] = set()
    distinctive = False
    for known in telling:
        best, by = 0, ()
        for index, word in asked.items():
            quality = same_name(word, known)
            if quality > best:
                best, by = quality, (index,)
        if best < 2 and len(known) >= 6:
            # A name heard as two words: "норд винд" for a chat titled NordWind.
            for pair, joined in pairs.items():
                quality = same_name(joined, known)
                if quality > best:
                    best, by = quality, pair
        if best:
            found += 1
            surely += best
            naming.update(by)
            distinctive = distinctive or len(known) >= 3
    username = fold(str(chat.get("username") or "")).removeprefix("@")
    for index, word in asked.items():
        if username and word == username:
            return _Match(chat, (found + 2, surely + 6), frozenset({*naming, index}))
    # A person is named by one word of the name; a group, a channel or a bot by
    # two, when its title has them: their titles are made of everyday words.
    full_words = sum(1 for word in telling if len(word) >= 3)
    needed = 1 if is_person(chat) else min(full_words, 2)
    if not distinctive or found < needed:
        return None
    return _Match(chat, (found, surely), frozenset(naming))


def _match_who(
    chats: list[dict[str, Any]], asked: dict[int, str]
) -> tuple[list[dict[str, Any]], set[int]]:
    """The chats and people the question names, and the question words that named them."""
    if not asked:
        return [], set()
    pairs = {
        (index, index + 1): asked[index] + asked[index + 1] for index in asked if index + 1 in asked
    }
    matches = [match for chat in chats if (match := _chat_match(chat, asked, pairs))]
    matches.sort(
        key=lambda match: (match.score, str(match.chat.get("last_activity_at") or "")),
        reverse=True,
    )
    chosen: list[_Match] = []
    claimed: dict[int, tuple[int, int]] = {}
    for match in matches:
        # A weaker reading of words a better match already explained is dropped;
        # an equal one is kept, because "Тимур" may honestly be either Тимур.
        if all(claimed.get(index, (0, 0)) > match.score for index in match.words):
            continue
        chosen.append(match)
        for index in match.words:
            claimed[index] = max(claimed.get(index, match.score), match.score)
        if len(chosen) == 3:
            break
    return [match.chat for match in chosen], set().union(*(match.words for match in chosen))


# --- dates -------------------------------------------------------------------

_RU_MONTHS = (
    "январ[ьяею]",
    "феврал[ьяею]",
    "март[ае]?",
    "апрел[ьяею]",
    "ма[йяе]",
    "июн[ьяею]",
    "июл[ьяею]",
    "август[ае]?",
    "сентябр[ьяею]",
    "октябр[ьяею]",
    "ноябр[ьяею]",
    "декабр[ьяею]",
)
_EN_MONTHS = (
    "january february march april may june july august september october november december"
).split()
_EN_SHORT = "jan feb mar apr may jun jul aug sept? oct nov dec".split()
_MONTH_NUMBER = [
    (re.compile(f"{ru}|{en}|{short}"), number)
    for number, (ru, en, short) in enumerate(zip(_RU_MONTHS, _EN_MONTHS, _EN_SHORT), 1)
]
_RU = "|".join(_RU_MONTHS)
_EN = "|".join(_EN_MONTHS)
_ANY = f"{_RU}|{_EN}|{'|'.join(_EN_SHORT)}"
_DAY = r"(\d{1,2})(?:-?(?:го|ое|е|st|nd|rd|th))?"
_YEAR = r"(?:,?\s+(20\d{2})(?!\d)(?:\s*(?:года|году|год|г\b\.?))?)?"
_SINCE = {"с", "со", "после", "since", "after"}
_UNTIL = {"до", "по", "until", "till", "before"}
_NOT_BEFORE_MAY = {"this", "of", "for"}
_WEEKDAYS = (
    "понедельник|monday",
    "вторник|tuesday",
    "сред[ау]|wednesday",
    "четверг|thursday",
    "пятниц[ау]|friday",
    "суббот[ау]|saturday",
    "воскресенье|sunday",
)
_SEASONS = (
    ("весной|spring", 3),
    ("летом|summer", 6),
    ("осенью|autumn|fall", 9),
    ("зимой|winter", 12),
)
_UNIT_DAYS = (("дн|день|day", 1), ("недел|week", 7), ("месяц|month", 30), ("год|year", 365))

_ISO = re.compile(r"\b(20\d{2})-(\d{2})-(\d{2})\b")
_DOTTED = re.compile(r"\b(\d{1,2})\.(\d{1,2})\.(\d{4}|\d{2})\b")
_WRITTEN = r"(?:\b20\d{2}-\d{2}-\d{2}\b|\b\d{1,2}\.\d{1,2}\.(?:\d{4}|\d{2})\b)"

_Window = tuple[date | None, date | None]


def _month(word: str) -> int:
    return next(number for pattern, number in _MONTH_NUMBER if pattern.fullmatch(word))


def _year(text: str | None) -> int | None:
    return int(text) if text else None


def _on(today: date, day: int, month: int, year: int | None) -> date | None:
    """A day of a month; with no year said, the latest one that has already come."""
    try:
        if year:
            return date(year, month, day)
        this_year = date(today.year, month, day)
        return this_year if this_year <= today else date(today.year - 1, month, day)
    except ValueError:
        return None


def _whole_month(today: date, month: int, year: int | None) -> tuple[date, date]:
    if year is None:
        year = today.year if month <= today.month else today.year - 1
    return date(year, month, 1), date(year, month, calendar.monthrange(year, month)[1])


def _open_ended(preposition: str | None, start: date, end: date) -> _Window:
    """A period, or one side of it when the question says "since" or "until"."""
    if preposition in _SINCE:
        return start, None
    if preposition in _UNTIL:
        # "по октябрь" and "до 5 ноября" include what they name; "до октября" and
        # "before" stop short of it.
        inclusive = preposition == "по" or (start == end and preposition != "before")
        return None, end if inclusive else start - timedelta(days=1)
    return start, end


def _two_months(today: date, m: re.Match[str], order: str) -> _Window | None:
    parts = dict(zip(order.split(), m.groups(), strict=True))
    end = _on(today, int(parts["d2"]), _month(parts["m2"]), _year(parts["y"]))
    if end is None:
        return None
    month, day = _month(parts["m1"]), int(parts["d1"])
    try:
        start = date(end.year - ((month, day) > (end.month, end.day)), month, day)
    except ValueError:
        return None
    return start, end


def _one_month(today: date, d1: str, d2: str, month: str, year: str | None) -> _Window | None:
    end = _on(today, int(d2), _month(month), _year(year))
    if end is None or int(d1) > end.day or int(d1) < 1:
        return None
    return end.replace(day=int(d1)), end


def _one_day(today: date, preposition, day: str, month: str, year: str | None) -> _Window | None:
    found = _on(today, int(day), _month(month), _year(year))
    return _open_ended(preposition, found, found) if found else None


def _written_dates(today: date, m: re.Match[str]) -> _Window | None:
    written = m.group()
    found = [(int(year), int(month), int(day)) for year, month, day in _ISO.findall(written)]
    found += [
        (int(year) + (2000 if len(year) == 2 else 0), int(month), int(day))
        for day, month, year in _DOTTED.findall(written)
    ]
    try:
        days = sorted(date(*parts) for parts in found)
    except ValueError:
        return None
    return days[0], days[-1]


def _only_month(today: date, preposition, month: str, year: str | None) -> _Window | None:
    # A bare month word may be a person (Марта, Майя, August): it is a date only
    # when a preposition or a year says so. And "this may be" is not about May.
    if not year and (not preposition or (month == "may" and preposition in _NOT_BEFORE_MAY)):
        return None
    return _open_ended(preposition, *_whole_month(today, _month(month), _year(year)))


def _only_year(today: date, preposition, year: str) -> _Window:
    return _open_ended(preposition, date(int(year), 1, 1), date(int(year), 12, 31))


def _back(today: date, days: int) -> _Window:
    return today - timedelta(days=days), today


def _rolling(today: date, count: str | None, unit: str) -> _Window:
    size = next(days for names, days in _UNIT_DAYS if re.match(names, unit))
    return _back(today, int(count or 1) * size)


def _weekday(today: date, strict: str | None, name: str) -> _Window:
    wanted = next(i for i, names in enumerate(_WEEKDAYS) if re.fullmatch(names, name))
    back = (today.weekday() - wanted) % 7
    day = today - timedelta(days=back or (7 if strict else 0))
    return day, day


def _season(today: date, name: str) -> _Window:
    first = next(month for names, month in _SEASONS if re.fullmatch(names, name))
    year = today.year if date(today.year, first, 1) <= today else today.year - 1
    last_month = (first + 1) % 12 + 1
    last_year = year + (last_month < first)
    return (
        date(year, first, 1),
        date(last_year, last_month, calendar.monthrange(last_year, last_month)[1]),
    )


def _this_week(today: date, weeks_back: int) -> _Window:
    monday = today - timedelta(days=today.weekday() + 7 * weeks_back)
    return monday, today if weeks_back == 0 else monday + timedelta(days=6)


def _last_month(today: date) -> _Window:
    end = today.replace(day=1) - timedelta(days=1)
    return end.replace(day=1), end


_Rule = tuple[re.Pattern[str], Callable[[date, re.Match[str]], _Window | None]]


def _rule(pattern: str, read: Callable[..., _Window | None]) -> _Rule:
    return re.compile(pattern), lambda today, m: read(today, *m.groups())


_PREPOSITION = r"(?:(с|со|после|до|по|since|after|until|till|before)\s+)?"
# First rule to read a date wins, so the more specific shapes come first.
_RULES: tuple[_Rule, ...] = (
    (
        re.compile(rf"\b(?:с|со)\s+{_DAY}\s+({_RU})\s+(?:по|до)\s+{_DAY}\s+({_RU}){_YEAR}"),
        lambda today, m: _two_months(today, m, "d1 m1 d2 m2 y"),
    ),
    (
        re.compile(
            rf"\b(?:from\s+)?({_ANY})\s+{_DAY}\s+(?:to|through|until|till)\s+({_ANY})\s+{_DAY}"
            rf"(?!\d){_YEAR}"
        ),
        lambda today, m: _two_months(today, m, "m1 d1 m2 d2 y"),
    ),
    _rule(
        rf"\b{_DAY}\s*(?:по|до|to|and|through|[-–—])\s*{_DAY}\s+(?:of\s+)?({_ANY})\b{_YEAR}",
        _one_month,
    ),
    _rule(
        rf"\b({_ANY})\s+{_DAY}\s*(?:to|through|until|till|[-–—])\s*{_DAY}(?!\d){_YEAR}",
        lambda today, month, d1, d2, year: _one_month(today, d1, d2, month, year),
    ),
    # One written date is that day; two are the first and last day of a period.
    (re.compile(rf"{_WRITTEN}(?:.*{_WRITTEN})?"), _written_dates),
    _rule(rf"\b{_PREPOSITION}{_DAY}\s+(?:of\s+)?({_ANY})\b{_YEAR}", _one_day),
    _rule(
        rf"\b{_PREPOSITION}({_ANY})\s+{_DAY}(?!\d){_YEAR}",
        lambda today, preposition, month, day, year: _one_day(today, preposition, day, month, year),
    ),
    _rule(
        rf"\b(?:(в|во|за|с|со|после|до|по|in|during|last|this|since|after|until|till|before|of|for)"
        rf"\s+)?({_RU}|{_EN})\b{_YEAR}",
        _only_month,
    ),
    _rule(
        r"\b(в|за|с|со|после|до|по|in|during|since|after|until|till|before)\s+(20\d{2})\b"
        r"(?:\s*(?:года|году|год|г\b\.?))?",
        _only_year,
    ),
    _rule(r"\b()(20\d{2})\s*(?:года|году|год)\b", _only_year),
    _rule(r"\bпозавчера\b|\bday before yesterday\b", lambda today: (today - timedelta(2),) * 2),
    _rule(r"\bвчера\w*|\byesterday\b", lambda today: (today - timedelta(1),) * 2),
    _rule(r"\bсегодня\w*|\btoday\b", lambda today: (today, today)),
    _rule(
        r"\b(?:на|за|в)\s+(?:эту|этой)\s+недел[еюи]\b|\bthis week\b",
        lambda today: _this_week(today, 0),
    ),
    _rule(
        r"\b(?:на|за|в)\s+(?:прошлой|прошлую|предыдущ\w+)\s+недел[еюи]\b|\b(?:last|previous) week\b",
        lambda today: _this_week(today, 1),
    ),
    _rule(
        r"\b(?:в|за)\s+(?:этом|этот)\s+месяц[еа]?\b|\bthis month\b",
        lambda today: (today.replace(day=1), today),
    ),
    _rule(
        r"\b(?:в|за)\s+(?:прошлом|прошлый|предыдущ\w+)\s+месяц[еа]?\b|\b(?:last|previous) month\b",
        _last_month,
    ),
    _rule(
        r"\b(?:в|за)\s+(?:этом|этот)\s+год[уа]?\b|\bthis year\b",
        lambda today: (date(today.year, 1, 1), today),
    ),
    _rule(
        r"\b(?:в|за)\s+(?:прошлом|прошлый)\s+год[уа]?\b|\blast year\b",
        lambda today: (date(today.year - 1, 1, 1), date(today.year - 1, 12, 31)),
    ),
    _rule(r"\bза\s+последн\w+\s+(\d+)\s+(дн|недел|месяц)\w*", _rolling),
    _rule(r"\b(?:last|past)\s+(\d+)\s+(day|week|month)s?\b", _rolling),
    _rule(r"\bза\s+(?:последн\w+\s+)?()(недел[юи]|месяца?|года?)\b", _rolling),
    _rule(r"\b(?:the\s+)?past\s+()(week|month|year)\b", _rolling),
    _rule(
        r"\b(\d+)\s+(?:дн\w+|день|days?)\s+(?:назад|ago)\b",
        lambda today, count: (today - timedelta(int(count)),) * 2,
    ),
    _rule(
        rf"\b(?:в|во|on|(last))\s+(?:(прошл)\w+\s+)?({'|'.join(_WEEKDAYS)})\b",
        lambda today, last, past, name: _weekday(today, last or past, name),
    ),
    _rule(r"\b(?:(?:этой|прошлой)\s+)?(весной|летом|осенью|зимой)\b", _season),
    _rule(r"\b(?:in\s+(?:the\s+)?|last\s+|this\s+)(spring|summer|autumn|fall|winter)\b", _season),
)


def parse_window(text: str, today: date) -> tuple[date | None, date | None, tuple[int, int] | None]:
    """The time window a folded question names, and where in the text it says so."""
    for pattern, read in _RULES:
        for m in pattern.finditer(text):
            try:
                window = read(today, m)
            except (ValueError, OverflowError):  # "100000000 дней назад"
                continue
            if window:
                return window[0], window[1], m.span()
    return None, None, None


# --- the whole question --------------------------------------------------------


@dataclass(frozen=True)
class Understood:
    since: date | None
    until: date | None
    who: list[dict[str, Any]]  # the chats named, as the archive lists them
    about: list[str]  # the subject, in the question's own words
    terms: list[str]  # the words of the subject worth searching for
    phrase: str | None  # words to find exactly as written, side by side
    hints: frozenset[str]  # file, photo, link, voice, address, phone, amount


def understand(
    question: str,
    chats: list[dict[str, Any]],
    *,
    today: date,
    since: date | None = None,
    until: date | None = None,
) -> Understood:
    text = question.lower()
    folded = text.replace("ё", "е")
    quoted = _QUOTED.search(text)
    # Words inside quotes are a quotation, not a date to work out.
    dated = (
        folded
        if not quoted
        else (
            folded[: quoted.start()]
            + " " * (quoted.end() - quoted.start())
            + folded[quoted.end() :]
        )
    )
    said_since, said_until, span = parse_window(dated, today)
    if since or until:
        said_since, said_until = since, until

    tokens = list(_WORD.finditer(text))
    hints: set[str] = set()
    pointers: set[int] = set()
    asked: dict[int, str] = {}
    subject: list[int] = []
    for index, word in enumerate(tokens):
        if span and span[0] <= word.start() < span[1]:
            continue
        key = fold(word.group())
        kinds = [(name, only) for name, (pattern, only) in _HINTS.items() if pattern.fullmatch(key)]
        hints.update(name for name, _ in kinds)
        if any(only for _, only in kinds):
            pointers.add(index)
        if key in _STOP or len(key) < 2:
            continue
        if not kinds:
            asked[index] = key
        if len(key) >= 3:
            subject.append(index)

    who, named = _match_who(chats, asked)
    subject = [index for index in subject if index not in named]
    about = list(dict.fromkeys(tokens[index].group() for index in subject))
    term_words = [index for index in subject if index not in pointers]
    terms = list(dict.fromkeys(tokens[index].group() for index in term_words))

    phrase = quoted.group(1).strip() if quoted else None
    if not phrase:
        run: list[int] = []
        best: list[int] = []
        for index in term_words:
            adjoining = (
                run
                and index == run[-1] + 1
                and not text[tokens[run[-1]].end() : tokens[index].start()].strip()
            )
            run = [*run, index] if adjoining else [index]
            if len(run) > len(best):
                best = run
        if 2 <= len(best) <= 6:
            phrase = text[tokens[best[0]].start() : tokens[best[-1]].end()]

    return Understood(
        since=said_since,
        until=said_until,
        who=who,
        about=about,
        terms=terms,
        phrase=phrase,
        hints=frozenset(hints),
    )
