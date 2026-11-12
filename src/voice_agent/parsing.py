"""Dates, times and party sizes from spoken words. Deterministic: the LLM never does date arithmetic."""
import datetime as dt
import re

NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
       "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "twenty": 20,
       "thirty": 30, "forty": 40, "forty-five": 45, "a couple": 2}
WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
          "november", "december"]


def _num(tok: str) -> int | None:
    tok = tok.lower()
    return int(tok) if tok.isdigit() else NUM.get(tok)


def parse_party(text: str) -> int | None:
    for m in re.finditer(r"\b(?:for|party of|table for|group of|we are|we're|there(?:'ll| will) be|make (?:it|that))\s+(\d+|[a-z]+)\b"
                         r"(?!\s*(?::|\d|pm|am|p\.m|a\.m|o'?clock|thirty))", text, re.I):
        if _num(m.group(1)):
            return _num(m.group(1))
    m = re.search(r"\b(\d+|[a-z]+)\s+(?:people|persons|guests|of us|adults)\b", text, re.I)
    return _num(m.group(1)) if m and _num(m.group(1)) else None


def parse_time(text: str) -> str | None:
    """'7:30', '7.30 pm', '19:30', 'seven thirty', 'half past seven', 'at nine' -> 'HH:MM'. Evening by default."""
    t = text.lower().replace("o'clock", "").replace("half past", "half-past").replace("p.m.", "pm").replace("a.m.", "am")
    m = re.search(r"\b(\d{1,2})(?::|\.|\s)(\d{2})\s*(am|pm)?", t)
    if m:
        h, mins, ampm = int(m.group(1)), int(m.group(2)), m.group(3) or ""
    elif m := re.search(r"\b(\d{1,2})\s*(am|pm)\b", t):
        h, mins, ampm = int(m.group(1)), 0, m.group(2)
    elif m := re.search(r"\bhalf-past\s+(five|six|seven|eight|nine|ten|\d{1,2})\b", t):
        h, mins, ampm = _num(m.group(1)), 30, ""
    elif m := re.search(r"\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+(thirty|fifteen|forty-five)\b", t):
        h, mins, ampm = NUM[m.group(1)], NUM[m.group(2)], ""
    elif m := re.search(r"\bat\s+(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\d{1,2})\b(?!\s*(people|of us|guests))", t):
        h, mins, ampm = _num(m.group(1)), 0, ""
    else:
        return None
    if h is None or h > 23 or mins > 59:
        return None
    if ampm == "pm" and h < 12 or (not ampm and 1 <= h <= 11):
        h += 12  # a restaurant that opens in the evening: "seven" means 19:00
    return f"{h:02d}:{mins:02d}"


def parse_date(text: str, today: dt.date) -> str | None:
    """'tonight', 'tomorrow', 'Sunday', 'next Friday', 'November 20', 'the 20th', '2026-11-20' -> ISO date."""
    t = text.lower().strip()
    if m := re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", t):
        return dt.date(int(m[1]), int(m[2]), int(m[3])).isoformat()
    if re.search(r"\b(tonight|today|this evening)\b", t):
        return today.isoformat()
    if "day after tomorrow" in t:
        return (today + dt.timedelta(days=2)).isoformat()
    if "tomorrow" in t:
        return (today + dt.timedelta(days=1)).isoformat()
    for i, wd in enumerate(WEEKDAYS):
        if re.search(rf"\b{wd}s?\b", t):
            delta = (i - today.weekday()) % 7
            if "next" in t and delta == 0:
                delta = 7
            return (today + dt.timedelta(days=delta)).isoformat()
    for i, mon in enumerate(MONTHS, 1):
        if m := re.search(rf"\b{mon}\s+(\d{{1,2}})(?:st|nd|rd|th)?\b|\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?{mon}\b", t):
            day = int(m[1] or m[2])
            d = dt.date(today.year, i, day)
            return (d if d >= today else dt.date(today.year + 1, i, day)).isoformat()
    if m := re.search(r"\bthe\s+(\d{1,2})(?:st|nd|rd|th)\b", t):
        day = int(m[1])
        d = today.replace(day=day) if day >= today.day else (today.replace(day=1) + dt.timedelta(days=32)).replace(day=day)
        return d.isoformat()
    return None
