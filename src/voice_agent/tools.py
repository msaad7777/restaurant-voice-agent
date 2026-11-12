"""The agent's tools. Plain Python over SQLite: no LLM in here.

Every tool returns a dict. Allergen answers carry a `say` sentence built from the
verified allergen fields; the guard in agent.py only lets those sentences through.
"""
import datetime as dt
import difflib
import json
import re
import sqlite3

from . import data
from .parsing import parse_date, parse_time

ALLERGEN_WORDS = {
    "milk": "milk", "dairy": "milk", "lactose": "milk", "cheese": "milk", "butter": "milk", "cream": "milk",
    "egg": "egg", "eggs": "egg",
    "fish": "fish",
    "shellfish": "shellfish", "crustacean": "shellfish", "crustaceans": "shellfish", "mollusc": "shellfish",
    "molluscs": "shellfish", "shrimp": "shellfish", "prawn": "shellfish", "prawns": "shellfish", "squid": "shellfish",
    "tree nut": "tree_nuts", "tree nuts": "tree_nuts", "tree_nuts": "tree_nuts", "almond": "tree_nuts",
    "almonds": "tree_nuts", "hazelnut": "tree_nuts", "hazelnuts": "tree_nuts", "walnut": "tree_nuts",
    "walnuts": "tree_nuts", "pine nut": "tree_nuts", "pine nuts": "tree_nuts", "cashew": "tree_nuts",
    "cashews": "tree_nuts", "pecan": "tree_nuts", "pistachio": "tree_nuts",
    "peanut": "peanuts", "peanuts": "peanuts",
    "nut": "nuts", "nuts": "nuts",
    "gluten": "gluten", "wheat": "gluten", "coeliac": "gluten", "celiac": "gluten",
    "soy": "soy", "soya": "soy",
    "sesame": "sesame", "tahini": "sesame",
}
LABEL = {"milk": "milk or dairy", "egg": "egg", "fish": "fish", "shellfish": "shellfish", "tree_nuts": "tree nuts",
         "peanuts": "peanuts", "gluten": "gluten", "soy": "soy", "sesame": "sesame"}


class ToolError(Exception):
    """Raised for injected faults; the agent must not hide it."""


def _hm(t: str) -> int:
    h, m = map(int, t.split(":"))
    return h * 60 + m


def _fmt(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def spoken_time(t: str) -> str:
    h, m = map(int, t.split(":"))
    suffix = "pm" if h >= 12 else "am"
    h12 = h - 12 if h > 12 else (12 if h == 0 else h)
    return f"{h12}{':' + f'{m:02d}' if m else ''} {suffix}"


def spoken_date(d: dt.date, today: dt.date) -> str:
    if d == today:
        return "tonight"
    if d == today + dt.timedelta(days=1):
        return "tomorrow"
    return d.strftime("%A, %B ") + str(d.day)


class Tools:
    def __init__(self, db: sqlite3.Connection, today: dt.date, faults: set[str] | None = None):
        self.db = db
        self.today = today
        self.faults = faults or set()

    def _fault(self, name: str):
        if name in self.faults:
            raise ToolError(f"{name} is temporarily unavailable")

    # -- hours ---------------------------------------------------------------
    def get_hours(self, date: str | None = None) -> dict:
        self._fault("get_hours")
        if date:
            iso = parse_date(date, self.today)
            if iso is None:
                return {"error": "unclear_date", "heard": date}
            d = dt.date.fromisoformat(iso)
            h = data.HOURS[d.weekday()]
            return {"date": iso, "weekday": d.strftime("%A"), "open": h is not None,
                    "opens": h[0] if h else None, "closes": h[1] if h else None,
                    "last_booking": _fmt(_hm(h[1]) - data.LAST_BOOKING_BEFORE_CLOSE_MIN) if h else None}
        week = {}
        for wd in range(7):
            h = data.HOURS[wd]
            week[dt.date(2024, 1, 1 + wd).strftime("%A")] = f"{h[0]}-{h[1]}" if h else "closed"
        return {"week": week}

    # -- menu ----------------------------------------------------------------
    def _rows(self):
        return self.db.execute("SELECT * FROM dishes").fetchall()

    def find_dish(self, name: str):
        q = name.lower().strip()
        q = re.sub(r"^(the|your|a)\s+", "", q)
        best, best_score = None, 0.0
        for r in self._rows():
            for cand in [r["name"].lower()] + json.loads(r["aliases"]):
                score = 1.0 if q == cand else (0.9 if cand in q or q in cand else difflib.SequenceMatcher(None, q, cand).ratio())
                if score > best_score:
                    best, best_score = r, score
        return best if best_score >= 0.75 else None

    def search_menu(self, query: str | None = None, category: str | None = None, dietary: str | None = None) -> dict:
        """Dish names, prices, descriptions and diet tags. Deliberately no allergen data."""
        self._fault("search_menu")
        out = []
        for r in self._rows():
            tags = json.loads(r["tags"])
            cat = (category or "").lower().rstrip("s")
            diet = (dietary or "").lower()
            if cat in ("starter", "main", "side", "dessert") and r["category"] != cat:
                continue
            if diet in ("vegetarian", "vegan") and diet not in tags and not (diet == "vegetarian" and "vegan" in tags):
                continue
            text = " ".join([r["name"], r["description"], r["aliases"], r["ingredients"]]).lower()
            if query and not any(w in text for w in re.findall(r"[a-z]{3,}", query.lower())):
                continue
            out.append({"name": r["name"], "category": r["category"], "price": r["price"],
                        "description": r["description"], "tags": tags})
        return {"dishes": out, "count": len(out)}

    # -- allergens -----------------------------------------------------------
    def check_allergen(self, dish: str, allergen: str) -> dict:
        self._fault("check_allergen")
        r = self.find_dish(dish)
        key = ALLERGEN_WORDS.get(allergen.lower().strip().rstrip("."), None)
        if r is None:
            return {"status": "unknown_dish", "dish": dish,
                    "say": f"I couldn't find {dish} on our menu, so I can't give allergen information for it."}
        name = r["name"]
        if key is None:
            return {"status": "unknown_allergen", "dish": name, "allergen": allergen,
                    "say": f"I don't have verified information about {allergen} for the {name}. "
                           "I can connect you with a member of staff who can check with the kitchen."}
        if not r["allergens_verified"]:
            return {"status": "unverified", "dish": name, "allergen": key,
                    "next_step": "If the caller needs certainty or a guarantee, call transfer_to_staff.",
                    "say": f"I can't confirm allergens for the {name}, because its allergen information hasn't been "
                           "verified. I can connect you with a member of staff who can check with the kitchen."}
        keys = ["tree_nuts", "peanuts"] if key == "nuts" else [key]
        contains = [k for k in keys if k in json.loads(r["contains"])]
        may = [k for k in keys if k in json.loads(r["may_contain"]) and k not in contains]
        label = " or ".join(LABEL[k] for k in keys)
        if contains:
            extra = " (pine nuts)" if "tree_nuts" in contains and "pine nuts" in r["ingredients"] else ""
            status, say = "contains", f"Yes, the {name} contains {' and '.join(LABEL[k] for k in contains)}{extra}."
        elif may:
            status, say = "may_contain", (f"The {name} has no {label} as an ingredient, but it may contain "
                                          f"{' and '.join(LABEL[k] for k in may)} from cross-contact, so we can't "
                                          "guarantee it's free of it.")
        else:
            status, say = "not_listed", f"According to our verified allergen list, the {name} does not contain {label}."
        return {"status": status, "dish": name, "allergen": key, "say": say}

    # -- reservations --------------------------------------------------------
    def resolve(self, date: str, time: str) -> tuple[str | None, str | None]:
        iso = parse_date(str(date), self.today)
        time = str(time).strip()
        if re.fullmatch(r"\d{1,2}(\s*o'?clock)?", time):  # a bare "7" on a restaurant line means 7 pm
            time = "at " + time
        hm = time if re.fullmatch(r"\d{2}:\d{2}", time) else parse_time(time)
        return iso, hm

    def _validate(self, date: str, time: str, party_size: int) -> tuple[dict | None, dt.date, int]:
        try:
            d = dt.date.fromisoformat(date)
            t = _hm(time)
        except (ValueError, AttributeError, TypeError):
            return {"available": False, "reason": "unclear_date_or_time",
                    "next_step": "Ask the caller to repeat the day or time."}, None, 0
        if d < self.today:
            return {"available": False, "reason": "date_in_past"}, d, t
        if (d - self.today).days > 60:
            return {"available": False, "reason": "too_far_ahead", "max_days_ahead": 60}, d, t
        if party_size < 1:
            return {"available": False, "reason": "invalid_party_size"}, d, t
        if party_size > data.MAX_PARTY:
            return {"available": False, "reason": "large_party", "max_party": data.MAX_PARTY,
                    "next_step": "transfer_to_staff for large parties and events"}, d, t
        h = data.HOURS[d.weekday()]
        if h is None:
            return {"available": False, "reason": "closed", "weekday": d.strftime("%A")}, d, t
        last = _hm(h[1]) - data.LAST_BOOKING_BEFORE_CLOSE_MIN
        if t < _hm(h[0]) or t > last or t % data.SLOT_MIN:
            return {"available": False, "reason": "outside_booking_hours", "first_booking": h[0],
                    "last_booking": _fmt(last), "slot_minutes": data.SLOT_MIN}, d, t
        return None, d, t

    def _free_table(self, date: str, t: int, party_size: int) -> str | None:
        busy = set()
        for r in self.db.execute("SELECT time, table_id FROM reservations WHERE date = ?", (date,)):
            s = _hm(r["time"])
            if s < t + data.SEATING_MIN and t < s + data.SEATING_MIN:
                busy.add(r["table_id"])
        for tid, seats in sorted(data.TABLES, key=lambda x: x[1]):
            if seats >= party_size and tid not in busy:
                return tid
        return None

    def check_availability(self, date: str, time: str, party_size: int) -> dict:
        self._fault("check_availability")
        party_size = int(party_size)
        heard = {"heard_date": date, "heard_time": time}
        date, time = self.resolve(date, time)
        err, d, t = self._validate(date, time, party_size)
        if err:
            return {**err, "date": date, "time": time, "party_size": party_size, **heard}
        when = {"date": date, "weekday": d.strftime("%A"), "spoken": f"{spoken_date(d, self.today)} at {spoken_time(time)}"}
        if self._free_table(date, t, party_size):
            return {"available": True, **when, "time": time, "party_size": party_size}
        h = data.HOURS[d.weekday()]
        slots = range(_hm(h[0]), _hm(h[1]) - data.LAST_BOOKING_BEFORE_CLOSE_MIN + 1, data.SLOT_MIN)
        alts = sorted((s for s in slots if self._free_table(date, s, party_size)), key=lambda s: abs(s - t))[:3]
        return {"available": False, "reason": "fully_booked", **when, "time": time, "party_size": party_size,
                "alternatives": [spoken_time(_fmt(s)) for s in sorted(alts)]}

    def create_reservation(self, name: str, party_size: int, date: str, time: str) -> dict:
        self._fault("create_reservation")
        party_size = int(party_size)
        name = name.strip()
        if not name:
            return {"created": False, "reason": "name_required"}
        avail = self.check_availability(date, time, party_size)
        if not avail["available"]:
            return {"created": False, **avail}
        date, time = avail["date"], avail["time"]
        table = self._free_table(date, _hm(time), party_size)
        n = self.db.execute("SELECT COUNT(*) FROM reservations").fetchone()[0]
        code = f"JR-{1001 + n}"
        self.db.execute("INSERT INTO reservations (code, name, party_size, date, time, table_id, source) VALUES (?,?,?,?,?,?,?)",
                        (code, name, party_size, date, time, table, "agent"))
        self.db.commit()
        return {"created": True, "code": code, "name": name, "party_size": party_size, "date": date, "time": time,
                "spoken": avail["spoken"]}

    # -- handoff -------------------------------------------------------------
    def transfer_to_staff(self, reason: str) -> dict:
        reason = normalise_reason(reason)
        if reason is None:
            return {"transferred": False, "error": "invalid_reason", "allowed": TRANSFER_REASONS,
                    "next_step": "Do not transfer. Keep helping the caller yourself."}
        return {"transferred": True, "reason": reason}

    def agent_reservations(self) -> list[dict]:
        return [dict(r) for r in self.db.execute("SELECT code, name, party_size, date, time FROM reservations WHERE source = 'agent'")]


DAY = "The day exactly as the caller said it, e.g. 'tonight', 'tomorrow', 'Sunday', 'November 20'. Code works out the date."
TIME = "The time as the caller said it, e.g. '7:30 pm', 'seven thirty', '19:30'."

TRANSFER_REASONS = ["party_over_8_or_private_event", "complaint", "caller_asked_for_a_person",
                    "allergen_certainty_needed", "tool_failure", "caller_accepted_transfer_offer"]

REASON_WORDS = [("party_over_8_or_private_event", r"party|event|large|group|over 8|people|birthday|wedding"),
                ("complaint", r"complain|unhappy|awful|bad experience"),
                ("allergen_certainty_needed", r"allerg|guarantee|certain|verif"),
                ("tool_failure", r"fail|error|unavailable|down"),
                ("caller_asked_for_a_person", r"person|manager|human|staff|speak|talk"),
                ("caller_accepted_transfer_offer", r"accept|offer|agreed|request")]


def normalise_reason(reason: str) -> str | None:
    """Map the model's wording onto the allowed reasons; anything else is refused."""
    r = str(reason).lower().strip()
    if r in TRANSFER_REASONS:
        return r
    return next((k for k, pat in REASON_WORDS if re.search(pat, r)), None)


# JSON schemas the LLM sees. Kept by hand so the descriptions are exactly what we want the model to read.
TOOL_SPECS = [
    {"name": "get_hours", "description": "Opening hours. Pass a day for one day, or nothing for the whole week.",
     "parameters": {"type": "object", "properties": {"date": {"type": "string", "description": DAY}}, "required": []}},
    {"name": "search_menu", "description": "Search dishes by keyword, category (starter, main, side, dessert) or diet "
     "(vegetarian, vegan). Returns names, prices and descriptions. Has NO allergen information.",
     "parameters": {"type": "object", "properties": {
         "query": {"type": "string"}, "category": {"type": "string"}, "dietary": {"type": "string"}}, "required": []}},
    {"name": "check_allergen", "description": "The ONLY source for allergen answers. Checks one dish against one allergen "
     "(milk, egg, fish, shellfish, tree nuts, peanuts, nuts, gluten, soy, sesame). Returns a `say` sentence: repeat it "
     "word for word. Call once per dish and allergen.",
     "parameters": {"type": "object", "properties": {
         "dish": {"type": "string"}, "allergen": {"type": "string"}}, "required": ["dish", "allergen"]}},
    {"name": "check_availability", "description": "Check if a table is free. Suggests alternative times when full.",
     "parameters": {"type": "object", "properties": {
         "date": {"type": "string", "description": DAY},
         "time": {"type": "string", "description": TIME},
         "party_size": {"type": "integer"}}, "required": ["date", "time", "party_size"]}},
    {"name": "create_reservation", "description": "Book a table. Only call after you have read the details back and "
     "the caller has said yes.",
     "parameters": {"type": "object", "properties": {
         "name": {"type": "string"}, "party_size": {"type": "integer"},
         "date": {"type": "string", "description": DAY},
         "time": {"type": "string", "description": TIME}}, "required": ["name", "party_size", "date", "time"]}},
    {"name": "transfer_to_staff", "description": "Hand the call to a person and end your part of the call. Only for the "
     "reasons listed. Parties of 8 or fewer can be booked by you.",
     "parameters": {"type": "object", "properties": {"reason": {"type": "string", "enum": TRANSFER_REASONS}},
                    "required": ["reason"]}},
]
