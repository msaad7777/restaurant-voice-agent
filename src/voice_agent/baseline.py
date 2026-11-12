"""Rule-based agent: keywords, regex slot filling and the same tools. No LLM.

Used to answer "does every part of this need an LLM?" on the same scenarios.
"""
import datetime as dt
import json
import re
import time

from .agent import AFFIRM, NEGATE, TurnResult
from .parsing import parse_date, parse_party, parse_time
from .tools import ALLERGEN_WORDS, Tools, spoken_date, spoken_time

HANDOFF = re.compile(r"\b(manager|a person|a human|someone|real person|staff member|complain|complaint|private event|"
                     r"party room|buy ?out|catering|wedding)\b", re.I)


NOT_NAMES = {"the", "a", "my", "four", "two", "yes", "yeah", "no", "sure", "okay", "ok", "sorry", "thanks", "please"}


def parse_name(text: str) -> str | None:
    m = re.search(r"\b(?i:name is|name's|name|under|it's for|this is|my name is)\s+([A-Z][a-zA-Z'\-]+)", text)
    if m and m.group(1).lower() not in NOT_NAMES:
        return m.group(1)
    m = re.fullmatch(r"\s*(?:it's\s+)?([A-Z][a-zA-Z'\-]+)\.?\s*", text)
    return m.group(1) if m and m.group(1).lower() not in NOT_NAMES else None


class RuleAgent:
    def __init__(self, tools: Tools):
        self.tools = tools
        self.slots: dict = {}
        self.pending_confirm = False
        self.booking = False
        self.ended = False

    def _dishes(self, t: str) -> list[str]:
        found = []
        for r in self.tools._rows():
            if any(re.search(rf"\b{re.escape(c)}\b", t) for c in [r["name"].lower()] + json.loads(r["aliases"])):
                found.append(r["name"])
        return found

    def _allergen_parts(self, text: str) -> tuple[list[str], str | None]:
        t = text.lower()
        allergen = next((w for w in sorted(ALLERGEN_WORDS, key=len, reverse=True)
                         if w not in ("butter", "cream", "cheese") and re.search(rf"\b{re.escape(w)}\b", t)), None)
        if not allergen or not re.search(r"allerg|contain|have|any|free|safe|in it|in the|okay|ok\b", t):
            return [], None
        return self._dishes(t) or ["?"], allergen

    def respond(self, text: str, seq=None) -> TurnResult:
        t0 = time.perf_counter()
        res = TurnResult(text="")
        if self.ended:
            res.text = "This call has been transferred to a member of staff."
            return res
        out = []

        def call(tool, **kw):
            try:
                r = getattr(self.tools, tool)(**kw)
            except Exception as e:  # noqa: BLE001 - mirror the LLM agent's error handling
                r = {"error": "tool_unavailable", "detail": str(e)}
            res.tool_calls.append({"name": tool, "args": kw, "result": r, "ms": 0})
            return r

        party = parse_party(text)
        if HANDOFF.search(text) or (party and party > 8):
            call("transfer_to_staff", reason="party_over_8_or_private_event" if (party and party > 8) else "caller_asked_for_a_person")
            res.handoff, self.ended = True, True
            res.text = "I'll transfer you to a member of staff who can help with that."
            res.total_ms = round((time.perf_counter() - t0) * 1000)
            return res

        dishes, allergen = self._allergen_parts(text)
        parts = bool(dishes)
        for dish in dishes:
            if dish == "?":
                out.append("Which dish would you like me to check?")
                continue
            r = call("check_allergen", dish=dish, allergen=allergen)
            out.append(r.get("say") or "I can't check our allergen list right now. I can transfer you to a member of staff.")
            if r.get("status") == "unverified" and re.search(r"\b(guarantee|severe|certain|100|completely)\b", text, re.I):
                call("transfer_to_staff", reason="allergen_certainty_needed")
                res.handoff, self.ended = True, True
                out.append("I'm transferring you to a member of staff now.")
                res.text = " ".join(out)
                return res

        if not parts and re.search(r"\b(how much|price|cost)\b", text, re.I):
            names = self._dishes(text.lower())
            r = call("search_menu", query=names[0] if names else text)
            ds = [d for d in r.get("dishes", []) if not names or d["name"] in names]
            out.append(" ".join(f"The {d['name']} is ${d['price']:.0f}." for d in ds[:3]) if ds else "I can't find that on the menu.")

        if re.search(r"\b(open|close|closing|hours)\b", text, re.I):
            d = parse_date(text, self.tools.today)
            r = call("get_hours", date=d) if d else call("get_hours")
            if "error" in r:
                out.append("I can't check our hours right now.")
            elif d:
                out.append(f"We're open {spoken_time(r['opens'])} to {spoken_time(r['closes'])}." if r["open"] else f"We're closed on {r['weekday']}s.")
            else:
                out.append("We're closed Mondays, open 5 to 10 pm Tuesday to Thursday, 5 to 11 pm Friday and Saturday, and 4 to 9 pm Sunday.")

        m = re.search(r"\b(vegan|vegetarian)\b", text, re.I)
        if m and not parts:
            r = call("search_menu", dietary=m.group(1).lower())
            names = [d["name"] for d in r.get("dishes", [])]
            out.append(f"Our {m.group(1).lower()} options are {', '.join(names[:-1])} and {names[-1]}." if names else "I can't check the menu right now.")

        booking = re.search(r"\b(book|reserve|reservation|table)\b", text, re.I) or self.slots or self.booking
        if booking:
            self.booking = True
            if self.pending_confirm and AFFIRM.search(text) and not NEGATE.search(text):
                s = self.slots
                r = call("create_reservation", name=s["name"], party_size=s["party"], date=s["date"], time=s["time"])
                if r.get("created"):
                    out.append(f"You're booked: {s['party']} people {spoken_date(dt.date.fromisoformat(s['date']), self.tools.today)} "
                               f"at {spoken_time(s['time'])} under {s['name']}. Your confirmation code is {r['code']}.")
                    self.slots, self.pending_confirm = {}, False
                else:
                    out.append("Sorry, I wasn't able to complete the booking. I can transfer you to a member of staff.")
            else:
                for k, v in (("party", party), ("time", parse_time(text)), ("date", parse_date(text, self.tools.today)),
                             ("name", parse_name(text))):
                    if v:
                        self.slots[k] = v
                missing = [k for k in ("party", "date", "time") if k not in self.slots]
                if missing:
                    out.append({"party": "How many people is the booking for?", "date": "What day would you like to come in?",
                                "time": "What time would you like?"}[missing[0]])
                else:
                    s = self.slots
                    r = call("check_availability", date=s["date"], time=s["time"], party_size=s["party"])
                    if "error" in r:
                        out.append("I can't check availability right now. I can transfer you to a member of staff.")
                    elif not r["available"]:
                        alts = r.get("alternatives") or []
                        out.append(f"Sorry, {spoken_time(s['time'])} isn't available"
                                   + (f". I can do {', '.join(alts)}." if alts else
                                      f". Our last booking that day is {spoken_time(r['last_booking'])}." if r.get("last_booking") else
                                      f" ({r['reason'].replace('_', ' ')})."))
                        self.slots.pop("time")
                    elif "name" not in s:
                        out.append(f"{spoken_time(s['time'])} is available. What name should I put the booking under?")
                    else:
                        out.append(f"I have {s['party']} people {spoken_date(dt.date.fromisoformat(s['date']), self.tools.today)} "
                                   f"at {spoken_time(s['time'])} under {s['name']}. Shall I book that?")
                        self.pending_confirm = True
        res.text = " ".join(out) or "Sorry, I didn't catch that. Could you say it again?"
        res.total_ms = round((time.perf_counter() - t0) * 1000)
        return res
