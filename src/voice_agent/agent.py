"""LLM agent with tool calling, plus deterministic guards around it.

The LLM understands the caller and picks tools. Code decides what is allowed:
  - booking only after the caller confirms (confirmation gate),
  - allergen statements only from check_allergen's verified `say` sentences (allergen guard),
  - no "you're booked" unless a reservation was actually created (false-success guard),
  - tool failures surface as errors, never as invented answers.
"""
import datetime as dt
import difflib
import json
import re
import threading
import time
from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from . import data
from .tools import ALLERGEN_WORDS, TOOL_SPECS, ToolError, Tools

MAX_TOOL_ROUNDS = 5

SYSTEM = """You are the phone host for {restaurant}, a restaurant. You are speaking on a voice call.
Today is {weekday}, {today}. Pass days and times to tools as the caller said them ("Sunday", "7:30 pm"); the tools work out the dates. Use the `spoken` field when you repeat a date.

Rules:
- Keep replies short: one to three spoken sentences. No lists, no markdown, no emoji. Say times like "7:30 pm".
- Use tools for facts. Never guess hours, availability, prices or dishes.
- Allergens: only use check_allergen, and repeat its `say` sentence word for word. Never add your own allergen advice.
  If the caller asks about "nuts", check "nuts". If they name an allergy without a dish, ask which dish.
- Bookings: you need name, party size, date and time. First call check_availability. If it is free, read the details
  back and ask "Shall I book that?" Only call create_reservation after the caller says yes. Never say a table is booked
  unless create_reservation returned created: true, and then give the confirmation code.
- If a time is full, offer the alternatives the tool gives you.
- If the caller's words are cut off or unclear, ask them to repeat. Do not act on half a request.
- If a tool returns an error, say you can't check that right now and offer to transfer them to staff. Do not guess.
- Use transfer_to_staff only for: a party over 8 (call it as soon as you hear the size) or a private event, a complaint,
  the caller asking for a person, the caller needing a guarantee about an allergen you cannot verify, or the caller
  accepting your offer of a transfer. Never transfer for anything else; you can book parties of 1 to 8 yourself.
- Answer every question the caller asked in their turn."""

AFFIRM = re.compile(r"\b(yes|yeah|yep|yup|sure|correct|right|go ahead|book it|please do|sounds good|perfect|confirm(ed)?|that works|ok|okay)\b", re.I)
NEGATE = re.compile(r"\b(no|not|don't|dont|wait|actually|change|instead|hold on)\b", re.I)
ALLERGEN_INTENT = re.compile(
    r"allerg|contain|intoleran|coeliac|celiac|anaphyla|\b(nut|gluten|dairy|lactose|shellfish|peanut|sesame|soy|egg)[- ]?free\b"
    r"|\bsafe (for|to)\b|\b(any|have|has|with) (tree )?(nuts?|peanuts?|gluten|dairy|milk|eggs?|fish|shellfish|soy|sesame|wheat)\b", re.I)
ALLERGEN_TERMS = re.compile(r"allerg|contain|\bfree of\b|-free\b|\bsafe\b|\b(" + "|".join(
    sorted((re.escape(w) for w in ALLERGEN_WORDS if w not in ("butter", "cream", "cheese")), key=len, reverse=True)) + r")\b", re.I)
SUCCESS_CLAIM = re.compile(r"(you'?re all set|you are all set|(is|are|has been|have been) (now )?(booked|confirmed|reserved)"
                           r"|i'?ve (booked|reserved|made)|i have (booked|reserved|made)|confirmation (code|number)|\bJR-\d+)", re.I)
TRANSFER_CLAIM = re.compile(r"(i'?ll|i will|i'?m going to|let me) (now )?(transfer|connect|put you through)|"
                            r"(transferring|connecting) you|you'?re being transferred|i'?ve transferred", re.I)
TRANSFER_WORDS = re.compile(r"transfer|connect|put you through|member of (our )?staff", re.I)


PUNCT = str.maketrans({"\u2011": "-", "\u2010": "-", "\u2013": "-", "\u2014": ", ", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u202f": " "})


def sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


@dataclass
class TurnResult:
    text: str
    tool_calls: list[dict] = field(default_factory=list)
    guards: list[str] = field(default_factory=list)
    llm_ms: list[int] = field(default_factory=list)
    total_ms: int = 0
    handoff: bool = False
    raw_text: str = ""
    committed: bool = True

    def to_dict(self):
        return self.__dict__.copy()


def make_llm(model: str, base_url: str = "http://127.0.0.1:11434"):
    from langchain_ollama import ChatOllama
    kw = {"model": model, "base_url": base_url, "temperature": 0, "num_ctx": 8192, "keep_alive": "30m"}
    if model.startswith("gpt-oss"):
        kw["reasoning"] = "low"
    elif model.startswith("qwen3"):
        kw["reasoning"] = False
    return ChatOllama(**kw).bind_tools([{"type": "function", "function": s} for s in TOOL_SPECS])


class Agent:
    """One phone call. Not shared between callers."""

    def __init__(self, tools: Tools, llm, guards: bool = True):
        self.tools = tools
        self.llm = llm
        self.guards = guards
        today = tools.today
        self.history = [SystemMessage(SYSTEM.format(restaurant=data.RESTAURANT, weekday=today.strftime("%A"),
                                                    today=today.isoformat()))]
        self.ended = False
        self._seq = 0
        self._lock = threading.Lock()

    # Barge-in: a newer turn supersedes an older one still running.
    def begin_turn(self) -> int:
        with self._lock:
            self._seq += 1
            return self._seq

    def _allergen_heard(self, allergen: str, user_text: str) -> bool:
        """The caller must have said this allergen (or a synonym) in this turn or the two before it.
        Stops the model from guessing the allergen when speech-to-text dropped the word."""
        key = ALLERGEN_WORDS.get(str(allergen).lower().strip())
        if key is None:
            return True  # unknown allergen: the tool itself refuses to answer
        keys = {"nuts", "tree_nuts", "peanuts"} if key in ("nuts", "tree_nuts", "peanuts") else {key}
        recent = [m.content for m in self.history if isinstance(m, HumanMessage)][-2:] + [user_text]
        text = " ".join(recent).lower()
        return any(ALLERGEN_WORDS[w] in keys and re.search(rf"\b{re.escape(w)}", text) for w in ALLERGEN_WORDS)

    def _call_tool(self, name: str, args: dict, user_text: str, result: TurnResult) -> dict:
        t = time.perf_counter()
        try:
            if name == "create_reservation" and self.guards and not (AFFIRM.search(user_text) and not NEGATE.search(user_text)):
                out = {"created": False, "reason": "needs_confirmation",
                       "next_step": "Read the details back and ask the caller to confirm before booking."}
                result.guards.append("confirmation_gate")
            elif name == "check_allergen" and self.guards and not self._allergen_heard(args.get("allergen", ""), user_text):
                dish = self.tools.find_dish(str(args.get("dish", "")))
                out = {"status": "allergen_not_heard", "say": f"Which allergen would you like me to check the "
                       f"{dish['name'] if dish else 'dish'} for?"}
                result.guards.append("allergen_not_heard")
            elif name not in {s["name"] for s in TOOL_SPECS}:
                out = {"error": "unknown_tool"}
            else:
                out = getattr(self.tools, name)(**args)
        except ToolError as e:
            out = {"error": "tool_unavailable", "detail": str(e)}
        except (TypeError, ValueError, KeyError) as e:
            out = {"error": "bad_arguments", "detail": str(e)}
        result.tool_calls.append({"name": name, "args": args, "result": out, "ms": round((time.perf_counter() - t) * 1000)})
        return out

    def respond(self, user_text: str, seq: int | None = None) -> TurnResult:
        t0 = time.perf_counter()
        seq = seq if seq is not None else self.begin_turn()
        result = TurnResult(text="")
        if self.ended:
            result.text = "This call has been transferred to a member of staff."
            return result
        msgs = self.history + [HumanMessage(user_text)]
        new = [msgs[-1]]
        nudged = False
        for _ in range(MAX_TOOL_ROUNDS + 1):
            t = time.perf_counter()
            ai: AIMessage = self.llm.invoke(msgs)
            result.llm_ms.append(round((time.perf_counter() - t) * 1000))
            msgs.append(ai)
            new.append(ai)
            if not ai.tool_calls:
                wants_allergen = self.guards and ALLERGEN_INTENT.search(user_text)
                called = any(c["name"] == "check_allergen" for c in result.tool_calls)
                if wants_allergen and not called and not nudged:
                    nudged = True
                    result.guards.append("allergen_nudge")
                    nudge = SystemMessage("Guard: the caller asked about allergens. Call check_allergen for each dish "
                                          "they asked about, or ask which dish if none was named.")
                    msgs.append(nudge)
                    new.append(nudge)
                    continue
                transferred = any(c["name"] == "transfer_to_staff" and c["result"].get("transferred") for c in result.tool_calls)
                if self.guards and not transferred and TRANSFER_CLAIM.search((ai.content or "").translate(PUNCT)) and "transfer_nudge" not in result.guards:
                    result.guards.append("transfer_nudge")
                    nudge = SystemMessage("Guard: you told the caller you are transferring them but did not call "
                                          "transfer_to_staff. Call it now if the transfer is warranted; otherwise do not "
                                          "claim a transfer.")
                    msgs.append(nudge)
                    new.append(nudge)
                    continue
                break
            for call in ai.tool_calls:
                out = self._call_tool(call["name"], call["args"], user_text, result)
                tm = ToolMessage(json.dumps(out), tool_call_id=call["id"])
                msgs.append(tm)
                new.append(tm)
        result.raw_text = (msgs[-1].content if isinstance(msgs[-1], AIMessage) else "") or ""
        result.raw_text = result.raw_text.translate(PUNCT)
        result.text = self._guard_text(result.raw_text, user_text, result) if self.guards else result.raw_text
        result.handoff = any(c["name"] == "transfer_to_staff" and c["result"].get("transferred") for c in result.tool_calls)
        if result.handoff:
            self.ended = True
            if not TRANSFER_WORDS.search(result.text):
                result.text = (result.text + " I'm transferring you to a member of staff now.").strip()
        if not result.text.strip():
            result.text = "Sorry, could you say that again?"
            result.guards.append("empty_reply")
        # Commit to history unless a newer turn started meanwhile and nothing changed state.
        side_effect = any(c["name"] in ("create_reservation", "transfer_to_staff") for c in result.tool_calls)
        with self._lock:
            superseded = seq != self._seq
        if superseded and not side_effect:
            result.committed = False
        else:
            final = AIMessage(result.text)
            self.history += [m for m in new if not (isinstance(m, AIMessage) and not m.tool_calls)] + [final]
        result.total_ms = round((time.perf_counter() - t0) * 1000)
        return result

    def _guard_text(self, text: str, user_text: str, result: TurnResult) -> str:
        checks = [c for c in result.tool_calls if c["name"] == "check_allergen"]
        says = [c["result"]["say"] for c in checks if "say" in c["result"]]
        failed = any("error" in c["result"] for c in checks)
        out = text
        if says or ALLERGEN_INTENT.search(user_text) or failed:
            kept, placed, dropped = [], False, 0
            for s in sentences(text):
                if ALLERGEN_TERMS.search(s) and not any(difflib.SequenceMatcher(None, _norm(s), _norm(v)).ratio() > 0.9 for v in says):
                    dropped += 1
                    if not placed:
                        kept += says
                        placed = True
                    continue
                if any(difflib.SequenceMatcher(None, _norm(s), _norm(v)).ratio() > 0.9 for v in says):
                    if not placed:
                        kept += says
                        placed = True
                    continue
                kept.append(s)
            if says and not placed:
                kept = says + kept
            if dropped:
                result.guards.append(f"allergen_sentences_dropped:{dropped}")
            if not says and dropped:
                kept.append("I can't check our allergen list right now. I can connect you with a member of staff who can "
                            "check with the kitchen." if failed else
                            "I can only give allergen information from our verified list. Which dish would you like me to check?")
            out = " ".join(dict.fromkeys(kept))
        transferred = any(c["name"] == "transfer_to_staff" and c["result"].get("transferred") for c in result.tool_calls)
        if not transferred and TRANSFER_CLAIM.search(out):
            result.guards.append("false_transfer_blocked")
            kept = [s for s in sentences(out) if not TRANSFER_CLAIM.search(s)]
            out = " ".join(kept + ["I can transfer you to a member of staff if you'd like."])
        created = any(c["name"] == "create_reservation" and c["result"].get("created") for c in result.tool_calls)
        if not created and SUCCESS_CLAIM.search(out):
            result.guards.append("false_success_blocked")
            out = ("Sorry, I wasn't able to complete the booking. I can transfer you to a member of staff who can help."
                   if any(c["name"] == "create_reservation" for c in result.tool_calls) else
                   " ".join(s for s in sentences(out) if not SUCCESS_CLAIM.search(s)) or "Sorry, could you say that again?")
        return out


def new_call(model: str, db=None, today: dt.date | None = None, faults: set[str] | None = None, guards: bool = True) -> Agent:
    db = db or data.build()
    return Agent(Tools(db, today or dt.date.today(), faults), make_llm(model), guards=guards)
