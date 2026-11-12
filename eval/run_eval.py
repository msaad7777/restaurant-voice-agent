"""Scripted-call eval for the voice agent.

  uv run python eval/run_eval.py text  --agent gpt-oss:20b            # LLM agent, guards on
  uv run python eval/run_eval.py text  --agent gpt-oss:20b --no-guards
  uv run python eval/run_eval.py text  --agent rules                  # no-LLM baseline
  uv run python eval/run_eval.py audio --agent gpt-oss:20b --stt base.en   # Piper caller voices -> Whisper -> agent -> Piper

Each scenario runs on a fresh database with "today" fixed to Friday 2026-11-13.
Writes eval/results/<run>.json.
"""
import argparse
import datetime as dt
import hashlib
import json
import re
import statistics
import time
from pathlib import Path

from voice_agent import data
from voice_agent.agent import Agent, make_llm
from voice_agent.baseline import RuleAgent
from voice_agent.tools import Tools

E = Path(__file__).parent
R = E / "results"
AUDIO = E / "audio"
TODAY = dt.date(2026, 11, 13)
SIDE_EFFECTS = {"create_reservation", "transfer_to_staff"}


def load(name="scenarios"):
    return [json.loads(line) for line in open(E / f"{name}.jsonl") if line.strip()]


def turn_text(turn) -> tuple[str, float | None]:
    if isinstance(turn, str):
        return turn, None
    words = turn["text"].split()
    return " ".join(words[:int(len(words) * turn["cut"])]), turn["cut"]


# ---- audio ---------------------------------------------------------------
def caller_audio(sid: str, i: int, turn) -> bytes:
    """Synthesise the caller's words with a LibriTTS-R speaker picked from the scenario id."""
    from voice_agent import speech
    AUDIO.mkdir(exist_ok=True)
    full = turn if isinstance(turn, str) else turn["text"]
    speaker = int(hashlib.sha1(sid.encode()).hexdigest(), 16) % 904
    path = AUDIO / f"{sid}_{i}.wav"
    if not path.exists():
        wav, _ = speech.synthesize(full, voice="en_US-libritts_r-medium", speaker_id=speaker)
        path.write_bytes(wav)
    wav = path.read_bytes()
    if not isinstance(turn, str):
        wav = speech.truncate_wav(wav, turn["cut"])
    return wav


def wer(ref: str, hyp: str) -> tuple[int, int]:
    norm = lambda s: re.sub(r"[^a-z0-9' ]", " ", s.lower().replace(":", " ")).split()
    r, h = norm(ref), norm(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
    return d[len(h)], len(r)


# ---- scoring -------------------------------------------------------------
def resolved(call: dict, tools: Tools) -> dict:
    """What a call actually meant, after the tools resolved spoken dates and times."""
    a, res = dict(call["args"]), call["result"]
    if call["name"] in ("check_availability", "create_reservation"):
        if res.get("date"):
            a.update(date=res["date"], time=res.get("time"), party_size=res.get("party_size", a.get("party_size")))
        else:
            d, t = tools.resolve(a.get("date", ""), a.get("time", ""))
            a.update(date=d, time=t)
    elif call["name"] == "get_hours":
        a["date"] = res.get("date", a.get("date"))
    elif call["name"] == "check_allergen":
        a["dish"] = res.get("dish", a.get("dish"))
        a["allergen"] = res.get("allergen", a.get("allergen"))
    elif call["name"] == "search_menu":
        a = {k: (v.lower() if isinstance(v, str) else v) for k, v in a.items()}
    return a


def match(expected, calls, tools) -> bool:
    names, want = expected[0].split("|"), expected[1]
    for c in calls:
        if c["name"] in names:
            got = resolved(c, tools)
            if all(str(got.get(k, "")).lower() == str(v).lower() for k, v in want.items()):
                return True
    return False


def score(sc, turns, tools):
    calls = [c for t in turns for c in t["tool_calls"]]
    texts = " ".join(t["reply"] for t in turns)
    made = tools.agent_reservations()
    exp = sc["reservation"]
    res_ok = (not made) if exp is None else (len(made) == 1 and all(
        str(made[0][k]).lower() == str(v).lower() for k, v in exp.items()))
    # Names are scored separately: a misheard name with the right party, day and time is a different
    # failure from booking the wrong table.
    details_ok = (not made) if exp is None else (len(made) == 1 and all(
        str(made[0][k]).lower() == str(v).lower() for k, v in exp.items() if k != "name"))
    handoff = any(t["handoff"] for t in turns)
    handoff_ok = sc["handoff"] is None or handoff == sc["handoff"]
    say_ok = all(s.lower() in texts.lower() for s in sc.get("must_say", []))
    bad = [p for p in sc.get("must_not_say", []) if re.search(p, texts, re.I)]
    cut_side_effect = any(t["cut"] and any(c["name"] in SIDE_EFFECTS for c in t["tool_calls"]) for t in turns)
    tool_hits = [match(e, calls, tools) for e in sc["tools"]]
    done = [c for c in calls if c["result"].get("created") or c["result"].get("transferred")]  # attempts that failed don't count
    unexpected_side = [c["name"] for c in done if not (
        (c["name"] == "create_reservation" and exp) or (c["name"] == "transfer_to_staff" and sc["handoff"] is not False))]
    booked_unconfirmed = sum(1 for t in turns for c in t["tool_calls"]
                             if c["name"] == "create_reservation" and c["result"].get("created") and t.get("unconfirmed"))
    return {"task_complete": res_ok and handoff_ok and say_ok and not bad and not cut_side_effect,
            "task_complete_except_name": details_ok and handoff_ok and say_ok and not bad and not cut_side_effect,
            "reservation_ok": res_ok, "handoff": handoff, "handoff_ok": handoff_ok, "must_say_ok": say_ok,
            "must_not_say_hits": bad, "cut_turn_side_effect": cut_side_effect,
            "expected_tools": len(tool_hits), "expected_tools_hit": sum(tool_hits),
            "unexpected_side_effects": unexpected_side, "booked_without_confirmation": booked_unconfirmed}


# ---- run -----------------------------------------------------------------
def run(args):
    scenarios = load(args.set)
    if args.only:
        scenarios = [s for s in scenarios if s["id"] in args.only.split(",")]
    llm = None if args.agent == "rules" else make_llm(args.agent)
    if llm is not None:
        llm.invoke("hi")  # load the model before timing anything
    rows = []
    for sc in scenarios:
        tools = Tools(data.build(), TODAY, set(sc.get("faults", [])))
        agent = RuleAgent(tools) if llm is None else Agent(tools, llm, guards=not args.no_guards)
        turns = []
        for i, turn in enumerate(sc["turns"]):
            said, cut = turn_text(turn)
            stt_ms = tts_ms = 0
            heard = said
            if args.mode == "audio":
                from voice_agent import speech
                heard, stt_ms = speech.transcribe(caller_audio(sc["id"], i, turn), model=args.stt)
            r = agent.respond(heard)
            if args.mode == "audio":
                from voice_agent import speech
                _, tts_ms = speech.synthesize(r.text)
            from voice_agent.agent import AFFIRM, NEGATE
            unconfirmed = not (AFFIRM.search(heard) and not NEGATE.search(heard))
            turns.append({"said": said, "heard": heard, "cut": cut, "reply": r.text, "raw_reply": r.raw_text,
                          "tool_calls": r.tool_calls, "guards": r.guards, "handoff": r.handoff,
                          "unconfirmed": unconfirmed, "ms": {"stt": stt_ms, "agent": r.total_ms, "llm": r.llm_ms,
                                                             "tts": tts_ms}})
            if getattr(agent, "ended", False):
                break
        s = score(sc, turns, tools)
        rows.append({"id": sc["id"], "category": sc["category"], "expected_handoff": sc["handoff"], **s, "turns": turns})
        print(f"{sc['id']:4} {'PASS' if s['task_complete'] else 'FAIL'}  tools {s['expected_tools_hit']}/{s['expected_tools']}"
              f"  {[t['reply'][:70] for t in turns][-1]}")
    return rows


def summarise(rows, args):
    turns = [t for r in rows for t in r["turns"]]
    agent_ms = sorted(t["ms"]["agent"] for t in turns)
    p = lambda xs, q: xs[min(len(xs) - 1, int(len(xs) * q))] if xs else None
    scored = [r for r in rows if r["expected_handoff"] is not None]
    tp = sum(r["handoff"] and r["expected_handoff"] for r in scored)
    fp = sum(r["handoff"] and not r["expected_handoff"] for r in scored)
    fn = sum((not r["handoff"]) and r["expected_handoff"] for r in scored)
    out = {
        "run": args.name, "set": args.set, "mode": args.mode, "agent": args.agent, "guards": not args.no_guards, "stt": args.stt if args.mode == "audio" else None,
        "scenarios": len(rows), "turns": len(turns),
        "task_completion": round(sum(r["task_complete"] for r in rows) / len(rows), 3),
        "task_completion_except_name": round(sum(r["task_complete_except_name"] for r in rows) / len(rows), 3),
        "tool_call_recall": round(sum(r["expected_tools_hit"] for r in rows) / max(1, sum(r["expected_tools"] for r in rows)), 3),
        "unexpected_side_effects": sum(len(r["unexpected_side_effects"]) for r in rows),
        "booked_without_confirmation": sum(r["booked_without_confirmation"] for r in rows),
        "must_not_say_violations": sum(len(r["must_not_say_hits"]) for r in rows),
        "handoff": {"tp": tp, "fp": fp, "fn": fn,
                    "precision": round(tp / (tp + fp), 3) if tp + fp else None,
                    "recall": round(tp / (tp + fn), 3) if tp + fn else None},
        "agent_ms": {"p50": p(agent_ms, 0.5), "p95": p(agent_ms, 0.95), "max": agent_ms[-1]},
        "guards_fired": {},
        "by_category": {},
        "failed": [r["id"] for r in rows if not r["task_complete"]],
    }
    for t in turns:
        for g in t["guards"]:
            k = g.split(":")[0]
            out["guards_fired"][k] = out["guards_fired"].get(k, 0) + 1
    for r in rows:
        c = out["by_category"].setdefault(r["category"], [0, 0])
        c[0] += r["task_complete"]
        c[1] += 1
    out["by_category"] = {k: f"{a}/{b}" for k, (a, b) in sorted(out["by_category"].items())}
    if args.mode == "audio":
        e = n = 0
        for t in turns:
            if not t["cut"]:
                de, dn = wer(t["said"], t["heard"])
                e, n = e + de, n + dn
        stt = sorted(t["ms"]["stt"] for t in turns)
        tts = sorted(t["ms"]["tts"] for t in turns)
        first_audio = sorted(t["ms"]["stt"] + t["ms"]["agent"] + t["ms"]["tts"] for t in turns)
        out.update(wer=round(e / n, 3), stt_ms={"p50": p(stt, .5), "p95": p(stt, .95)},
                   tts_ms={"p50": p(tts, .5), "p95": p(tts, .95)},
                   end_of_speech_to_reply_audio_ms={"p50": p(first_audio, .5), "p95": p(first_audio, .95)})
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["text", "audio"])
    ap.add_argument("--agent", default="gpt-oss:20b")
    ap.add_argument("--no-guards", action="store_true")
    ap.add_argument("--stt", default="base.en")
    ap.add_argument("--only")
    ap.add_argument("--set", default="scenarios", choices=["scenarios", "heldout"])
    ap.add_argument("--name")
    args = ap.parse_args()
    args.name = args.name or "_".join(x for x in [args.mode, args.agent.replace(":", "-"),
                                                   "noguards" if args.no_guards else "", "heldout" if args.set == "heldout" else "",
                                                   args.stt if args.mode == "audio" else ""] if x)
    t0 = time.time()
    rows = run(args)
    summary = summarise(rows, args)
    summary["wall_s"] = round(time.time() - t0)
    R.mkdir(exist_ok=True)
    json.dump({"summary": summary, "rows": rows}, open(R / f"{args.name}.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))
