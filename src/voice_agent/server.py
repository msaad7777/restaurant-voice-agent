"""FastAPI app: one push-to-talk page and a turn endpoint.

Runs on 127.0.0.1 only. No auth: this is a local demo, not a phone line.
"""
import base64
import datetime as dt
import os
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from . import data, speech
from .agent import Agent, make_llm
from .tools import Tools

MODEL = os.environ.get("AGENT_MODEL", "gpt-oss:20b")
TODAY = dt.date.fromisoformat(os.environ["AGENT_TODAY"]) if os.environ.get("AGENT_TODAY") else None

app = FastAPI(title="Juniper and Rye voice agent")
_db = data.build()
_db_lock = threading.Lock()
_calls: dict[str, Agent] = {}
_llm = None


def llm():
    global _llm
    if _llm is None:
        _llm = make_llm(MODEL)
    return _llm


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/health")
def health():
    try:
        speech.synthesize("ok")
        piper = True
    except OSError:
        piper = False
    return {"ok": True, "model": MODEL, "stt": speech.STT_MODEL, "piper": piper}


@app.post("/api/call")
def start_call():
    call_id = uuid.uuid4().hex[:12]
    _calls[call_id] = Agent(Tools(_db, TODAY or dt.date.today()), llm())
    greeting = f"Thanks for calling {data.RESTAURANT}. How can I help?"
    return {"call_id": call_id, "reply": greeting, "audio": _tts(greeting)[0]}


def _tts(text: str) -> tuple[str | None, int]:
    try:
        wav, ms = speech.synthesize(text)
        return base64.b64encode(wav).decode(), ms
    except OSError:
        return None, 0  # text still works without Piper


@app.post("/api/turn")
def turn(call_id: str = Form(...), audio: UploadFile | None = File(None), text: str | None = Form(None)):
    agent = _calls.get(call_id)
    if agent is None:
        raise HTTPException(404, "unknown call")
    seq = agent.begin_turn()  # anything still running for this call is now superseded (barge-in)
    stt_ms = 0
    if audio is not None:
        text, stt_ms = speech.transcribe(audio.file.read())
    if not text:
        return {"transcript": "", "reply": "Sorry, I didn't catch that.", "audio": _tts("Sorry, I didn't catch that.")[0]}
    with _db_lock:
        r = agent.respond(text, seq=seq)
    b64, tts_ms = _tts(r.text)
    return {"transcript": text, "reply": r.text, "audio": b64, "committed": r.committed, "handoff": r.handoff,
            "tool_calls": [{"name": c["name"], "args": c["args"], "result": c["result"]} for c in r.tool_calls],
            "guards": r.guards, "timings_ms": {"stt": stt_ms, "agent": r.total_ms, "llm_calls": r.llm_ms, "tts": tts_ms}}
