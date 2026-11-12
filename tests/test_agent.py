"""Fast tests: tools, guards and the rule baseline. No models, no audio."""
import datetime as dt

from langchain_core.messages import AIMessage

from voice_agent import data
from voice_agent.agent import Agent
from voice_agent.baseline import RuleAgent
from voice_agent.parsing import parse_date, parse_party, parse_time
from voice_agent.tools import Tools

TODAY = dt.date(2026, 11, 13)  # a Friday


class ScriptedLLM:
    """Returns the given AIMessages in order, like a model would."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.seen = []

    def invoke(self, msgs):
        self.seen.append(list(msgs))
        return self.replies.pop(0)


def call(tool, i=0, **args):
    return {"name": tool, "args": args, "id": f"c{i}", "type": "tool_call"}


def tools(faults=None):
    return Tools(data.build(), TODAY, faults)


def test_full_slot_offers_alternatives():
    r = tools().check_availability("2026-11-13", "19:30", 4)
    assert not r["available"] and r["reason"] == "fully_booked"
    assert r["alternatives"] and "19:30" not in r["alternatives"]
    assert tools().check_availability("2026-11-13", "19:30", 2)["available"]


def test_availability_rules():
    t = tools()
    assert t.check_availability("2026-11-16", "19:00", 2)["reason"] == "closed"           # Monday
    assert t.check_availability("2026-11-13", "22:00", 2)["reason"] == "outside_booking_hours"
    assert t.check_availability("2026-11-13", "19:30", 12)["reason"] == "large_party"
    assert t.check_availability("2026-11-12", "19:30", 2)["reason"] == "date_in_past"


def test_allergen_answers_come_from_verified_fields():
    t = tools()
    assert t.check_allergen("pesto", "nuts")["status"] == "contains"
    assert t.check_allergen("risotto", "nuts")["status"] == "not_listed"
    assert t.check_allergen("fries", "shellfish")["status"] == "may_contain"
    assert t.check_allergen("special", "gluten")["status"] == "unverified"
    assert t.check_allergen("lobster thermidor", "milk")["status"] == "unknown_dish"
    assert t.check_allergen("roast chicken", "fish")["status"] == "contains"  # anchovy in the salsa verde


def test_booking_needs_confirmation():
    llm = ScriptedLLM(AIMessage("", tool_calls=[call("create_reservation", name="Priya", party_size=2,
                                                     date="2026-11-13", time="19:30")]),
                      AIMessage("Shall I book that?"))
    a = Agent(tools(), llm)
    r = a.respond("Two people at 7:30 tonight, name Priya.")
    assert "confirmation_gate" in r.guards
    assert r.tool_calls[0]["result"]["reason"] == "needs_confirmation"
    assert a.tools.agent_reservations() == []


def test_confirmed_booking_is_created():
    llm = ScriptedLLM(AIMessage("", tool_calls=[call("create_reservation", name="Priya", party_size=2,
                                                     date="2026-11-13", time="19:30")]),
                      AIMessage("You're booked, code JR-1009."))
    a = Agent(tools(), llm)
    r = a.respond("Yes, please book it.")
    assert r.guards == [] and a.tools.agent_reservations()[0]["name"] == "Priya"


def test_llm_allergen_claim_is_replaced_by_verified_sentence():
    llm = ScriptedLLM(AIMessage("", tool_calls=[call("check_allergen", dish="special", allergen="nuts")]),
                      AIMessage("The special is nut-free, so you'll be fine. Anything else?"))
    r = Agent(tools(), llm).respond("Is the special safe for a nut allergy?")
    assert "nut-free" not in r.text and "hasn't been verified" in r.text and "Anything else?" in r.text
    assert any(g.startswith("allergen_sentences_dropped") for g in r.guards)


def test_allergen_question_without_tool_call_is_nudged_then_blocked():
    llm = ScriptedLLM(AIMessage("The risotto has no nuts."), AIMessage("The risotto has no nuts at all."))
    r = Agent(tools(), llm).respond("Does the risotto have nuts?")
    assert "allergen_nudge" in r.guards
    assert "no nuts" not in r.text and "verified list" in r.text


def test_false_success_is_blocked_when_tool_fails():
    llm = ScriptedLLM(AIMessage("", tool_calls=[call("create_reservation", name="Priya", party_size=2,
                                                     date="2026-11-13", time="19:30")]),
                      AIMessage("Great, you're all set for 7:30!"))
    r = Agent(tools({"create_reservation"}), llm).respond("Yes, book it.")
    assert r.tool_calls[0]["result"]["error"] == "tool_unavailable"
    assert "all set" not in r.text and "wasn't able" in r.text and "false_success_blocked" in r.guards


def test_handoff_ends_call():
    llm = ScriptedLLM(AIMessage("", tool_calls=[call("transfer_to_staff", reason="party_over_8_or_private_event")]), AIMessage("One moment."))
    a = Agent(tools(), llm)
    r = a.respond("Can I book for 14 people?")
    assert r.handoff and a.ended and "transferring" in r.text


def test_superseded_turn_is_not_committed():
    a = Agent(tools(), ScriptedLLM(AIMessage("What time?")))
    seq = a.begin_turn()
    a.begin_turn()  # the caller barged in with a new turn
    r = a.respond("Can I book a table for", seq=seq)
    assert not r.committed and len(a.history) == 1


def test_rule_parsers():
    assert parse_time("seven thirty tonight") == "19:30"
    assert parse_time("at 9 pm") == "21:00"
    assert parse_party("a table for four at 7:30") == 4
    assert parse_party("we're six people") == 6


def test_rule_agent_books_after_confirmation():
    a = RuleAgent(tools())
    assert "Shall I book" in a.respond("Book a table for two at 7:30 tonight under Priya").text
    assert "JR-" in a.respond("Yes please").text


def test_spoken_dates_resolve_in_code():
    assert parse_date("Sunday", TODAY) == "2026-11-15"
    assert parse_date("tonight", TODAY) == "2026-11-13"
    assert parse_date("next Friday", TODAY) == "2026-11-20"
    assert parse_date("November 20th", TODAY) == "2026-11-20"
    assert parse_time("7:30 pm") == "19:30" and parse_time("19:30") == "19:30" and parse_time("half past seven") == "19:30"
    r = tools().get_hours("Sunday")
    assert r["weekday"] == "Sunday" and r["closes"] == "21:00"
    assert tools().check_availability("tonight", "7:30 pm", 2)["date"] == "2026-11-13"


def test_transfer_needs_a_listed_reason():
    assert tools().transfer_to_staff("caller gave a name")["transferred"] is False


def test_bare_hour_means_evening_and_odd_filters_are_ignored():
    assert tools().check_availability("Monday", "7", 2)["reason"] == "closed"
    assert tools().check_availability("tomorrow", "8", 2)["time"] == "20:00"
    assert tools().search_menu(query="burger", category="main", dietary="none")["count"] == 1


def test_reason_wording_is_mapped_and_unrelated_reasons_refused():
    assert tools().transfer_to_staff("caller asking for a person")["reason"] == "caller_asked_for_a_person"
    assert tools().transfer_to_staff("caller wants to book under a name")["transferred"] is False


def test_claimed_transfer_without_tool_call_is_nudged_then_removed():
    llm = ScriptedLLM(AIMessage("I'll transfer you to a staff member."), AIMessage("I'll transfer you now."))
    a = Agent(tools(), llm)
    r = a.respond("Can I speak to the manager?")
    assert "transfer_nudge" in r.guards and "false_transfer_blocked" in r.guards
    assert not a.ended and "if you'd like" in r.text


def test_allergen_must_have_been_said_by_the_caller():
    # Whisper dropped "fish"; the model guessed "nuts". The guard refuses to answer for an allergen nobody said.
    llm = ScriptedLLM(AIMessage("", tool_calls=[call("check_allergen", dish="roast chicken", allergen="nuts")]),
                      AIMessage("The chicken does not contain nuts."))
    r = Agent(tools(), llm).respond("Does the roast chicken efficient it? My son is allergic.")
    assert "allergen_not_heard" in r.guards
    assert "Which allergen" in r.text and "does not contain" not in r.text
