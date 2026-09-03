"""Deterministic scripted reasoners for the example skills (``--model mock``) and tests.

A script sees exactly what a real model would see: the ``ModelContext``. It must therefore
parse the state and observation *from the rendered context* — which doubles as a test that
the context contains everything needed to act.
"""

from __future__ import annotations

import json
import re
from typing import Any

from skillstate.context.builder import ModelContext
from skillstate.models.decision import AgentDecision


def parse_state(context: ModelContext) -> dict[str, Any]:
    section = context.sections["execution_state"]
    body = section.split("\n", 1)[1].rsplit("\n", 1)[0]
    return json.loads(body)


def observation_text(context: ModelContext) -> str:
    section = context.sections["latest_observation"]
    return section.split("\n", 1)[1].rsplit("\n", 1)[0]


def observation_attr(context: ModelContext, name: str) -> str:
    m = re.search(rf'{name}="([^"]*)"', context.sections["latest_observation"].split("\n", 1)[0])
    return m.group(1) if m else ""


def counter_script(context: ModelContext) -> AgentDecision:
    state = parse_state(context)
    env = state["environment"]["properties"]
    target = env.get("target")
    ops: list[dict[str, Any]] = []
    if target is None:
        m = re.search(r"(\d+)", observation_text(context))
        target = int(m.group(1)) if m else 3
        ops += [{"op": "set_environment", "key": "target", "value": target}, {"op": "set_environment", "key": "counter", "value": 0},
                {"op": "set_phase", "phase": "counting"}]
        counter = 0
    else:
        counter = int(env.get("counter", 0))
    if counter >= int(target):
        ops.append({"op": "set_phase", "phase": "done"})
        return AgentDecision(
            rationale_summary="Target reached.", state_patch={"expected_state_version": state["state_version"], "ops": ops},
            completion={"outcome": "success", "summary": f"counted to {target}", "final_answer": str(target)},
        )
    ops.append({"op": "set_environment", "key": "counter", "value": counter + 1})
    ops.append({"op": "set_observation_summary", "summary": f"counter now {counter + 1}"})
    return AgentDecision(
        rationale_summary=f"Increment to {counter + 1}.", state_patch={"expected_state_version": state["state_version"], "ops": ops},
        action={"kind": "continue", "note": f"count {counter + 1}/{target}"},
    )


def codebase_research_script(context: ModelContext) -> AgentDecision:
    """A fixed research procedure: list root, search for a term from the question, read the first hit, write a report."""
    state = parse_state(context)
    v = state["state_version"]
    phase = state["current_phase"]
    obs = observation_text(context)
    obs_event = observation_attr(context, "event_id")
    obs_kind = observation_attr(context, "kind")
    question = state["objective"]["statement"]
    m = re.search(r"`([^`]+)`|\"([^\"]+)\"|'([^']+)'", question)
    term = next((g for g in (m.groups() if m else ()) if g), None) or (question.split()[-1].strip("?.") if question else "TODO")

    if obs_kind == "runtime":
        # Our previous decision was rejected — this deterministic script simply gives up gracefully.
        return AgentDecision(rationale_summary="Runtime rejected the previous decision; failing.",
                             state_patch={"expected_state_version": v, "ops": []},
                             completion={"outcome": "failure", "summary": obs[:500]})

    if phase == "survey":
        if obs_kind == "task_input":
            return AgentDecision(
                rationale_summary="Survey the workspace root.",
                state_patch={"expected_state_version": v, "ops": [
                    {"op": "set_plan", "steps": ["list workspace", f"search for {term}", "read the relevant file", "write report"]},
                    {"op": "set_entity", "name": "search_term", "description": term},
                ]},
                action={"kind": "tool", "tool_name": "list_directory", "arguments": {"path": "."}, "purpose": "survey"},
            )
        entries = [line.split()[-1] for line in obs.splitlines() if line and line[0] in "dfl"]
        return AgentDecision(
            rationale_summary="Recorded layout; searching for the term.",
            state_patch={"expected_state_version": v, "ops": [
                {"op": "add_fact", "id": "fact_layout", "statement": f"workspace root contains: {', '.join(entries[:20])}", "evidence_event_ids": [obs_event]},
                {"op": "update_plan_step", "step_id": "plan_1", "status": "done"},
                {"op": "set_phase", "phase": "investigate"},
                {"op": "add_hypothesis", "id": "hyp_term", "statement": f"the term {term} appears in the workspace", "confidence": 0.5},
            ]},
            action={"kind": "tool", "tool_name": "search_text", "arguments": {"pattern": term, "path": "."}, "purpose": "find occurrences"},
        )

    if phase == "investigate":
        hyp_open = "hyp_term" in {h["id"] for h in state["active_hypotheses"]}
        if hyp_open and (obs.startswith("(no matches)") or obs.startswith("TOOL FAILED")):
            return AgentDecision(
                rationale_summary="Term not found; cannot answer.",
                state_patch={"expected_state_version": v, "ops": [
                    {"op": "reject_hypothesis", "hypothesis_id": "hyp_term", "reason": "search returned no matches", "evidence_event_ids": [obs_event]},
                    {"op": "set_phase", "phase": "done"},
                ]},
                completion={"outcome": "failure", "summary": f"'{term}' does not occur in the workspace", "final_answer": "not found"},
            )
        if hyp_open:
            first = obs.splitlines()[0]
            path = first.split(":", 1)[0]
            count = len([line for line in obs.splitlines() if ":" in line])
            return AgentDecision(
                rationale_summary="Term found; reading the first matching file.",
                state_patch={"expected_state_version": v, "ops": [
                    {"op": "promote_hypothesis", "hypothesis_id": "hyp_term", "evidence_event_ids": [obs_event]},
                    {"op": "add_fact", "id": "fact_matches", "statement": f"{term} occurs in {count} line(s); first in {path}", "evidence_event_ids": [obs_event]},
                    {"op": "add_artifact", "id": "art_first_hit", "kind": "file", "locator": path, "description": f"first file mentioning {term}", "originating_event_id": obs_event},
                    {"op": "update_plan_step", "step_id": "plan_2", "status": "done"},
                ]},
                action={"kind": "tool", "tool_name": "read_text_file", "arguments": {"path": path, "max_lines": 40}, "purpose": "inspect"},
            )
        if obs.startswith("TOOL FAILED"):
            return AgentDecision(
                rationale_summary="Could not read the file; reporting what is known.",
                state_patch={"expected_state_version": v, "ops": [
                    {"op": "add_blocker", "id": "blk_read", "description": "first matching file could not be read"},
                    {"op": "set_phase", "phase": "report"},
                ]},
                action={"kind": "continue", "note": "synthesize"},
            )
        lines = len(obs.splitlines())
        return AgentDecision(
            rationale_summary="Read the file; moving to report.",
            state_patch={"expected_state_version": v, "ops": [
                {"op": "add_fact", "id": "fact_read", "statement": f"read {lines} lines of the first matching file", "evidence_event_ids": [obs_event]},
                {"op": "update_plan_step", "step_id": "plan_3", "status": "done"},
                {"op": "set_phase", "phase": "report"},
                {"op": "set_observation_summary", "summary": f"file excerpt of {lines} lines inspected"},
            ]},
            action={"kind": "continue", "note": "synthesize"},
        )

    if phase == "report":
        if obs_kind == "tool_result" and obs.startswith("wrote"):
            return AgentDecision(
                rationale_summary="Report written; done.",
                state_patch={"expected_state_version": v, "ops": [
                    {"op": "add_artifact", "id": "art_report", "kind": "file", "locator": "RESEARCH_REPORT.md", "description": "final research report", "originating_event_id": obs_event},
                    {"op": "update_plan_step", "step_id": "plan_4", "status": "done"},
                    {"op": "set_phase", "phase": "done"},
                ]},
                completion={"outcome": "success", "summary": "report written",
                            "final_answer": "; ".join(f["statement"] for f in state["verified_facts"]), "artifact_ids": ["art_report"]},
            )
        facts = "\n".join(f"- {f['statement']} (evidence: {', '.join(f['evidence_event_ids'])})" for f in state["verified_facts"])
        report = f"# Research report\n\n**Question:** {question}\n\n**Answer:** see facts below.\n\n## Supporting facts\n{facts}\n"
        return AgentDecision(
            rationale_summary="Writing the report.",
            state_patch={"expected_state_version": v, "ops": []},
            action={"kind": "tool", "tool_name": "write_workspace_file", "arguments": {"path": "RESEARCH_REPORT.md", "content": report}, "purpose": "deliverable"},
        )

    return AgentDecision(rationale_summary="Nothing left to do.", state_patch={"expected_state_version": v, "ops": []},
                         completion={"outcome": "partial", "summary": f"unexpected phase {phase}"})


def load_script(dotted: str) -> Any:
    """Resolve ``package.module:function``."""
    import importlib

    mod_name, _, attr = dotted.partition(":")
    if not attr:
        raise ValueError("reasoner_script must be 'module:function'")
    return getattr(importlib.import_module(mod_name), attr)
