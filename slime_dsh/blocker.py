"""Qwen3CN anti-cheat interceptor: repo-link + network-keyword tool-call blocking.

Locked design: research/reward-design-stage2-grpo.md §2.4/§3.4. Qwen3-Coder-Next
TR §4.2.4 (arXiv 2603.00729): "Any tool call containing both a repository link
(e.g. github.com/{repo}) and network-access keywords (e.g. git, curl, wget) is
blocked, and the agent receives explicit feedback indicating the prohibited
action" -- kept despite our network-namespace default-deny egress because
Qwen3CN observed exploits emerging as capability grows (their Figure 7).

Layer: defense-in-depth at the adapter reply path, after the model output is
parsed and before the tool call reaches the client harness. An offending call
is NOT dropped: the matched command string is rewritten to a harmless echo that
prints the feedback, so the tool result carries the explicit feedback and the
trajectory continues (Qwen3CN "the agent receives explicit feedback", the
design's "agent 可继续任务而非直接终止"). Rewrites flow into both the wire and
manager messages through the unchanged _build_reply_parts, so the next-turn
history echo still matches and the trajectory tree does not fork. Training
tokens are unaffected: they are the model's actual generated token ids recorded
at request time, not the reply we hand to the client.

The local sandbox already strips git remotes/branches/tags at environment
preparation (standard protections, design §2.4.1) and the network namespace is
the authoritative egress deny; this module is the second layer.

Enable explicitly with DSH_BLOCKER=1 in the rollout launch environment (opt-in
keeps existing evidence paths byte-identical until the RL smoke arms it; the
wiring belongs to the smoke hyperparameter sheet).
"""
from __future__ import annotations

import dataclasses
import logging
import re

logger = logging.getLogger(__name__)

REPO_LINK = re.compile(
    r"(?:github|gitlab|bitbucket|gitee)\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.%$~-]+"
    r"|git@[A-Za-z0-9.-]+:[A-Za-z0-9_./-]+\.git"
)
# Qwen3CN: "network-access keywords (e.g. git, curl, wget)"; word-boundary match.
NETWORK_KEYWORDS = ("git", "curl", "wget", "ssh", "scp", "nc")

FEEDBACK = (
    "BLOCKED (anti-cheat): this tool call combined a repository link with a network-access "
    "keyword; retrieving upstream history or code is prohibited. Work offline on the local checkout."
)


def string_is_blocked(s: str) -> bool:
    """True when one string carries BOTH a repository link and a network keyword."""
    return bool(REPO_LINK.search(s)) and any(re.search(rf"\b{re.escape(k)}\b", s) for k in NETWORK_KEYWORDS)


def scan_call(call: dict) -> list[str]:
    """Offending strings inside one parsed tool call ({name, input})."""
    found: list[str] = []

    def walk(obj):
        if isinstance(obj, str):
            if string_is_blocked(obj):
                found.append(obj)
        elif isinstance(obj, dict):
            for v in obj.values():
                walk(v)
        elif isinstance(obj, list):
            for v in obj:
                walk(v)

    walk(call.get("input"))
    return found


def echo_feedback() -> str:
    return "echo '" + FEEDBACK + "'"


def rewrite_call(call: dict) -> dict:
    """Deep copy of the call with every offending string replaced by the feedback echo."""
    def walk(obj):
        if isinstance(obj, str):
            return echo_feedback() if string_is_blocked(obj) else obj
        if isinstance(obj, dict):
            return {k: walk(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [walk(v) for v in obj]
        return obj

    return {"name": call.get("name", "tool"), "input": walk(call.get("input"))}


def apply_blocker(parsed):
    """Filter a ParsedModelOutput; returns (new_parsed, blocked_count, blocked_examples)."""
    if not getattr(parsed, "tool_uses", None):
        return parsed, 0, []
    rewritten, examples = [], []
    for call in parsed.tool_uses:
        offending = scan_call(call)
        if offending:
            examples.extend(offending)
            rewritten.append(rewrite_call(call))
        else:
            rewritten.append(call)
    if not examples:
        return parsed, 0, []
    return dataclasses.replace(parsed, tool_uses=rewritten), len(examples), examples


def blocker_enabled() -> bool:
    import os

    return os.environ.get("DSH_BLOCKER", "").strip().lower() in {"1", "true", "yes", "on"}


def blocked_adapter_cls():
    """OpenAIAdapter subclass with the interceptor wired into reply building."""
    from slime.agent.adapters.openai import OpenAIAdapter

    class BlockingOpenAIAdapter(OpenAIAdapter):
        def _build_reply(self, parsed, raw_finish, translated, tools_schema):
            parsed, blocked, examples = apply_blocker(parsed)
            if blocked:
                logger.warning(
                    "[slime_dsh.blocker] blocked %d offending tool-call string(s): %s",
                    blocked,
                    [e[:80] for e in examples],
                )
            return super()._build_reply(parsed, raw_finish, translated, tools_schema)

    return BlockingOpenAIAdapter
