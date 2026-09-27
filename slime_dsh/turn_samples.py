"""Turn-level training samples for message-level agent clients (DSH over
OpenAI chat-completions).

Why this exists (round1-runlog incident #5, 2026-09-26): the upstream chain
builder keeps a turn's sampled tokens trainable only while every LATER prompt
still contains them verbatim. The OpenAI adapter's recorded ``manager_message``
deliberately drops ``reasoning_content`` (clients strip it on echo), so a
thinking agent's next-turn prompt renders without the think block it actually
sampled: token drift on every turn -> chain forks (or REALIGN overwrites the
sampled span as context) -> only the final message trains. Observed: 1-3% of
generated tokens in the loss mask across a whole run.

The industry answer is per-turn data -- verl's TITO golden rule ("never
re-encode tokens you've already received"), rllm's gateway TraceRecords,
Agent Lightning's transition-level aggregation. Every ``TurnRecord`` already
carries exactly that: the ``prompt_ids`` actually fed in that request, the
``output_ids`` actually sampled, and per-token logprobs. This module
linearizes ONE Sample per turn (full trajectory reward assigned to every
turn, shared rollout_id) instead of one per surviving chain. Contexts differ
per turn -- which is correct: each (prompt_ids, output_ids) pair is precisely
what the policy saw and did in that request.

Vendor stays untouched; this subclasses the adapter at the seam the
orchestrator already exposes (ADAPTER_CLS).
"""
from __future__ import annotations

import dataclasses
from typing import Any

from slime.utils.types import Sample


def adapter_cls(base_cls):
    """Wrap an OpenAIAdapter subclass (optionally the blocker's) with
    turn-level finish_session. Call once before the upstream singleton boots."""

    class _TurnSampleAdapter(base_cls):
        async def finish_session(
            self,
            sid: str,
            *,
            base_sample,
            reward: float = 0.0,
            extra_metadata: dict | None = None,
            wait_timeout: float = 5.0,
        ) -> list:
            await self.shutdown_session(sid, wait_timeout=wait_timeout)
            session = self.store.pop(sid, None)
            cap = int(getattr(session, "max_context_tokens", 0) or 0) if session is not None else 0
            root = getattr(self.manager, "_trees", {}).get(sid)
            samples: list[Sample] = []
            if root is not None:
                for leaf in root.leaves():
                    if leaf.is_root:
                        continue
                    for node in leaf.path_from_root():
                        turn = node.turn
                        if turn is None or node.response_trained:
                            continue
                        node.response_trained = True  # first leaf claims it (upstream semantics)
                        s = self._turn_to_sample(
                            turn, node, base_sample, reward, extra_metadata, cap)
                        if s is not None:
                            samples.append(s)
                self.manager._trees.pop(sid, None)
                getattr(self.manager, "_turn_count", {}).pop(sid, None)
            for s in samples:
                rlen = int(s.response_length or 0)
                s.response = (
                    self.tokenizer.decode(s.tokens[-rlen:], skip_special_tokens=False)
                    if rlen and s.tokens else "")
            return samples

        def _turn_to_sample(self, turn, node, base_sample, reward, extra_metadata, cap: int):
            md = {**(extra_metadata or {}),
                  "turn_level_sample": True,
                  "truncated": turn.finish_reason == "length",
                  "ill_formed": bool(turn.ill_formed),
                  "use_tool": bool((node.message or {}).get("tool_calls")),
                  "turn_finish_reason": turn.finish_reason}
            prompt_ids = list(turn.prompt_ids)
            output_ids = list(turn.output_ids)
            # session-level cap mirrors upstream to_sample truncation; a turn
            # whose RESPONSE no longer fits is dropped rather than half-trained
            if cap and len(prompt_ids) + len(output_ids) > cap:
                room = cap - len(prompt_ids)
                if room <= 0:
                    return None
                output_ids = output_ids[:room]
                turn = dataclasses.replace(
                    turn, output_ids=output_ids,
                    output_log_probs=list(turn.output_log_probs or [])[:room])
                md["truncated"] = True
            tokens = prompt_ids + output_ids
            # slime convention (ray/rollout.py:349 asserts it): loss_mask lives
            # in RESPONSE space, len(loss_mask) == response_length
            loss_mask = [1] * len(output_ids)
            return Sample(
                index=base_sample.index,
                group_index=base_sample.group_index,
                rollout_id=(base_sample.rollout_id if base_sample.rollout_id is not None
                            else base_sample.index),
                prompt=base_sample.prompt,
                label=base_sample.label,
                tokens=tokens,
                response_length=len(output_ids),
                loss_mask=loss_mask,
                rollout_log_probs=list(turn.output_log_probs or []),
                reward=reward,
                status=Sample.Status.COMPLETED,
                metadata=md,
            )

    return _TurnSampleAdapter


def bind(upstream_generate_module) -> dict[str, Any]:
    """Select the turn-sample adapter as ADAPTER_CLS (layered after the blocker)."""
    from slime.agent.adapters import OpenAIAdapter
    from slime_dsh import blocker

    base = (blocker.blocked_adapter_cls()
            if blocker.blocker_enabled() else OpenAIAdapter)
    upstream_generate_module.ADAPTER_CLS = adapter_cls(base)
    return {"adapter_cls": f"{base.__name__} + turn-level finish_session"}
