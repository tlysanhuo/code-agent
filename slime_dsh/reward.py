"""Stage-2 reward core: binary outcome + optional unfinished penalty + failure classification.

Locked design: research/reward-design-stage2-grpo.md §2.1/§3.1 (user approval
2026-09-12). The reward domain is

    R = 1.0 * [F2P fully passing + P2P not regressing, frozen gym evaluator]   (binary core; D11a)
        - c_unfinished * 1[trajectory did not terminate within budget]         (Qwen3CN unfinished penalty)

with c_unfinished read from DSH_C_UNFINISHED (default 0 = pure binary smoke
start, per the locked staged rollout). Anti-cheat lives in the execution layer
(slime_dsh.blocker), not in the reward; environment collapse is excluded from
the reward domain entirely and handled by slime_dsh.group_repair (GLM-5 rule).

Failure classification (GLM-5 TR §4.1.2 "record the failure reason for each
sample"): ENVIRONMENT failures are excluded from group statistics; BUDGET
signals only drive the unfinished penalty; MODEL failures are ordinary reward-0
samples. The upstream abort channel (examples/coding_agent_rl/generate.py
_abort_result) is the metadata source; nothing here re-runs evaluation.

"Unfinished" observable signals (turn counts are not carried in sample metadata
by the pinned upstream; approximation documented in the locked design):
  * metadata.agent_exit_code < 0  - DSH harness reports time budget exceeded
  * metadata.truncated            - adapter finish_reason == "length"

Hook wiring (values go to the GRPO smoke hyperparameter sheet, not here):
  --custom-rm-path slime_dsh.reward.custom_rm
The agent rollout path sets sample.reward inside generate(), so rm_hub only
calls custom_rm for reward-less samples; the SAME core (compute_reward) is also
applied by slime_dsh.group_repair.all_samples_process on every sample before
group statistics, which is the integration point that actually runs in the
coding-agent flow.
"""
from __future__ import annotations

import enum
import os


class FailureClass(enum.Enum):
    """Why a trajectory ended without solving, per GLM-5 §4.1.2."""

    NONE = "none"                # solved, or ordinary model failure (reward 0, no infra cause)
    ENVIRONMENT = "environment"  # infra collapse: exclude from group stats (group_repair)
    MODEL = "model"              # model produced work, failed tests: plain reward 0
    BUDGET = "budget"            # trajectory did not terminate within budget: unfinished penalty


# abort_reason prefixes from upstream _abort_result + our local evaluator's
# grading_error category; wall_clock_timeout is budget-class, the rest are
# infrastructure collapses (KAT-V2.5: sandbox reliability decides training).
_ENVIRONMENT_ABORT_PREFIXES = (
    "missing_image_or_workdir",
    "unevaluatable",
    "adapter_session_empty",
    "exception",
    "grading",
)


def classify_failure(sample) -> FailureClass:
    """Classify one rollout sample from its metadata (GLM-5 failure-reason recording)."""
    md = sample.metadata if isinstance(sample.metadata, dict) else {}
    reason = md.get("abort_reason")
    if reason:
        if reason == "wall_clock_timeout":
            return FailureClass.BUDGET
        if any(str(reason).startswith(p) for p in _ENVIRONMENT_ABORT_PREFIXES):
            return FailureClass.ENVIRONMENT
        return FailureClass.ENVIRONMENT  # unknown abort causes are infra-conservative
    status = getattr(sample, "status", None)
    if status is not None and getattr(status, "value", status) == "aborted":
        return FailureClass.ENVIRONMENT  # aborted without a recorded reason
    if md.get("truncated"):
        return FailureClass.BUDGET  # context-length exhaustion
    exit_code = md.get("agent_exit_code")
    if isinstance(exit_code, (int, float)) and exit_code < 0:
        return FailureClass.BUDGET  # DSH harness: time budget exceeded
    # completed within budget: model-attributable failure when tests did not pass
    return FailureClass.MODEL if binary_outcome(sample) == 0.0 else FailureClass.NONE


def is_unfinished(sample) -> bool:
    """Budget-exhaustion signal driving the Qwen3CN unfinished penalty."""
    return classify_failure(sample) == FailureClass.BUDGET


def binary_outcome(sample) -> float:
    """Frozen gym evaluator verdict: 1.0 iff F2P fully passing + P2P not regressing."""
    md = sample.metadata if isinstance(sample.metadata, dict) else {}
    if md.get("grading_solved") is not None:
        return 1.0 if md["grading_solved"] else 0.0
    reward = getattr(sample, "reward", None)
    if isinstance(reward, (int, float)):
        return 1.0 if float(reward) == 1.0 else 0.0
    return 0.0


def compute_reward(sample, *, c_unfinished: float = 0.0) -> tuple[float, dict]:
    """Final reward for one sample plus a classification/penalty info dict."""
    cls = classify_failure(sample)
    unfinished = cls == FailureClass.BUDGET
    reward = binary_outcome(sample) - (c_unfinished if unfinished else 0.0)
    info = {
        "failure_class": cls.value,
        "unfinished": unfinished,
        "c_unfinished": c_unfinished,
        "binary_outcome": binary_outcome(sample),
    }
    return reward, info


def c_unfinished_from_env() -> float:
    """Hyperparameters stay in the user-approved smoke sheet; env is the transport."""
    return float(os.environ.get("DSH_C_UNFINISHED", "0") or 0)


async def custom_rm(args, sample, **kwargs) -> float:
    """slime --custom-rm-path hook; accepts one Sample or a list (batched_async_rm)."""
    c = c_unfinished_from_env()
    if isinstance(sample, list):
        return [compute_reward(s, c_unfinished=c)[0] for s in sample]
    return compute_reward(sample, c_unfinished=c)[0]
