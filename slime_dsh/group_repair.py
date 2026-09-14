"""GLM-5 noise hygiene: crash exclusion, group pad/drop, and group statistics.

Locked design: research/reward-design-stage2-grpo.md §2.3 (row "组填充/丢弃")
and §3.2. GLM-5 TR §4.1.2 (arXiv 2602.15763): "we record the failure reason for
each sample and exclude samples that fail due to environment collapse. For
group-based sampling methods such as GRPO, removing failed samples can leave an
incomplete group. In that case, we pad the group by repeating valid samples if
the number of valid samples exceeds half of the group size; otherwise, we drop
the entire group."

Why slime needs help here (design's "半缺口"): Sample.remove_sample zeroes the
loss mask but the sample still participates in advantage normalization
(vendor/slime/slime/utils/arguments.py, --rollout-sample-filter-path help), and
the default group normalization reshapes rewards into (batch, K) which silently
degenerates to batch-global mean normalization whenever segment counts vary
(vendor/slime/slime/ray/rollout.py _post_process_rewards else-branch) -- which
multi-segment DSH rollouts always do.

Two independent hooks (wire per the smoke hyperparameter sheet):

  --rollout-all-samples-process-path slime_dsh.group_repair.all_samples_process
      Runs after generation, before train-data conversion. Group shape here is
      list[K slots], each slot = list of segment Samples from one request.
      Applies the unfinished penalty (reward core from slime_dsh.reward), then,
      when DSH_GROUP_REPAIR=1, repairs groups in place: environment-crash slots
      are excluded; if valid slots > K/2 the crash slots are replaced by
      round-robin repeats of valid slots (same Sample objects, GLM-5 "pad by
      repeating"); otherwise every sample in the group is remove_sample=True
      ("drop the entire group"). In-place slot replacement propagates to the
      training data because the rollout loop and all_samples share group objects.

  --custom-reward-post-process-path slime_dsh.group_repair.reward_post_process
      Replaces _post_process_rewards: per-group mean (and optional std)
      normalization computed over non-environment samples only; environment
      samples and dropped groups get normalized reward 0.0 (their loss masks are
      zero). Groups are consecutive instance_id runs (the coding-agent metadata
      contract); samples without instance_id fall back to positional reshaping.

Both default-off knobs are transport, not decisions: DSH_GROUP_REPAIR=1 enables
GLM-5 pad/drop (v1 smoke stays on native DAPO dynamic sampling + the upstream
abort channel per the locked staged rollout), DSH_C_UNFINISHED sets the
unfinished penalty coefficient (default 0 = pure binary core).
"""
from __future__ import annotations

import logging
import os

from slime_dsh.reward import FailureClass, classify_failure, compute_reward, c_unfinished_from_env

logger = logging.getLogger(__name__)


def _slot_is_environment_crash(slot: list) -> bool:
    return any(classify_failure(s) == FailureClass.ENVIRONMENT for s in slot)


def _apply_unfinished_penalty(samples) -> float:
    """Adjust rewards via the locked reward core; returns the number changed."""
    c = c_unfinished_from_env()
    if c == 0.0:
        return 0
    changed = 0
    for s in samples:
        adjusted, info = compute_reward(s, c_unfinished=c)
        if info["unfinished"]:
            md = s.metadata if isinstance(s.metadata, dict) else {}
            md.setdefault("raw_reward", getattr(s, "reward", None))
            md["failure_class"] = info["failure_class"]
            s.metadata = md
            s.reward = adjusted
            changed += 1
    return changed


def repair_group(group: list) -> str:
    """GLM-5 pad/drop on one in-place group (list[K slots] of segment lists).

    Returns the repair verdict: "passthrough" | "padded" | "dropped".
    """
    k = len(group)
    crash_idx = [i for i, slot in enumerate(group) if _slot_is_environment_crash(slot)]
    if not crash_idx:
        return "passthrough"
    valid_idx = [i for i in range(k) if i not in set(crash_idx)]
    if len(valid_idx) <= k // 2:
        for slot in group:
            for s in slot:
                s.remove_sample = True
        return "dropped"
    for pos, crash_i in enumerate(crash_idx):
        donor = group[valid_idx[pos % len(valid_idx)]]
        group[crash_i] = list(donor)  # same Sample objects: GLM-5 "repeating valid samples"
    return "padded"


def group_repair_enabled() -> bool:
    return os.environ.get("DSH_GROUP_REPAIR", "").strip().lower() in {"1", "true", "yes", "on"}


def all_samples_process(args, all_samples, data_source) -> None:
    """slime --rollout-all-samples-process-path hook (see module docstring)."""
    for group in all_samples:  # defensive slot normalization (upstream aborts return [sample])
        for i, slot in enumerate(group):
            if not isinstance(slot, list):
                group[i] = [slot]
    flat = [s for group in all_samples for slot in group for s in slot]
    penalized = _apply_unfinished_penalty(flat)
    verdicts = {"passthrough": 0, "padded": 0, "dropped": 0}
    if group_repair_enabled():
        for group in all_samples:
            verdicts[repair_group(group)] += 1
    if penalized or verdicts["padded"] or verdicts["dropped"]:
        logger.info(
            "[slime_dsh.group_repair] groups passthrough=%d padded=%d dropped=%d unfinished_penalized=%d",
            verdicts["passthrough"], verdicts["padded"], verdicts["dropped"], penalized,
        )


def _group_samples(samples: list, group_size: int) -> list[list]:
    """Consecutive instance_id runs; positional fallback for metadata-less samples."""
    has_ids = all(isinstance(s.metadata, dict) and s.metadata.get("instance_id") is not None for s in samples)
    if not has_ids:
        return [samples[i : i + group_size] for i in range(0, len(samples), group_size)]
    groups, current, current_id = [], [], None
    for s in samples:
        sid = s.metadata.get("instance_id")
        if current and sid != current_id:
            groups.append(current)
            current = []
        current_id = sid
        current.append(s)
    if current:
        groups.append(current)
    return groups


def reward_post_process(args, samples):
    """slime --custom-reward-post-process-path hook: group stats without crashes.

    Mirrors _post_process_rewards semantics (mean subtraction, optional
    (std + 1e-6) division for grpo/gspo/cispo) but per instance_id group and
    computed over non-environment samples only; environment samples receive
    normalized reward 0.0 (loss masks are zero downstream).
    """
    raw_rewards = [
        s.reward if not getattr(args, "reward_key", None) else s.reward[args.reward_key] for s in samples
    ]
    if not (
        getattr(args, "advantage_estimator", "grpo")
        in ["grpo", "gspo", "cispo", "reinforce_plus_plus_baseline"]
        and getattr(args, "rewards_normalization", True)
    ):
        return raw_rewards, list(raw_rewards)

    rewards = [0.0] * len(samples)
    group_size = int(getattr(args, "n_samples_per_prompt", 1) or 1)
    cursor = 0  # groups are consecutive partitions of the flat sample list
    for group in _group_samples(samples, group_size):
        keep = [i for i, s in enumerate(group) if classify_failure(s) != FailureClass.ENVIRONMENT]
        if keep:  # all-environment group: dropped, normalized reward stays 0.0
            vals = [raw_rewards[cursor + i] for i in keep]
            mean = sum(vals) / len(vals)
            centered = [raw_rewards[cursor + i] - mean for i in keep]
            if (
                getattr(args, "advantage_estimator", "grpo") in ["grpo", "gspo", "cispo"]
                and getattr(args, "grpo_std_normalization", False)
            ):
                # sample variance (torch.std Bessel semantics, vendor parity)
                var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1) if len(vals) > 1 else 0.0
                std = var**0.5
                centered = [v / (std + 1e-6) for v in centered]
            for i, value in zip(keep, centered):
                rewards[cursor + i] = value
        cursor += len(group)
    return raw_rewards, rewards
