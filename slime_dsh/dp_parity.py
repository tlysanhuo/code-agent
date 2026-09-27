"""DP-schedule parity guard for the round1 GSPO run (runlog incident #8, 2026-09-27).

What happened: with native turn stitching each session normally yields ONE
Sample, but the upstream chain builder still forks a session into two Samples
whenever it detects residual token drift (tool-call re-serialization etc.).
Rollout 2 of attempt 15 had 3 forked sessions -> 64 + 3 = 67 samples. Our
trajectories (~21-32k tokens) each exceed ``--max-tokens-per-gpu`` 8192, so
dynamic packing leaves every micro-batch a singleton and the mbs count equals
the sample count; an ODD count can never reach ``build_dp_schedule``'s
``align_to = dp_size`` multiple and the run dies on

    AssertionError: dynamic path: could only produce 67 mbs after maximal
    splitting; need 68. step 0 has 67 samples, below the alignment threshold (2).

Fix: a ``--rollout-sample-filter-path`` hook that drops the minimum number of
REDUNDANT fork renderings (a fork pair shares one trajectory and one reward;
keeping one of the two loses nothing) so the total is a multiple of the DP
size. Even totals are a strict no-op. Reward semantics untouched: this runs
BEFORE the default reward post-processing, so GSPO re-centers over the
surviving siblings exactly as if the fork had never happened.
"""
from __future__ import annotations


def _dp_align(args) -> int:
    """Training DP size as configured for this launch (TP2 x DP2 on 4 GPUs)."""
    try:
        align = (
            args.actor_num_gpus_per_node
            // args.tensor_model_parallel_size
            // args.pipeline_model_parallel_size
        )
        if align >= 1:
            return align
    except (AttributeError, TypeError, ZeroDivisionError):
        pass
    return 2


def _reward_value(sample) -> float:
    r = getattr(sample, "reward", None)
    if isinstance(r, (list, tuple)):
        r = r[0] if r else None
    try:
        return float(r)
    except (TypeError, ValueError):
        return 0.0


def sample_filter(args, data) -> None:
    """``--rollout-sample-filter-path`` hook; mutates ``data`` in place.

    ``data`` shape (agent path): list of prompt-groups, each a list of
    ``n_samples_per_prompt`` session-results, each session-result the
    ``list[Sample]`` one generate() call returned. The rollout manager
    flattens with ``chain.from_iterable`` AFTER this hook, so deleting from
    the innermost lists removes the sample from training entirely.
    """
    align = _dp_align(args)

    def _slots():
        for group in data:
            if not isinstance(group, list):
                continue
            for slot in group:
                if isinstance(slot, list):
                    yield slot

    slots = list(_slots())
    total = sum(len(s) for s in slots)
    drop_n = total % align
    if drop_n == 0:
        return

    forked = [s for s in slots if len(s) > 1]
    if len(forked) < drop_n:
        raise RuntimeError(
            f"dp_parity: total {total} needs {drop_n} drop(s) for align {align} "
            f"but only {len(forked)} forked session(s) available; refusing to "
            f"drop unique trajectories"
        )

    dropped = []
    for _ in range(drop_n):
        # prefer a zero-reward duplicate: fork pairs share the trajectory
        # reward, so this only ever discards a redundant losing rendering
        target, idx = None, -1
        for s in forked:
            if len(s) <= 1:
                continue
            for k, samp in enumerate(s):
                if _reward_value(samp) == 0.0:
                    target, idx = s, k
                    break
            if target is not None:
                break
        if target is None:  # every fork pair is all-reward-1: drop the shortest
            target = max((s for s in forked if len(s) > 1), key=len)
            idx = 0
        dropped.append(target.pop(idx))

    print(
        f"[dp_parity] dropped {len(dropped)} forked duplicate(s) "
        f"(total {total} -> {total - len(dropped)}, align {align}); "
        f"rewards={[round(_reward_value(s), 3) for s in dropped]}",
        flush=True,
    )


if __name__ == "__main__":
    # self-test with the exact attempt-15 rollout-2 shape: 64 sessions,
    # three of them forked (61*1 + 3*2 = 67) -> one drop -> 66.
    class _S:
        def __init__(self, reward):
            self.reward = reward

    class _Args:
        actor_num_gpus_per_node = 4
        tensor_model_parallel_size = 2
        pipeline_model_parallel_size = 1

    def build(n_forked):
        data, idx = [], 0
        for _g in range(8):
            group = []
            for _s in range(8):
                fork = idx < n_forked * 10 and idx % 10 == 3  # spread forks
                samples = [_S(0.0), _S(0.0)] if fork else [_S(1.0 if idx % 7 == 0 else 0.0)]
                group.append(samples)
                idx += 1
            data.append(group)
        return data

    d = build(3)
    sample_filter(_Args(), d)
    total = sum(len(s) for g in d for s in g)
    assert total == 66, total
    assert all(len(s) >= 1 for g in d for s in g)

    d = build(4)  # 68 already even -> strict no-op
    before = [len(s) for g in d for s in g]
    sample_filter(_Args(), d)
    assert [len(s) for g in d for s in g] == before

    d = build(0)  # 64 unique trajectories, odd impossible here; test the guard
    try:
        d[0][0].append(_S(0.0)); d[0][0].append(_S(0.0)); d[0][0].append(_S(0.0))  # 67, 1 forked
        sample_filter(_Args(), d)
        assert sum(len(s) for g in d for s in g) == 66
    except RuntimeError:
        raise AssertionError("guard fired although a fork existed")
    print("dp_parity self-test OK")
