"""CPU unit checks for the stage-2 reward stack (locked design §3, slime_dsh/).

Covers, per the design's unit-test list:
  reward.py        binary core, unfinished penalty, failure classification, custom_rm
  group_repair.py  all-crash drop / partial-crash pad / no-crash passthrough,
                   group statistics without crash rewards, slime-parity normalization
  format_penalty   rule validation, exact byte-level token localization against the
                   REAL Qwen3.5 tokenizer, advantage hook incl. loss-mask orthogonality
  blocker.py       link+keyword hits, pure link / pure keyword passes, feedback keeps
                   the trajectory continuable (tool result carries the feedback)

No GPU, no ray, no sglang, no megatron. Run:
  source scripts/env.sh
  PYTHONPATH="$CODE_AGENT_ROOT:$CODE_AGENT_ROOT/vendor/slime" \
    .venv-train-rl/bin/python scripts/check_reward_stack.py
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import py_compile
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor/slime"))

from slime.agent.parsing import ParsedModelOutput  # noqa: E402
from slime.utils.types import Sample  # noqa: E402

from slime_dsh import blocker, format_penalty, group_repair, reward  # noqa: E402

TOKENIZER_DIR = ROOT / "models/Qwen3.5-9B/c202236235762e1c871ad0ccb60c8ee5ba337b9a"
_ENV_KEYS = ["DSH_C_UNFINISHED", "DSH_GROUP_REPAIR", "DSH_C_FMT", "DSH_TOOL_NAMES", "DSH_BLOCKER"]


def sample(**kw) -> Sample:
    base = dict(index=0, group_index=0, reward=0.0, loss_mask=None, response_length=0, metadata={})
    base.update(kw)
    return Sample(**base)


def setenv(**kv) -> None:
    for k in _ENV_KEYS:
        os.environ.pop(k, None)
    for k, v in kv.items():
        if v is not None:
            os.environ[k] = v


# ---------------------------------------------------------------- reward.py

def check_reward_core() -> None:
    s = sample(metadata={"grading_solved": True}, reward=1.0)
    assert reward.binary_outcome(s) == 1.0
    s2 = sample(metadata={"grading_solved": False}, reward=0.0)
    assert reward.binary_outcome(s2) == 0.0
    s3 = sample(metadata={}, reward=0.0)  # metadata fallback
    assert reward.binary_outcome(s3) == 0.0
    r, info = reward.compute_reward(s, c_unfinished=0.5)
    assert r == 1.0 and info["failure_class"] == "none" and not info["unfinished"]


def check_classification() -> None:
    cases = [
        ({"abort_reason": "wall_clock_timeout"}, "aborted", reward.FailureClass.BUDGET),
        ({"abort_reason": "exception:RuntimeError"}, "aborted", reward.FailureClass.ENVIRONMENT),
        ({"abort_reason": "unevaluatable:no_f2p"}, "aborted", reward.FailureClass.ENVIRONMENT),
        ({"abort_reason": "missing_image_or_workdir"}, "aborted", reward.FailureClass.ENVIRONMENT),
        ({"abort_reason": "adapter_session_empty"}, "aborted", reward.FailureClass.ENVIRONMENT),
        ({"abort_reason": "grading_error:ValueError"}, "aborted", reward.FailureClass.ENVIRONMENT),
        ({}, "aborted", reward.FailureClass.ENVIRONMENT),  # aborted, reason unrecorded
        ({"truncated": True}, "completed", reward.FailureClass.BUDGET),
        ({"agent_exit_code": -1}, "completed", reward.FailureClass.BUDGET),
        ({"agent_exit_code": 0, "grading_solved": False}, "completed", reward.FailureClass.MODEL),
        ({"agent_exit_code": 0, "grading_solved": True}, "completed", reward.FailureClass.NONE),
    ]
    for md, status, expected in cases:
        s = sample(metadata=dict(md), status=getattr(Sample.Status, status.upper()) if status != "aborted" else Sample.Status.ABORTED)
        assert reward.classify_failure(s) == expected, (md, status, reward.classify_failure(s))


def check_unfinished_penalty() -> None:
    setenv(DSH_C_UNFINISHED="0.5")
    solved_overbudget = sample(metadata={"grading_solved": True, "agent_exit_code": -1}, reward=1.0)
    r, info = reward.compute_reward(solved_overbudget, c_unfinished=reward.c_unfinished_from_env())
    assert r == 0.5 and info["unfinished"]
    unsolved_overbudget = sample(metadata={"grading_solved": False, "truncated": True}, reward=0.0)
    r2, _ = reward.compute_reward(unsolved_overbudget, c_unfinished=0.5)
    assert r2 == -0.5  # literal locked formula: core - c * 1[unfinished]
    setenv()
    # custom_rm: single + batch list, async signature
    single = asyncio.run(reward.custom_rm(None, solved_overbudget))
    batch = asyncio.run(reward.custom_rm(None, [solved_overbudget, unsolved_overbudget]))
    assert single == 1.0 and batch == [1.0, 0.0]  # env cleared -> pure binary core


# ------------------------------------------------------------ group_repair.py

def group(slots: list[list[Sample]], instance_id: str) -> list[list[Sample]]:
    for slot in slots:
        for s in slot:
            s.metadata = {**s.metadata, "instance_id": instance_id}
    return slots


def check_repair_group() -> None:
    # no crash -> passthrough, untouched
    g = group([[sample(metadata={"grading_solved": True}, reward=1.0)],
               [sample(metadata={"grading_solved": False}, reward=0.0)]], "t0")
    before = [list(slot) for slot in g]
    assert group_repair.repair_group(g) == "passthrough"
    assert g == before
    # K=4, 1 environment crash, 3 valid -> padded (valid > K/2)
    g = group([
        [sample(metadata={"grading_solved": True}, reward=1.0)],
        [sample(metadata={"abort_reason": "exception:X"}, status=Sample.Status.ABORTED, remove_sample=True)],
        [sample(metadata={"grading_solved": False}, reward=0.0)],
        [sample(metadata={"grading_solved": True}, reward=1.0)],
    ], "t1")
    assert group_repair.repair_group(g) == "padded"
    assert len(g) == 4
    assert g[1] == g[0] and g[1] is not g[0]  # crash slot replaced by repeat of valid slot 0
    flat = [s for slot in g for s in slot]
    assert all(not s.remove_sample for s in flat)
    assert {s.reward for s in flat} == {1.0, 0.0}  # crash reward 0 excluded by replacement
    assert sum(s.reward for s in flat) / len(flat) == (1 + 0 + 1 + 1) / 4  # stats over valid only
    # K=4, 2 environment crashes (valid == K/2, not > half) -> dropped
    g = group([
        [sample(metadata={"grading_solved": True}, reward=1.0)],
        [sample(metadata={"abort_reason": "exception:X"}, status=Sample.Status.ABORTED)],
        [sample(metadata={"abort_reason": "adapter_session_empty"}, status=Sample.Status.ABORTED)],
        [sample(metadata={"grading_solved": False}, reward=0.0)],
    ], "t2")
    assert group_repair.repair_group(g) == "dropped"
    assert all(s.remove_sample for slot in g for s in slot)
    # K=4 all crash -> dropped; K=3, 1 crash -> padded (2 > 1.5); K=2, 1 crash -> dropped (1 <= 1)
    gall = group([[sample(metadata={"abort_reason": "exception:X"}, status=Sample.Status.ABORTED)] for _ in range(4)], "t3")
    assert group_repair.repair_group(gall) == "dropped"
    g3 = group([[sample(metadata={"grading_solved": True}, reward=1.0)],
                [sample(metadata={"abort_reason": "exception:X"}, status=Sample.Status.ABORTED)],
                [sample(metadata={"grading_solved": False}, reward=0.0)]], "t4")
    assert group_repair.repair_group(g3) == "padded" and len(g3) == 3
    g2 = group([[sample(metadata={"grading_solved": True}, reward=1.0)],
                [sample(metadata={"abort_reason": "exception:X"}, status=Sample.Status.ABORTED)]], "t5")
    assert group_repair.repair_group(g2) == "dropped"
    # budget-class abort is NOT excluded from the group (only environment collapses are)
    gb = group([[sample(metadata={"grading_solved": True}, reward=1.0)],
                [sample(metadata={"abort_reason": "wall_clock_timeout"}, status=Sample.Status.ABORTED)]], "t6")
    assert group_repair.repair_group(gb) == "passthrough"


def check_all_samples_process() -> None:
    # unfinished penalty rides the hook; group repair gated by env
    setenv(DSH_C_UNFINISHED="0.5", DSH_GROUP_REPAIR="1")
    over = sample(metadata={"grading_solved": True, "agent_exit_code": -1, "instance_id": "z"}, reward=1.0)
    ok = sample(metadata={"grading_solved": True, "instance_id": "z"}, reward=1.0)
    crash = sample(metadata={"abort_reason": "exception:Y", "instance_id": "z"}, status=Sample.Status.ABORTED)
    groups = [group([[over], [ok], [crash], [ok]], "z")]
    data_alias = groups  # the rollout loop's data and all_samples share group objects
    group_repair.all_samples_process(argparse.Namespace(), groups, None)
    assert over.reward == 0.5 and over.metadata["raw_reward"] == 1.0  # penalty applied, raw preserved
    assert groups is data_alias and len(groups[0]) == 4
    assert groups[0][2] == groups[0][0]  # crash slot padded from the first valid slot (round-robin)
    setenv()
    # group repair off: crash slot stays (upstream abort channel handles it), penalty off too
    crash2 = sample(metadata={"abort_reason": "exception:Y", "instance_id": "w"}, status=Sample.Status.ABORTED)
    ok2 = sample(metadata={"grading_solved": True, "instance_id": "w"}, reward=1.0)
    groups2 = [[[ok2], [crash2]]]
    group_repair.all_samples_process(argparse.Namespace(), groups2, None)
    assert groups2[0][1] == [crash2] and ok2.reward == 1.0


def ns(**kv) -> argparse.Namespace:
    return argparse.Namespace(**kv)


def check_reward_post_process() -> None:
    # stats exclude environment crashes (GLM-5): mean over [1,0,1] not over [1,0(env),0,1]
    a = sample(metadata={"instance_id": "g", "grading_solved": True}, reward=1.0)
    b = sample(metadata={"instance_id": "g", "abort_reason": "exception:Z"}, status=Sample.Status.ABORTED, reward=0.0)
    c = sample(metadata={"instance_id": "g", "grading_solved": False}, reward=0.0)
    d = sample(metadata={"instance_id": "g", "grading_solved": True}, reward=1.0)
    raw, norm = group_repair.reward_post_process(ns(rewards_normalization=True, n_samples_per_prompt=4), [a, b, c, d])
    assert raw == [1.0, 0.0, 0.0, 1.0]
    assert all(abs(x - y) < 1e-9 for x, y in zip(norm, [1 / 3, 0.0, -2 / 3, 1 / 3]))  # env crash 0.0, stats over {1,0,1}
    # two instance_id groups normalize independently
    e = sample(metadata={"instance_id": "h", "grading_solved": True}, reward=1.0)
    f = sample(metadata={"instance_id": "h", "grading_solved": False}, reward=0.0)
    _, norm2 = group_repair.reward_post_process(
        ns(rewards_normalization=True, n_samples_per_prompt=2), [a, c, e, f])
    assert norm2 == [0.5, -0.5, 0.5, -0.5]
    # std normalization parity with slime (torch sample std, +1e-6)
    _, norm3 = group_repair.reward_post_process(
        ns(rewards_normalization=True, grpo_std_normalization=True, n_samples_per_prompt=4,
           advantage_estimator="grpo"), [a, c, d, sample(metadata={"instance_id": "g"}, reward=0.0)])
    vals = torch.tensor([1.0, 0.0, 1.0, 0.0])
    expected = (vals - vals.mean()) / (vals.std() + 1e-6)
    assert torch.allclose(torch.tensor(norm3), expected, atol=1e-6), (norm3, expected)
    # normalization off -> identity
    _, norm4 = group_repair.reward_post_process(
        ns(rewards_normalization=False, n_samples_per_prompt=2), [a, c])
    assert norm4 == [1.0, 0.0]
    # all-environment group: dropped, all zeros
    x = sample(metadata={"instance_id": "k", "abort_reason": "exception:Z"}, status=Sample.Status.ABORTED)
    y = sample(metadata={"instance_id": "k", "abort_reason": "exception:W"}, status=Sample.Status.ABORTED)
    _, norm5 = group_repair.reward_post_process(ns(rewards_normalization=True, n_samples_per_prompt=2), [x, y])
    assert norm5 == [0.0, 0.0]
    # metadata-less samples fall back to positional grouping (slime default shape)
    p = [sample(reward=float(v), metadata={}) for v in (1, 0, 1, 1)]
    _, norm6 = group_repair.reward_post_process(ns(rewards_normalization=True, n_samples_per_prompt=2), p)
    assert norm6 == [0.5, -0.5, 0.0, 0.0]  # positional fallback groups [1,0] and [1,1]
    # padded duplicates (same objects twice) keep per-occurrence positions correct
    v1 = sample(metadata={"instance_id": "m", "grading_solved": True}, reward=1.0)
    v2 = sample(metadata={"instance_id": "m", "grading_solved": False}, reward=0.0)
    _, norm7 = group_repair.reward_post_process(ns(rewards_normalization=True, n_samples_per_prompt=3), [v1, v2, v1])
    assert all(abs(x - y) < 1e-9 for x, y in zip(norm7, [1 / 3, -2 / 3, 1 / 3]))


# ---------------------------------------------------------- format_penalty.py

VALID_CALL = '<tool_call>\n{"name": "terminal-bash", "arguments": {"command": "ls -la"}}\n</tool_call>'
INVALID_JSON_CALL = '<tool_call>\n{"name": "terminal-bash", "arguments": {"command": "ls\n</tool_call>'
BAD_ARGS_CALL = '<tool_call>\n{"name": "terminal-bash", "arguments": "ls"}\n</tool_call>'


def check_span_validation() -> None:
    text = f"prose {VALID_CALL} more prose"
    assert format_penalty.find_invalid_toolcall_spans(text) == []
    wrapped = f"a {INVALID_JSON_CALL} b"
    spans = format_penalty.find_invalid_toolcall_spans(wrapped)
    assert spans == [(wrapped.find("<tool_call>"), wrapped.find("</tool_call>") + len("</tool_call>"))]
    assert format_penalty.find_invalid_toolcall_spans(BAD_ARGS_CALL)
    assert format_penalty.find_invalid_toolcall_spans(VALID_CALL, tool_names={"other-tool"})
    assert format_penalty.find_invalid_toolcall_spans(VALID_CALL, tool_names={"terminal-bash"}) == []
    unclosed = "x <tool_call> {broken json"
    spans_u = format_penalty.find_invalid_toolcall_spans(unclosed)
    assert spans_u == [(2, len(unclosed))]
    stray = "a </tool_call> b"
    spans_s = format_penalty.find_invalid_toolcall_spans(stray)
    assert spans_s == [(2, 2 + len("</tool_call>"))]
    # no false positives on prose that merely mentions tools
    assert format_penalty.find_invalid_toolcall_spans("you should run git clone then curl the docs") == []


def check_tokenizer_alignment() -> None:
    assert (TOKENIZER_DIR / "tokenizer_config.json").is_file(), f"tokenizer missing at {TOKENIZER_DIR}"
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR), trust_remote_code=True)
    text = (
        "I will inspect.\n" + VALID_CALL + "\n"
        "TOOL OUTPUT: docs say <tool_call>{\"name\": \"x\", \"arguments\": truncated\n</tool_call> end\n"
        + INVALID_JSON_CALL
    )
    ids = tok.encode(text, add_special_tokens=False)
    lengths = format_penalty.token_byte_lengths(ids, tok)
    assert sum(lengths) == len(tok.decode(ids, skip_special_tokens=False).encode("utf-8"))  # exact byte alignment
    mask = format_penalty.invalid_token_mask(ids, tok)
    assert any(mask)
    decoded_per_token = [tok.decode([i], skip_special_tokens=False) for i in ids]
    joined = tok.decode(ids, skip_special_tokens=False)
    # every masked token decodes inside an invalid span's char range
    spans = format_penalty.find_invalid_toolcall_spans(joined)
    assert len(spans) == 2  # observation-region invalid block + model invalid block
    acc = 0
    for flag, frag in zip(mask, decoded_per_token):
        if flag:
            assert any(sp <= acc and acc + len(frag.encode("utf-8")) <= ep for sp, ep in spans), (acc, frag)
        acc += len(frag.encode("utf-8"))


def check_advantage_hook() -> None:
    from transformers import AutoTokenizer

    setenv(DSH_C_FMT="0.25")
    tok = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR), trust_remote_code=True)
    model_a = "Inspecting.\n" + VALID_CALL + "\n"
    observation = "TOOL OUTPUT: <tool_call>{\"name\": \"x\", \"arguments\": truncated\n</tool_call>\n"
    model_b = "Trying again.\n" + INVALID_JSON_CALL
    a_ids = tok.encode(model_a, add_special_tokens=False)
    o_ids = tok.encode(observation, add_special_tokens=False)
    b_ids = tok.encode(model_b, add_special_tokens=False)
    prompt_ids = tok.encode("SYSTEM prompt", add_special_tokens=False)
    resp_ids = a_ids + o_ids + b_ids
    tokens = prompt_ids + resp_ids
    loss_mask = torch.tensor([1.0] * len(a_ids) + [0.0] * len(o_ids) + [1.0] * len(b_ids))
    clean_ids = tok.encode("All good.\n" + VALID_CALL, add_special_tokens=False)
    clean_prompt = tok.encode("P2", add_special_tokens=False)

    args = ns(advantage_estimator="grpo", hf_checkpoint=str(TOKENIZER_DIR))
    rollout_data = {
        "rewards": [0.5, -0.5],
        "kl": [torch.zeros(len(resp_ids)), torch.zeros(len(clean_ids))],
        "loss_masks": [loss_mask, torch.ones(len(clean_ids))],
        "tokens": [tokens, clean_prompt + clean_ids],
        "response_lengths": [len(resp_ids), len(clean_ids)],
    }
    format_penalty.apply_format_penalty_advantage(args, rollout_data)
    adv, ret = rollout_data["advantages"], rollout_data["returns"]
    # baseline parity: clean sample is a pure reward broadcast
    assert torch.allclose(adv[1], torch.full_like(adv[1], -0.5)) and torch.equal(adv[1], ret[1])
    # penalized positions: invalid block inside model_b only (mask==1), value = reward - c_fmt
    base = torch.full_like(adv[0], 0.5)
    hit = (adv[0] - base) != 0
    assert hit.sum() > 0
    assert torch.allclose(adv[0][hit], torch.full((int(hit.sum()),), 0.25))
    obs_lo, obs_hi = len(a_ids), len(a_ids) + len(o_ids)
    assert not any(hit[obs_lo:obs_hi])  # observation tokens never penalized (orthogonal to loss_mask)
    assert not any(hit[: len(a_ids)])   # the valid call in model_a is untouched
    # c_fmt = 0 -> hook is a pure baseline replication
    setenv()
    rollout_data2 = {
        "rewards": [1.0], "kl": [torch.zeros(4)], "loss_masks": [torch.ones(4)],
        "tokens": [[1, 2, 3, 4]], "response_lengths": [4],
    }
    format_penalty.apply_format_penalty_advantage(args, rollout_data2)
    assert torch.allclose(rollout_data2["advantages"][0], torch.ones(4))
    # estimator guard
    try:
        format_penalty.apply_format_penalty_advantage(ns(advantage_estimator="ppo"), rollout_data2)
        raise AssertionError("ppo should be rejected")
    except NotImplementedError:
        pass
    setenv()


# ---------------------------------------------------------------- blocker.py

BLOCKED_CMD = "git clone https://github.com/attacker/repo.git upstream"


def check_blocker_rules() -> None:
    assert blocker.string_is_blocked(BLOCKED_CMD)
    assert blocker.string_is_blocked("curl -s git@github.com:attacker/repo.git")
    assert not blocker.string_is_blocked("echo https://github.com/a/b")     # pure link, no keyword
    assert not blocker.string_is_blocked("git status")                       # pure keyword, no link
    assert not blocker.string_is_blocked("ls -la && cat README.md")
    call = {"name": "terminal-bash", "input": {"command": BLOCKED_CMD, "timeout_ms": 30000}}
    assert blocker.scan_call(call) == [BLOCKED_CMD]  # nested scan, other args untouched
    rewritten = blocker.rewrite_call(call)
    assert rewritten["input"]["timeout_ms"] == 30000
    assert rewritten["input"]["command"].startswith("echo '") and "BLOCKED" in rewritten["input"]["command"]
    assert not blocker.string_is_blocked(rewritten["input"]["command"])


def check_blocker_trajectory_continues() -> None:
    parsed = ParsedModelOutput(
        reasoning="", text="fetching upstream",
        tool_uses=[{"name": "terminal-bash", "input": {"command": BLOCKED_CMD}}],
    )
    new_parsed, count, examples = blocker.apply_blocker(parsed)
    assert count == 1 and examples == [BLOCKED_CMD]
    assert new_parsed.tool_uses[0]["input"]["command"].startswith("echo '")
    assert new_parsed.text == "fetching upstream"  # model text preserved
    # clean call passes through untouched
    clean = ParsedModelOutput(reasoning="", text="", tool_uses=[{"name": "bash", "input": {"command": "ls"}}])
    same, n, ex = blocker.apply_blocker(clean)
    assert n == 0 and same is clean and not ex
    # adapter-level: reply keeps a tool call (agent continues), feedback is the command,
    # wire/manager arguments agree so the next-turn history echo still matches
    cls = blocker.blocked_adapter_cls()
    adapter = object.__new__(cls)  # _build_reply is stateless; no server needed
    reply = adapter._build_reply(parsed, "stop", [], [])
    wire_msg, wire_finish = reply.wire
    assert wire_finish == "tool_calls"
    sent = wire_msg["tool_calls"][0]["function"]["arguments"]
    assert "BLOCKED" in sent and "github.com" not in sent
    assert reply.manager_message["tool_calls"][0]["function"]["arguments"] == json.loads(sent)
    # env gate
    setenv(DSH_BLOCKER="0")
    assert not blocker.blocker_enabled()
    setenv(DSH_BLOCKER="1")
    assert blocker.blocker_enabled()
    setenv()


def main() -> None:
    py_compile.compile(str(ROOT / "slime_dsh/generate.py"), doraise=True)  # wiring edit syntax check
    checks = [
        ("reward.core", check_reward_core),
        ("reward.classification", check_classification),
        ("reward.unfinished_penalty", check_unfinished_penalty),
        ("group_repair.repair_group", check_repair_group),
        ("group_repair.all_samples_process", check_all_samples_process),
        ("group_repair.reward_post_process", check_reward_post_process),
        ("format_penalty.span_validation", check_span_validation),
        ("format_penalty.tokenizer_alignment", check_tokenizer_alignment),
        ("format_penalty.advantage_hook", check_advantage_hook),
        ("blocker.rules", check_blocker_rules),
        ("blocker.trajectory_continues", check_blocker_trajectory_continues),
    ]
    for name, fn in checks:
        fn()
        print(f"PASS {name}", flush=True)
    setenv()
    print(f"OK reward-stack checks: {len(checks)} passed")


if __name__ == "__main__":
    main()
