"""Qwen3CN tool-format token penalty: rule-based validation + token localization.

Locked design: research/reward-design-stage2-grpo.md §2.1/§2.3 (row "工具格式
token 级惩罚") and §3.3. Qwen3-Coder-Next TR §4.2.4 (arXiv 2603.00729): "At
each interaction step, we perform rule-based validation of tool-call format
correctness. During optimization, tokens associated with invalid tool calls
receive token-level penalties." MiniMax M2 TR §6.1.5 carries the same penalty
inside its process reward -- dual TR backing (design §4.6 point 4); no public
open-source implementation exists, hence this module.

Validation is rule-level over the Qwen tool-call text form the pinned adapter
parses (`<tool_call>{json}</tool_call>`, vendor/slime/slime/agent/parsing.py
delegating to sglang's function-call parser): a block is invalid when its JSON
does not decode to {"name": str, "arguments": dict}, when the name is not in
the configured tool set, or when the tags are unbalanced. Text outside
<tool_call> blocks is never penalized.

Token localization is exact for byte-level BPE tokenizers (Qwen family):
per-token byte strings come from convert_ids_to_tokens + the GPT-2
bytes-to-unicode inverse; their concatenation is asserted equal to the decoded
text's utf-8 bytes, so char spans map to token indices without drift.

Penalty enters AFTER group normalization, per token, only on supervised tokens
(loss-mask intersection): observation text echoing <tool_call> inside a tool
result is not the model's own output and is never penalized. c_fmt comes from
DSH_C_FMT (default 0 = pure binary smoke start). NOTE for the smoke sheet: with
DSH_C_FMT > 0, disable --normalize-advantages (post-hook whitening would rescale
the penalty away); ToolRL puts format rewards' saturation at ~30 steps (design
§4.7) -- plan the ablation arm and its retirement condition from logged
invalid-token rates.

Hook wiring (hyperparameter sheet decides): 
  --custom-advantage-function-path slime_dsh.format_penalty.apply_format_penalty_advantage
The hook replaces the estimator dispatch for grpo/gspo/cispo and replicates the
baseline (per-token broadcast of the already-normalized scalar reward,
identical to get_grpo_returns) before subtracting the penalty.
"""
from __future__ import annotations

import json
import logging
import os
import re

import torch

logger = logging.getLogger(__name__)

TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"

_TOKENIZER = None


def _bytes_to_unicode_inverse() -> dict[str, int]:
    """Standard GPT-2 byte-level BPE char->byte map (identical for Qwen tokenizers)."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\xa1"), ord("\xac") + 1))
        + list(range(ord("\xae"), ord("\xff") + 1))
    )
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


_INVERSE_MAP = _bytes_to_unicode_inverse()


def token_byte_lengths(token_ids: list[int], tokenizer) -> list[int]:
    """Byte length of each token under byte-level BPE; specials fall back to literal utf-8."""
    lengths = []
    for tok in tokenizer.convert_ids_to_tokens(list(token_ids)):
        if tok is None:
            lengths.append(0)
        elif all(ch in _INVERSE_MAP for ch in tok):
            lengths.append(len(tok))
        else:  # special tokens render literally in decode(skip_special_tokens=False)
            lengths.append(len(tok.encode("utf-8")))
    return lengths


def find_invalid_toolcall_spans(text: str, tool_names: set[str] | None = None) -> list[tuple[int, int]]:
    """Char spans of invalid tool-call blocks; sorted, non-overlapping."""
    openers = [m.start() for m in re.finditer(re.escape(TOOL_CALL_OPEN), text)]
    closers = [m.start() for m in re.finditer(re.escape(TOOL_CALL_CLOSE), text)]
    spans: list[tuple[int, int]] = []
    used = set()
    for start in openers:
        end = next((c for c in closers if c > start and c not in used), None)
        if end is None:
            spans.append((start, len(text)))
            continue
        used.add(end)
        block_end = end + len(TOOL_CALL_CLOSE)
        if not _valid_block(text[start + len(TOOL_CALL_OPEN) : end], tool_names):
            spans.append((start, block_end))
    for c in closers:
        if c not in used:  # stray closer without an opener
            spans.append((c, c + len(TOOL_CALL_CLOSE)))
    return sorted(set(spans))


def _valid_block(inner: str, tool_names: set[str] | None) -> bool:
    try:
        obj = json.loads(inner.strip() or "null")
    except json.JSONDecodeError:
        return False
    if not isinstance(obj, dict):
        return False
    name, arguments = obj.get("name"), obj.get("arguments")
    if not isinstance(name, str) or not name:
        return False
    if not isinstance(arguments, dict):
        return False
    if tool_names is not None and name not in tool_names:
        return False
    return True


def invalid_token_mask(response_token_ids: list[int], tokenizer, tool_names: set[str] | None = None) -> list[bool]:
    """Per-response-token flags for tokens inside invalid tool-call spans."""
    lengths = token_byte_lengths(response_token_ids, tokenizer)
    text = tokenizer.decode(list(response_token_ids), skip_special_tokens=False)
    text_bytes = text.encode("utf-8")
    if sum(lengths) != len(text_bytes):
        raise AssertionError(
            "byte-level alignment failed for this tokenizer: "
            f"sum(token bytes)={sum(lengths)} != decoded utf-8 bytes={len(text_bytes)}"
        )
    spans = find_invalid_toolcall_spans(text, tool_names)
    mask = [False] * len(response_token_ids)
    for start_char, end_char in spans:
        b0, b1 = len(text[:start_char].encode("utf-8")), len(text[:end_char].encode("utf-8"))
        acc = 0
        for i, n in enumerate(lengths):
            if n and acc < b1 and acc + n > b0:  # token byte range intersects the span
                mask[i] = True
            acc += n
    return mask


def c_fmt_from_env() -> float:
    return float(os.environ.get("DSH_C_FMT", "0") or 0)


def tool_names_from_env() -> set[str] | None:
    raw = os.environ.get("DSH_TOOL_NAMES", "").strip()
    return {n.strip() for n in raw.split(",") if n.strip()} if raw else None


def _get_tokenizer(args):
    global _TOKENIZER
    if _TOKENIZER is None:
        from slime.utils.processing_utils import load_tokenizer

        _TOKENIZER = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
    return _TOKENIZER


def _as_int_list(tokens) -> list[int]:
    if torch.is_tensor(tokens):
        return [int(t) for t in tokens.tolist()]
    return [int(t) for t in tokens]


def apply_format_penalty_advantage(args, rollout_data) -> None:
    """slime --custom-advantage-function-path hook (see module docstring)."""
    estimator = getattr(args, "advantage_estimator", "grpo")
    if estimator not in ("grpo", "gspo", "cispo"):
        raise NotImplementedError(f"format penalty hook supports grpo/gspo/cispo, got {estimator}")
    rewards = rollout_data["rewards"]
    kl = rollout_data["kl"]  # per-token response-aligned; used for shape only (baseline parity)
    returns = [torch.ones_like(kl[i]) * float(rewards[i]) for i in range(len(rewards))]

    c_fmt = c_fmt_from_env()
    if c_fmt != 0.0:
        tokenizer = _get_tokenizer(args)
        tool_names = tool_names_from_env()
        penalized = 0
        for i in range(len(returns)):
            loss_mask = rollout_data["loss_masks"][i]
            if len(loss_mask) != returns[i].shape[-1]:
                raise AssertionError(
                    f"sample {i}: loss_mask len {len(loss_mask)} != advantage len {returns[i].shape[-1]}"
                )
            response_len = int(rollout_data["response_lengths"][i])
            token_ids = _as_int_list(rollout_data["tokens"][i])[-response_len:]
            if len(token_ids) != len(loss_mask):
                raise AssertionError(f"sample {i}: response tokens {len(token_ids)} != loss mask {len(loss_mask)}")
            mask = invalid_token_mask(token_ids, tokenizer, tool_names)
            hit = [
                j for j, flag in enumerate(mask)
                if flag and float(loss_mask[j]) == 1.0  # supervised tokens only (orthogonal to loss_mask)
            ]
            if hit:
                penalized += len(hit)
                penalty = torch.zeros_like(returns[i])
                penalty[hit] = -c_fmt
                returns[i] = returns[i] + penalty
        if penalized:
            logger.info("[slime_dsh.format_penalty] c_fmt=%s penalized_tokens=%d", c_fmt, penalized)

    rollout_data["advantages"] = [r for r in returns]
    rollout_data["returns"] = returns
