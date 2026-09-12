#!/usr/bin/env python3
"""Wrapper that runs slime's convert_hf_to_torch_dist.py on CPU.

Registers project-local CPU compatibility shims first (accelerator backend,
RNG guards, spec/norm patches, fla stubs), then executes the upstream
conversion script unchanged via runpy. Usage mirrors the upstream tool;
MODEL_ARGS come from vendor/slime/scripts/models/qwen3.5-9B.sh.

Patch order matters: the fla stubs must be in sys.modules BEFORE the
qwen3_5 plugin imports `from fla.modules import ...` (its try/except swallows
ImportError and the names would stay unbound).
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parent.parent
os.environ.setdefault("SLIME_ACCELERATOR", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

sys.path.insert(0, str(PROJECT / "scripts"))

# ---------------------------------------------------------------------------
# 1. CPU accelerator backend (slime only registers cuda/musa)
# ---------------------------------------------------------------------------
import cpu_accel_shim  # noqa: F401  (registers the cpu backend on import)

# ---------------------------------------------------------------------------
# 2. RNG guards: megatron's model_parallel_cuda_manual_seed touches torch.cuda
# RNG APIs unconditionally; CPU-only torch builds raise. Seeding is irrelevant
# for conversion (weights load deterministically), so no-op the seed calls and
# route state get/set to the CPU RNG. NOTE: manual_seed must NOT delegate to
# torch.manual_seed - that internally calls torch.cuda.manual_seed_all and
# would recurse back into these patches.
# ---------------------------------------------------------------------------
torch.cuda.manual_seed = lambda *a, **k: None
torch.cuda.manual_seed_all = lambda *a, **k: None
torch.cuda.get_rng_state = lambda *a, **k: torch.get_rng_state()
_orig_set_rng_state = torch.set_rng_state
torch.cuda.set_rng_state = lambda state, *a, **k: _orig_set_rng_state(state)

# megatron's get_model moves modules to cuda unconditionally; keep everything
# on CPU for this conversion-only process.
torch.cuda.current_device = lambda: 0
torch.nn.Module.cuda = lambda self, device=None: self

# ---------------------------------------------------------------------------
# 3. fla stubs (flash-linear-attention is a triton package absent on CPU).
# The Qwen3.5 GDN module resolves fla kernels/modules at __init__ but only
# invokes them at forward time; conversion never runs forward. The
# weight-bearing stubs must expose the exact parameter names/shapes the HF
# checkpoint uses, because hf_to_megatron's qwen3_5 mapping copies
# linear_attn.* tensors by identical name.
# ---------------------------------------------------------------------------
import types  # noqa: E402
import torch.nn as _nn  # noqa: E402


class _StubShortConvolution(_nn.Module):
    """Depthwise causal conv container; weight shape [hidden, 1, kernel]."""

    def __init__(self, hidden_size, kernel_size=4, bias=False, **kwargs):
        super().__init__()
        self.weight = _nn.Parameter(torch.empty(hidden_size, 1, kernel_size))
        if bias:
            self.bias = _nn.Parameter(torch.zeros(hidden_size))

    def forward(self, *a, **k):
        raise RuntimeError("conversion stub must never run forward")


class _StubFusedRMSNormGated(_nn.Module):
    def __init__(self, hidden_size, eps=1e-5, activation=None, device=None, dtype=None, **kwargs):
        super().__init__()
        self.weight = _nn.Parameter(torch.ones(hidden_size, dtype=dtype or torch.get_default_dtype()))

    def forward(self, *a, **k):
        raise RuntimeError("conversion stub must never run forward")


_fla = types.ModuleType("fla")
_fla_ops = types.ModuleType("fla.ops")
_fla_gdr = types.ModuleType("fla.ops.gated_delta_rule")
_fla_gdr.chunk_gated_delta_rule = lambda *a, **k: None  # never invoked during conversion
_fla_modules = types.ModuleType("fla.modules")
_fla_modules.ShortConvolution = _StubShortConvolution
_fla_modules.FusedRMSNormGated = _StubFusedRMSNormGated
_fla.ops = _fla_ops
_fla.modules = _fla_modules
_fla_ops.gated_delta_rule = _fla_gdr

# importlib.util.find_spec("fla...") requires real ModuleSpecs on the parents
from importlib.machinery import ModuleSpec  # noqa: E402

for _mod, _name, _pkg in (
    (_fla, "fla", True),
    (_fla_ops, "fla.ops", True),
    (_fla_gdr, "fla.ops.gated_delta_rule", False),
    (_fla_modules, "fla.modules", False),
):
    _mod.__spec__ = ModuleSpec(_name, None, is_package=_pkg)
    if _pkg:
        _mod.__path__ = []

sys.modules.setdefault("fla", _fla)
sys.modules.setdefault("fla.ops", _fla_ops)
sys.modules.setdefault("fla.ops.gated_delta_rule", _fla_gdr)
sys.modules.setdefault("fla.modules", _fla_modules)

# ---------------------------------------------------------------------------
# 4. The pinned qwen3_5 spec plugin hardcodes use_transformer_engine=True (the
# upstream training image always has TE). Force the local spec for this
# CPU-only conversion by wrapping the plugin's decoder-block-spec call in place.
# ---------------------------------------------------------------------------
import slime_plugins.models.qwen3_5 as _qwen35_plugin  # noqa: E402

_orig_block_spec = _qwen35_plugin.get_gpt_decoder_block_spec


def _local_block_spec(config, use_transformer_engine, **kwargs):  # noqa: ANN001
    return _orig_block_spec(config, use_transformer_engine=False, **kwargs)


_qwen35_plugin.get_gpt_decoder_block_spec = _local_block_spec

# ---------------------------------------------------------------------------
# 5. WrappedTorchNorm (apex-less fallback) refuses to construct under
# zero-centered-gamma / sequence-parallel configs. Weight layout is [hidden]
# either way and forward is never executed during conversion, so construct the
# plain torch norm directly (asserts dropped for this process only).
# ---------------------------------------------------------------------------
from megatron.core.transformer import torch_norm as _torch_norm  # noqa: E402


def _torch_norm_new(cls, config, hidden_size, eps=1e-5, **kwargs):  # noqa: ANN001
    if config.normalization == "LayerNorm":
        return torch.nn.LayerNorm(normalized_shape=hidden_size, eps=eps)
    if config.normalization == "RMSNorm":
        return torch.nn.RMSNorm(normalized_shape=hidden_size, eps=eps)
    raise Exception(f"conversion fallback supports LayerNorm/RMSNorm, got {config.normalization}")


_torch_norm.WrappedTorchNorm.__new__ = _torch_norm_new

# ---------------------------------------------------------------------------
# Run the upstream converter unchanged.
# ---------------------------------------------------------------------------
CONVERTER = PROJECT / "vendor/slime/tools/convert_hf_to_torch_dist.py"
sys.argv[0] = str(CONVERTER)
runpy.run_path(str(CONVERTER), run_name="__main__")
