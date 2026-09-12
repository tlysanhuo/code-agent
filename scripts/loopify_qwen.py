"""P1 surgery: convert dense Qwen3.5-9B into a looped (recurrent-depth) layout.

CPU-only, streaming per-tensor safetensors I/O; no GPU, no inference, no training.

Subcommands:
  plan     Derive and validate the layer sharing map from config.json alone.
  surgery  Write models/loop-qwen/{compact,expanded} from the downloaded snapshot.
           compact   = shared-base training form (shared-away layer tensors dropped)
           expanded  = standard 32-layer dense layout with duplicated donor weights,
                       directly servable by vLLM/SGLang as a plain Qwen3.5 checkpoint
  verify   Check expanded tensors equal their source/donor originals.

Run inside the project after `source scripts/env.sh`. Designed for
.venv-runtime python (torch + safetensors available).
"""
import argparse
import hashlib
import json
import random
import re
import shutil
import time
from pathlib import Path

from safetensors.torch import save_file
import torch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "models/Qwen3.5-9B/c202236235762e1c871ad0ccb60c8ee5ba337b9a"
OUT = ROOT / "models/loop-qwen"
LAYER_RE = re.compile(r"^(.*)\.layers\.(\d+)\.(.+)$")

KEEP_FIRST = 4
KEEP_LAST = 4
LOOPS = 2


def build_map(layer_types):
    n = len(layer_types)
    middle = n - KEEP_FIRST - KEEP_LAST
    assert middle % LOOPS == 0, f"middle {middle} not divisible by {LOOPS} loops"
    block = middle // LOOPS
    donor_lo, donor_hi = KEEP_FIRST, KEEP_FIRST + block
    shared = {}
    for j in range(donor_hi, KEEP_FIRST + middle):
        donor = j - block
        assert layer_types[j] == layer_types[donor], (
            f"layer_types mismatch: {j} ({layer_types[j]}) vs donor {donor} "
            f"({layer_types[donor]}); sharing must preserve the attention pattern")
        shared[j] = donor
    return {
        "num_hidden_layers": n,
        "layer_types": layer_types,
        "keep_first": list(range(0, KEEP_FIRST)),
        "keep_last": list(range(n - KEEP_LAST, n)),
        "donor_block": [donor_lo, donor_hi],
        "loops": LOOPS,
        "share_offset": block,
        "shared_to_donor": {str(k): v for k, v in shared.items()},
        "unique_layers": [i for i in range(n) if i not in shared],
        "unique_fraction": f"{len([i for i in range(n) if i not in shared])}/{n}",
    }


def load_plan_config(source):
    cfg = json.loads((source / "config.json").read_text())
    text = cfg.get("text_config", cfg)
    layer_types = text["layer_types"]
    assert len(set(layer_types)) <= 3, "unexpected layer_types alphabet"
    return cfg, layer_types


def split_name(name):
    m = LAYER_RE.match(name)
    if not m:
        return None
    return m.group(1), int(m.group(2)), m.group(3)


def donor_name(name, donor_idx):
    prefix, _, suffix = split_name(name)
    return f"{prefix}.layers.{donor_idx}.{suffix}"


def open_shards(source, index):
    handles = {}
    for shard in sorted(set(index["weight_map"].values())):
        from safetensors import safe_open
        handles[shard] = safe_open(source / shard, framework="pt")
    return handles


def read(handles, index, name):
    return handles[index["weight_map"][name]].get_tensor(name)


def cmd_plan(args):
    _, layer_types = load_plan_config(args.source)
    layer_map = build_map(layer_types)
    print(json.dumps(layer_map, indent=2))
    full = layer_types.count("full_attention")
    linear = layer_types.count("linear_attention")
    print(json.dumps({
        "pattern": f"{linear} linear_attention + {full} full_attention",
        "validation": "shared pairs preserve layer_types; pattern period alignment ok",
    }))


def cmd_surgery(args):
    cfg, layer_types = load_plan_config(args.source)
    layer_map = build_map(layer_types)
    shared = {int(k): v for k, v in layer_map["shared_to_donor"].items()}
    index = json.loads((args.source / "model.safetensors.index.json").read_text())
    handles = open_shards(args.source, index)

    compact_dir = OUT / "compact"
    expanded_dir = OUT / "expanded"
    for d in (compact_dir, expanded_dir):
        if d.exists() and not args.force:
            raise SystemExit(f"refusing to overwrite {d} without --force")
        d.mkdir(parents=True, exist_ok=True)

    provenance = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_model": "Qwen/Qwen3.5-9B",
        "source_revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        "keep_first": KEEP_FIRST, "keep_last": KEEP_LAST, "loops": LOOPS,
        "layer_map": layer_map,
        "notes": "hard-tied surgery: shared layers duplicate donor weights verbatim; "
                 "per-loop LoRA (P2) trains on top of the compact base",
    }

    # Group tensor names by output shard, preserving the original assignment.
    compact_shards, expanded_shards = {}, {}
    n_shared = n_kept = 0
    for name in index["weight_map"]:
        shard = index["weight_map"][name]
        parsed = split_name(name)
        if parsed is None or parsed[1] not in shared:
            compact_shards.setdefault(shard, []).append(name)
            expanded_shards.setdefault(shard, []).append(name)
            n_kept += 1
            continue
        n_shared += 1
        expanded_shards.setdefault(shard, []).append(name)  # compact drops these

    for label, groups, duplicate in (
            ("compact", compact_shards, False), ("expanded", expanded_shards, True)):
        for shard, names in sorted(groups.items()):
            payload = {}
            for name in names:
                parsed = split_name(name)
                if duplicate and parsed is not None and parsed[1] in shared:
                    payload[name] = read(handles, index, donor_name(name, shared[parsed[1]])).clone()
                else:
                    payload[name] = read(handles, index, name).clone()
            save_file(payload, str(OUT / label / shard),
                      metadata={"format": "pt", "surgery": label})
            del payload
        print(json.dumps({"form": label, "shards": len(groups), "tensors": sum(len(v) for v in groups.values())}), flush=True)
    del handles

    (compact_dir / "loop-map.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (expanded_dir / "loop-map.json").write_text(json.dumps(provenance, indent=2) + "\n")
    # expanded stays a complete HF checkpoint: copy every non-weight file verbatim
    for f in args.source.iterdir():
        if f.name.endswith(".safetensors") or f.name == "loop-map.json":
            continue
        if f.is_file():
            shutil.copy2(f, expanded_dir / f.name)
    print(json.dumps({
        "kept_tensors": n_kept, "shared_tensors_duplicated": n_shared,
        "compact": str(compact_dir.relative_to(ROOT)),
        "expanded": str(expanded_dir.relative_to(ROOT)),
    }))


def cmd_verify(args):
    _, layer_types = load_plan_config(args.source)
    layer_map = build_map(layer_types)
    shared = {int(k): v for k, v in layer_map["shared_to_donor"].items()}
    index = json.loads((args.source / "model.safetensors.index.json").read_text())
    src = open_shards(args.source, index)
    out_index = json.loads((OUT / "expanded" / "model.safetensors.index.json").read_text())
    out = open_shards(OUT / "expanded", out_index)

    layer_names = [n for n in index["weight_map"] if split_name(n) is not None]
    nonlayer = [n for n in index["weight_map"] if split_name(n) is None]
    rng = random.Random(0)
    donor_lo, donor_hi = layer_map["donor_block"]
    span_end = KEEP_FIRST + (layer_map["num_hidden_layers"] - KEEP_FIRST - KEEP_LAST)
    edges = {KEEP_FIRST - 1, donor_lo, donor_hi - 1, donor_hi, span_end - 1,
             layer_map["num_hidden_layers"] - 1}
    boundary = [n for n in layer_names if split_name(n)[1] in edges]
    sample = set(nonlayer) | set(boundary)
    sample |= {n for n in rng.sample(layer_names, min(24, len(layer_names)))}
    sample |= {n for n in layer_names if split_name(n)[1] in shared}

    checked = mismatched = 0
    for name in sorted(sample):
        parsed = split_name(name)
        is_shared = parsed is not None and parsed[1] in shared
        expect_src = donor_name(name, shared[parsed[1]]) if is_shared else name
        got = out[out_index["weight_map"][name]].get_tensor(name)
        want = src[index["weight_map"][expect_src]].get_tensor(expect_src)
        ok = got.dtype == want.dtype and got.shape == want.shape and torch.equal(got, want)
        checked += 1
        mismatched += 0 if ok else 1
        if not ok:
            print(json.dumps({"mismatch": name, "expected_from": expect_src}))
    print(json.dumps({"checked": checked, "mismatched": mismatched,
                      "status": "verified" if mismatched == 0 else "FAILED"}))
    if mismatched:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("plan")
    surg = sub.add_parser("surgery")
    surg.add_argument("--force", action="store_true")
    sub.add_parser("verify")
    args = parser.parse_args()
    if args.cmd == "plan":
        cmd_plan(args)
    elif args.cmd == "surgery":
        cmd_surgery(args)
    else:
        cmd_verify(args)


if __name__ == "__main__":
    main()
