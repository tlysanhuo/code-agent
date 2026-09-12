#!/usr/bin/env python3
"""Provision + qualify round-1 RL tasks; emit registry + prompt parquet.

For each task from the frozen RL order (curated-repo prefix):
  1. build/reuse the cached per-requirements env (gym_prepare.install)
  2. qualification on a throwaway case: apply GOLD patch -> frozen evaluator
     must resolve (same standard as the 20-task dev baseline)
  3. keep a BUGGY case for rollout (the agent works here)
  4. registry entry (instance_id -> case/python) + prompt-parquet row whose
     metadata matches slime_dsh.local_backend's contract

Run with .venv-train-rl (pyarrow) from the project root. Sequential: env
builds and grading are the cost; envs cache by requirements hash so same-repo
tasks amortize to ~one build each.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import types as _types
if "gym_facility" not in sys.modules:
    _stub = _types.ModuleType("gym_facility")
    _stub.execute = None
    sys.modules["gym_facility"] = _stub

import gym_prepare as gp
import gym_run as gr

CURATED_REPOS = {"dask/dask", "pydantic/pydantic", "facebookresearch/hydra", "bokeh/bokeh"}
SOURCES = ROOT / "runtime/gym-baseline/sources"


def fetch_source(row) -> Path:
    """Download the pinned base_commit archive (codeload) with a sha256
    sidecar; extract with swe_prepare.extract_bases strip-prefix semantics.
    Idempotent."""
    dst = SOURCES / row["instance_id"]
    if dst.exists():
        return dst
    tgz = SOURCES / (row["instance_id"] + ".tar.gz")
    sidecar = SOURCES / (row["instance_id"] + ".json")
    if not tgz.exists():
        url = f"https://codeload.github.com/{row['repo']}/tar.gz/{row['base_commit']}"
        part = tgz.with_name(tgz.name + ".part")
        h = hashlib.sha256()
        with urllib.request.urlopen(url, timeout=60) as r, part.open("wb") as f:
            while True:
                b = r.read(1024 * 1024)
                if not b:
                    break
                h.update(b)
                f.write(b)
        part.replace(tgz)
        sidecar.write_text(json.dumps({
            "instance_id": row["instance_id"], "repo": row["repo"],
            "base_commit": row["base_commit"], "url": url,
            "bytes": tgz.stat().st_size, "sha256": h.hexdigest()}, indent=1) + "\n")
    dst.mkdir()
    with tarfile.open(tgz) as archive:
        prefix = archive.getmembers()[0].name.split("/")[0] + "/"
        for member in archive.getmembers():
            if member.name.startswith(prefix):
                member.name = member.name[len(prefix):]
                if member.name:
                    archive.extract(member, dst, filter="data")
    return dst


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--source", choices=["gym", "smith"], default="gym")
    ap.add_argument("--out", type=Path, default=ROOT / "runtime/agent-rl/round1")
    ap.add_argument("--parquet", type=Path,
                    default=ROOT / "data/agent-rl/rl-round1-prompts.parquet")
    ap.add_argument("--registry", type=Path,
                    default=ROOT / "configs/agent-rl/local-task-registry.json")
    args = ap.parse_args()

    order_file = ROOT / f"data/agent-rl/rl-freeze-v1-{'gym' if args.source=='gym' else 'smith'}-order.txt"
    order = [l.strip() for l in order_file.read_text().splitlines() if l.strip()]
    repo_of = {}
    # resolve repos from the raw parquet
    import pyarrow.parquet as pq
    t = pq.read_table(ROOT / "data/raw/SWE-Gym__SWE-Gym/data/train-00000-of-00001.parquet",
                      columns=["instance_id", "repo"])
    repo_of = dict(zip(t["instance_id"].to_pylist(), t["repo"].to_pylist()))

    candidates = [i for i in order if repo_of.get(i) in CURATED_REPOS][: args.limit]
    print(f"candidates: {len(candidates)} from {order_file.name} (curated repos)", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "qual").mkdir(exist_ok=True)
    registry = {}
    prompts = []
    consumed = []
    t0 = time.time()
    for n, iid in enumerate(candidates, 1):
        try:
            row = gp.task(iid)
            fetch_source(row)
            python = Path(gp.install(row))
            qcase = args.out / "qual" / iid
            qcase.mkdir(parents=True, exist_ok=True)
            gp.make_case(qcase, row, python)
            ws = qcase / "sandbox/workspace"
            subprocess.run(["git", "apply", "--binary", "-"], input=row["patch"].encode(),
                           cwd=ws, check=True)
            res = gr.evaluate(qcase, row, python)
            ok = bool(res.get("resolved"))
            print(f"[{n}/{len(candidates)}] {iid}: qual={'OK' if ok else res.get('category')}", flush=True)
            if not ok:
                continue
            case = args.out / iid
            gp.make_case(case, row, python)  # buggy rollout case
            registry[iid] = {"case_dir": str(case), "python": str(python)}
            consumed.append(iid)
            meta = {
                "instance_id": iid,
                "image": "local",
                "workdir": str(case / "sandbox" / "workspace"),
                "problem_statement": row["problem_statement"],
                "local": {"case_dir": str(case), "python": str(python), "row": row},
            }
            prompts.append({"prompt": row["problem_statement"], "metadata": meta})
        except Exception as exc:
            print(f"[{n}/{len(candidates)}] {iid}: ERROR {type(exc).__name__}: {str(exc)[:150]}", flush=True)
    # persist everything even on partial failure
    args.registry.parent.mkdir(parents=True, exist_ok=True)
    args.registry.write_text(json.dumps(registry, indent=1) + "\n")
    (args.out / "consumed.txt").write_text("".join(i + "\n" for i in consumed))
    import pyarrow as pa
    args.parquet.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(prompts), args.parquet)
    print(json.dumps({"qualified": len(registry), "of": len(candidates),
                      "registry": str(args.registry), "parquet": str(args.parquet),
                      "minutes": round((time.time() - t0) / 60, 1)}, indent=2))


if __name__ == "__main__":
    main()
