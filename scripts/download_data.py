"""Download pinned public dataset files with bounded streaming and checksums.

Uses previously captured HF metadata, never executes downloaded code. Reruns
verify existing files. Interrupted files are restarted, not trusted as complete.
"""
import concurrent.futures
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from urllib.parse import quote

import requests

ROOT = Path(__file__).resolve().parents[1]
SELECTIONS = {
    "SWE-Gym/OpenHands-SFT-Trajectories": "data/",
    "SWE-Gym/SWE-Gym": "data/",
    "SWE-bench/SWE-smith-trajectories": "data/ticks-",
    "SWE-bench/SWE-smith": "data/",
    "nebius/SWE-rebench-V2": "data/",
}


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def fetch(item):
    repo, revision, info = item
    name = info["rfilename"]
    out = ROOT / "data/raw" / repo.replace("/", "__") / name
    out.parent.mkdir(parents=True, exist_ok=True)
    expected_size = info["size"]
    expected_sha = info.get("lfs", {}).get("sha256")
    url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{quote(name, safe='/')}"
    for attempt in range(3):
        try:
            cached = out.exists() and out.stat().st_size == expected_size
            if cached:
                sha = digest(out)
                cached = not expected_sha or sha == expected_sha
            if not cached:
                part = out.with_suffix(out.suffix + ".part")
                h = hashlib.sha256()
                size = 0
                with requests.get(url, stream=True, timeout=(20, 90)) as r:
                    r.raise_for_status()
                    with part.open("wb") as f:
                        for b in r.iter_content(1024 * 1024):
                            size += len(b)
                            if size > expected_size:
                                raise ValueError("Download exceeds metadata size")
                            h.update(b)
                            f.write(b)
                sha = h.hexdigest()
                if size != expected_size or (expected_sha and sha != expected_sha):
                    raise ValueError("File size or upstream SHA256 mismatch")
                os.replace(part, out)
            result = {"repo": repo, "revision": revision, "file": name,
                      "path": str(out.relative_to(ROOT)), "bytes": expected_size,
                      "sha256": sha, "upstream_sha256": expected_sha,
                      "url": url, "status": "verified"}
            print(json.dumps({"file": result["path"], "bytes": expected_size,
                              "status": result["status"]}), flush=True)
            return result
        except Exception as exc:
            if attempt == 2:
                return {"repo": repo, "file": name, "status": "failed",
                        "error": type(exc).__name__ + ": " + str(exc).split("?")[0]}
            time.sleep(2 ** attempt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["primary", "klear", "supplement"], default="primary")
    args = parser.parse_args()
    selections = {
        "primary": SELECTIONS,
        "klear": {"Kwai-Klear/SWE-smith-mini_swe_agent_plus-trajectories-66k": "data/"},
        "supplement": {"R2E-Gym/R2EGym-SFT-Trajectories": "data/", "SWE-bench/SWE-bench_Verified": "data/"},
    }[args.profile]
    items = []
    for repo, prefix in selections.items():
        meta = json.loads((ROOT / "data/metadata" / (repo.replace("/", "__") + ".json")).read_text())
        for info in meta["siblings"]:
            name = info["rfilename"]
            if (name.startswith(prefix) and name.endswith(".parquet")) or name in {"README.md", "LICENSE", "LICENSE.md"}:
                items.append((repo, meta["sha"], info))
    total = sum(info["size"] for _, _, info in items)
    if total > 2_500_000_000:
        raise RuntimeError(f"Initial download budget exceeded: {total}")
    print(json.dumps({"planned_files": len(items), "planned_bytes": total}), flush=True)
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        for result in pool.map(fetch, items):
            results.append(result)
            filename = "download_manifest.json" if args.profile == "primary" else f"{args.profile}_download_manifest.json"
            manifest = ROOT / "data/metadata" / filename
            temp = manifest.with_suffix(".tmp")
            temp.write_text(json.dumps({"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                        "planned_bytes": total, "files": results}, indent=2) + "\n")
            os.replace(temp, manifest)
    if any(r["status"] != "verified" for r in results):
        raise SystemExit("Some files failed; inspect manifest and rerun.")


if __name__ == "__main__":
    main()
