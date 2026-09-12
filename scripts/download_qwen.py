"""Pinned official snapshot, bounded streaming, HTTP Range resume, upstream hashes.

Run after sourcing scripts/env.sh. No tensor materialization or remote code.
"""
import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
import threading
from urllib.parse import quote

import requests

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "runtime/metadata/qwen-model.json"
REVISION = "fc05daec18b0a78c049392ed2e771dde82bdf654"
DEST = ROOT / "models/Qwen3.5-27B" / REVISION
MANIFEST = ROOT / "runtime/metadata/qwen-download-manifest.json"
NETWORK_SLOTS = threading.BoundedSemaphore(22)
# Overridden by --repo/--revision/--meta/--dest/--manifest for other pinned snapshots.
CONFIG = {"repo": "Qwen/Qwen3.5-27B", "revision": REVISION, "dest": DEST, "manifest": MANIFEST}


def checksums(path):
    sha = hashlib.sha256()
    git = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            sha.update(block)
            git.update(block)
    return sha.hexdigest(), git.hexdigest()


def verify(path, item):
    if not path.is_file() or path.stat().st_size != item["size"]:
        return None
    sha, git = checksums(path)
    expected = item.get("lfs", {}).get("sha256")
    if (sha != expected) if expected else (git != item["blobId"]):
        return None
    return sha


def download_range(url, path, base, length):
    """Resume one contiguous segment; only verified complete files are promoted."""
    request_url = url
    failures = 0
    with requests.Session() as session:
        while not path.exists() or path.stat().st_size < length:
            local_offset = path.stat().st_size if path.exists() else 0
            remote_offset = base + local_offset
            end = min(remote_offset + 8 * 1024 * 1024, base + length) - 1
            try:
                with NETWORK_SLOTS, session.get(request_url,
                                 params={"attempt": time.time_ns()} if request_url == url else None,
                                 headers={"Range": f"bytes={remote_offset}-{end}"},
                                 stream=True, timeout=(8, 8)) as response:
                    if response.status_code in (401, 403):
                        request_url = url
                    response.raise_for_status()
                    if response.status_code != 206 or not response.headers.get(
                            'Content-Range', '').startswith(f'bytes {remote_offset}-'):
                        raise ValueError('Server did not honor the exact requested range')
                    request_url = response.url
                    with path.open('ab') as output:
                        for block in response.iter_content(256 * 1024):
                            local_offset += len(block)
                            if local_offset > length:
                                raise ValueError('Range exceeded pinned segment size')
                            output.write(block)
                failures = 0
            except Exception as exc:
                failures += 1
                if failures >= 30:
                    raise RuntimeError(f'{path.name}: 30 consecutive failed range requests') from exc
                time.sleep(1)
        assert path.stat().st_size == length


def download_two_segments(url, part, size, base=0, split_tail=True):
    """Resume fixed segments and their merges, at most 22 connections overall."""
    split = size // 2
    tail = part.with_name(part.name + '.tail')
    existing = part.stat().st_size if part.exists() else 0
    if existing > split and not tail.exists():
        download_range(url, part, base, size)
        return
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        jobs = []
        if existing < split:
            jobs.append(pool.submit(download_range, url, part, base, split))
        if split_tail:
            jobs.append(pool.submit(download_two_segments, url, tail, size - split, base + split, False))
        else:
            jobs.append(pool.submit(download_range, url, tail, base + split, size - split))
        for job in jobs:
            job.result()
    # A stopped merge resumes from the already appended prefix.
    with part.open('ab') as output, tail.open('rb') as source:
        source.seek(part.stat().st_size - split)
        for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
            output.write(block)
        output.flush()
        os.fsync(output.fileno())
    if not split_tail:
        tail.unlink()


def fetch(item, verify_only):
    name = item["rfilename"]
    path = CONFIG["dest"] / name
    path.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/{CONFIG['repo']}/resolve/{CONFIG['revision']}/{quote(name, safe='/')}"
    request_url = url
    session = requests.Session()
    sha = verify(path, item)
    if sha is None and verify_only:
        raise ValueError(f"Missing or invalid: {name}")
    part = path.with_name(path.name + ".part")
    if sha is None and item['size'] > 1_000_000_000:
        download_two_segments(url, part, item['size'])
    for attempt in range(500):
        if sha:
            break
        try:
            offset = part.stat().st_size if part.exists() else 0
            if offset > item["size"]:
                part.unlink()
                offset = 0
            while offset < item["size"]:
                end = min(offset + 8 * 1024 * 1024, item["size"]) - 1
                headers = {"Range": f"bytes={offset}-{end}"}
                # Unique query avoids reusing expired signed redirects at a proxy.
                with session.get(request_url, params={"download": "true", "attempt": time.time_ns()} if request_url == url else None,
                                  headers=headers, stream=True, timeout=(8, 8)) as r:
                    if r.status_code in (401, 403):
                        request_url = url
                    r.raise_for_status()
                    request_url = r.url
                    if r.status_code == 206:
                        if not r.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                            raise ValueError("Invalid resume Content-Range")
                    elif r.status_code == 200:
                        offset = 0
                    else:
                        raise ValueError(f"Unexpected HTTP status {r.status_code}")
                    with part.open("ab" if offset else "wb") as f:
                        for block in r.iter_content(256 * 1024):
                            offset += len(block)
                            if offset > item["size"]:
                                raise ValueError("Response exceeds pinned size")
                            f.write(block)
                        f.flush()
            sha = verify(part, item)
            if sha is None:
                if part.stat().st_size >= item["size"]:
                    part.unlink()
                raise ValueError("Incomplete file or upstream checksum mismatch")
            os.replace(part, path)
        except Exception as exc:
            print(json.dumps({"file": name, "attempt": attempt + 1, "error": str(exc).split('?')[0]}), flush=True)
            if attempt == 499:
                raise
            time.sleep(min(2 ** attempt, 2))
    tail = part.with_name(part.name + '.tail')
    if tail.exists():
        tail.unlink()
    result = {"file": name, "path": str(path.relative_to(ROOT)), "bytes": item["size"],
              "sha256": sha, "upstream_sha256": item.get("lfs", {}).get("sha256"),
              "upstream_git_blob": item["blobId"], "url": url, "status": "verified"}
    print(json.dumps({"file": name, "bytes": item["size"], "status": "verified"}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--repo", default=CONFIG["repo"])
    parser.add_argument("--revision", default=CONFIG["revision"])
    parser.add_argument("--meta", default=str(META))
    parser.add_argument("--dest", default=str(DEST))
    parser.add_argument("--manifest", default=str(MANIFEST))
    args = parser.parse_args()
    CONFIG.update(repo=args.repo, revision=args.revision, dest=Path(args.dest).resolve(),
                  manifest=Path(args.manifest).resolve())
    meta = json.loads(Path(args.meta).read_text())
    assert meta["sha"] == args.revision
    CONFIG["dest"].mkdir(parents=True, exist_ok=True)
    lock_name = CONFIG["manifest"].stem + ".lock"
    with (ROOT / "runtime/metadata" / lock_name).open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        results = []
        print(json.dumps({"revision": args.revision, "files": len(meta["siblings"]),
                          "planned_bytes": sum(x["size"] for x in meta["siblings"])}), flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=11) as pool:
            futures = [pool.submit(fetch, item, args.verify_only) for item in sorted(meta["siblings"], key=lambda x: x["size"])]
            for f in concurrent.futures.as_completed(futures):
                results.append(f.result())
                doc = {"repo": args.repo, "revision": args.revision,
                       "verified_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                       "complete": len(results) == len(meta["siblings"]),
                       "total_bytes": sum(x["bytes"] for x in results),
                       "files": sorted(results, key=lambda x: x["file"])}
                temp = CONFIG["manifest"].with_suffix(".tmp")
                temp.write_text(json.dumps(doc, indent=2) + "\n")
                os.replace(temp, CONFIG["manifest"])


if __name__ == "__main__":
    main()
