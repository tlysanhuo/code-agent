"""Download locked wheels using small resumable HTTP ranges; no installation."""
import concurrent.futures
import hashlib
import fcntl
import json
import os
from pathlib import Path
import time
from urllib.parse import unquote, urlparse
try:
    import tomllib
except ImportError:
    from pip._vendor import tomli as tomllib
import requests

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "runtime/wheelhouse"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def fetch(w):
    name = unquote(Path(urlparse(w['url']).path).name)
    out = DEST / name
    expected = w['hashes']['sha256']
    if out.exists() and digest(out) == expected:
        return name
    part = out.with_name(out.name + '.part')
    size = w['size']
    retries = 0
    session = requests.Session()
    while not part.exists() or part.stat().st_size < size:
        offset = part.stat().st_size if part.exists() else 0
        end = min(offset + 8 * 1024 * 1024, size) - 1
        try:
            with session.get(w['url'], headers={'Range': f'bytes={offset}-{end}'},
                              stream=True, timeout=(8, 8)) as r:
                r.raise_for_status()
                if r.status_code == 206:
                    if not r.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                        raise ValueError('Incorrect Content-Range')
                elif r.status_code == 200:
                    offset = 0
                else:
                    raise ValueError('Unexpected status')
                with part.open('ab' if offset else 'wb') as f:
                    for b in r.iter_content(1024 * 1024):
                        offset += len(b)
                        if offset > size:
                            raise ValueError('Incorrect response size')
                        f.write(b)
            retries = 0
        except Exception as e:
            retries += 1
            print(name, 'retry', retries, type(e).__name__, flush=True)
            if retries >= 12:
                raise
            time.sleep(2)
    if part.stat().st_size != size or digest(part) != expected:
        raise ValueError(f'{name}: checksum mismatch')
    os.replace(part, out)
    print(name, 'verified', size, flush=True)
    return name


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    wheels = []
    lock = tomllib.loads((ROOT / 'configs/pylock.runtime.toml').read_text())
    for package in lock['packages']:
        candidates = package.get('wheels', [])
        if not candidates:
            raise ValueError(f"Expected platform-specific lock: {package['name']}")
        wheels.append(candidates[0])
    for meta in ['dsh-runtime-pypi.json', 'dsh-sdk-pypi.json']:
        data = json.loads((ROOT / 'runtime/metadata' / meta).read_text())
        item = next(w for w in data['urls'] if w['filename'].endswith('.whl') and
                    ('manylinux_2_28_x86_64' in w['filename'] or 'none-any' in w['filename']))
        wheels.append({'url': item['url'], 'size': item['size'], 'hashes': item['digests']})
    for w in wheels:
        if 'size' not in w:
            r = requests.head(w['url'], allow_redirects=True, timeout=30)
            r.raise_for_status()
            w['size'] = int(r.headers['Content-Length'])
    print('Wheels:', len(wheels), 'bytes:', sum(x['size'] for x in wheels), flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(fetch, sorted(wheels, key=lambda x: x['size'])))


if __name__ == '__main__':
    with (ROOT / 'runtime/metadata/wheel-download.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        main()
