"""Read-only progress, counting overlapping resumed segments only once."""
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
meta = json.loads((ROOT / 'runtime/metadata/qwen-model.json').read_text())
manifest = json.loads((ROOT / 'runtime/metadata/qwen-download-manifest.json').read_text())
present = 0
for item in meta['siblings']:
    path = ROOT / 'models/Qwen3.5-27B' / meta['sha'] / item['rfilename']
    split = item['size'] // 2
    tail_split = split + (item['size'] - split) // 2
    candidates = [(path, 0), (path.with_name(path.name + '.part'), 0),
                  (path.with_name(path.name + '.part.tail'), split),
                  (path.with_name(path.name + '.part.tail.tail'), tail_split)]
    end = 0
    for begin, stop in sorted((offset, min(item['size'], offset + p.stat().st_size))
                              for p, offset in candidates if p.is_file()):
        present += max(0, stop - max(begin, end))
        end = max(end, stop)
total = sum(item['size'] for item in meta['siblings'])
print(json.dumps({'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                  'bytes_present': present, 'total_bytes': total,
                  'percent_present': round(100 * present / total, 1),
                  'verified_files': len(manifest['files']),
                  'verified_bytes': manifest['total_bytes'],
                  'complete': manifest['complete']}, indent=2))
