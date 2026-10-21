#!/usr/bin/env python3
"""Record exact extraction provenance without copying private source/history."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source-repository', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parent.parent
plan = 'd6a7062c14c11c6838362bda307e267b2297c684'
extracted = '1acf012c3d2ffbc5f3e9a559fe9e282aa9ed46de'
source_map = json.loads((root / 'docs/extraction/source-map.json').read_text())
selections = [item['source'] for item in source_map['items']
              if item.get('source', '').startswith('rustyredcore_THG/')
              and item.get('destination', '').startswith(('crates/', 'vendor/'))
              and '{' not in item['source']]

def archived_hashes(revision):
    process = subprocess.Popen(['git', 'archive', revision, *selections], cwd=args.source_repository, stdout=subprocess.PIPE)
    hashes = {}
    with tarfile.open(fileobj=process.stdout, mode='r|') as archive:
        for member in archive:
            if member.isfile() and Path(member.name).suffix in {'.rs', '.pest', '.json', '.term', '.h', '.sh'}:
                hashes[member.name] = hashlib.sha256(archive.extractfile(member).read()).hexdigest()
    if process.wait() != 0:
        raise RuntimeError('Cannot read pinned source archive')
    return hashes

baselines = {'plan_pin': archived_hashes(plan), 'extraction_source': archived_hashes(extracted)}
rows = []
for item in source_map['items']:
    old, new = item.get('source', ''), item.get('destination', '')
    if not old.startswith('rustyredcore_THG/') or not new.startswith(('crates/', 'vendor/')) or '{' in old:
        continue
    for path in sorted((root / new).rglob('*')):
        if not path.is_file() or path.suffix not in {'.rs', '.pest', '.json', '.term', '.h', '.sh'}:
            continue
        relative = str(path.relative_to(root / new))
        source = old + '/' + relative
        hashes = {label: values.get(source) for label, values in baselines.items()}
        candidate = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append({'source': source, 'candidate': str(path.relative_to(root)), **hashes, 'candidate_sha256': candidate,
                     'matches_plan': candidate == hashes['plan_pin'], 'matches_extraction_source': candidate == hashes['extraction_source']})
record = {'plan_pin': plan, 'extraction_source': extracted, 'method': 'SHA-256 of exact git blobs and candidate bytes; hashes are provenance, runtime tests remain required', 'files': rows}
args.output.write_text(json.dumps(record, indent=2) + '\n')
print(f'Recorded {len(rows)} source/resource comparisons; no private source or history exported')
