#!/usr/bin/env python3
"""Measure a fresh release build and actual parts-catalog create/reopen runs."""
import argparse
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--target-dir', type=Path, required=True, help='New empty directory, distinct from other build lanes')
args = parser.parse_args()
root = Path(__file__).resolve().parent.parent
output = args.output.resolve()
target = args.target_dir.resolve()
if target.exists() and any(target.iterdir()):
    raise SystemExit('A cold target-directory measurement requires an empty target directory')
output.mkdir(parents=True, exist_ok=False)
start_revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
start_dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=root))
env = dict(os.environ, CARGO_TARGET_DIR=str(target), CARGO_INCREMENTAL='0')
command = ['cargo', '+1.96.1', 'build', '--locked', '--release', '-p', 'rustyred-cli', '-p', 'parts-catalog']
start = time.perf_counter()
with (output / 'build.log').open('w') as log:
    subprocess.run(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
seconds = time.perf_counter() - start
peak_rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
# getrusage reports bytes on Darwin and KiB on Linux.
peak_rss_bytes = peak_rss if platform.system() == 'Darwin' else peak_rss * 1024
runs = []
for run in range(5):
    db = output / f'parts-{run}.rr'
    for operation in ['create_seed_query', 'reopen_query']:
        start = time.perf_counter()
        result = subprocess.check_output([str(target / 'release/parts-catalog'), str(db)], cwd=output, text=True)
        elapsed = time.perf_counter() - start
        data = json.loads(result)
        if not data.get('ok') or len(data['rows']) != 2:
            raise RuntimeError('parts-catalog did not return both relationships')
        runs.append({'operation': operation, 'seconds': elapsed, 'database_bytes': db.stat().st_size, 'rows': len(data['rows'])})
data = {'revision_at_build_start': start_revision, 'dirty_at_build_start': start_dirty,
        'revision_at_completion': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
        'dirty_at_completion': bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=root)),
        'host': platform.platform(), 'rustc': subprocess.check_output(['rustc', '+1.96.1', '-Vv'], cwd=root, text=True).strip(),
        'command': command, 'target_initially_empty': True, 'cargo_incremental': '0', 'build_jobs': env.get('CARGO_BUILD_JOBS'),
        'rustc_wrapper': env.get('RUSTC_WRAPPER'), 'cold_target_build_seconds': seconds,
        'maximum_child_rss_bytes': peak_rss_bytes,
        'memory_method': 'getrusage RUSAGE_CHILDREN ru_maxrss: largest individual child high-water mark, not summed concurrent build memory',
        'cli_binary_bytes': (target / 'release/rustyred').stat().st_size,
        'runs': runs, 'limits': 'Cargo registry/toolchain caches remain warm; no throughput or comparative performance claim; tiny example includes process start and real fsyncs.'}
(output / 'measurements.json').write_text(json.dumps(data, indent=2) + '\n')
print(json.dumps(data, indent=2))
