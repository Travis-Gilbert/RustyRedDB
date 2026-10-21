#!/usr/bin/env python3
# /// script
# dependencies = ["psutil==7.2.2"]
# ///
"""Measure an immutable cold-target build and real example create/reopen runs.

Run with `uv run scripts/measure_release.py ...`, or install the pinned psutil
version into an isolated venv. --allow-dirty produces development evidence only.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import time

import psutil


def source_state(root: Path) -> dict:
    names = subprocess.check_output(['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard'], cwd=root).split(b'\0')
    digest = hashlib.sha256()
    for name in sorted(set(names)):
        if not name:
            continue
        path = root / os.fsdecode(name)
        digest.update(name + b'\0')
        if path.is_symlink():
            digest.update(os.fsencode(os.readlink(path)))
        elif path.is_file():
            digest.update(path.read_bytes())
        else:
            digest.update(b'<absent>')
        digest.update(b'\0')
    return {'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
            'dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=root)),
            'source_sha256': digest.hexdigest()}


def sample_rss(process: psutil.Popen) -> int:
    try:
        processes = [process, *process.children(recursive=True)]
    except psutil.NoSuchProcess:
        return 0
    resident = 0
    for child in processes:
        try:
            resident += child.memory_info().rss
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            pass
    return resident


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--target-dir', type=Path, required=True, help='New empty directory, distinct from other build lanes')
parser.add_argument('--allow-dirty', action='store_true', help='Permit development-only measurements, never clean candidate attribution')
args = parser.parse_args()
root = Path(__file__).resolve().parent.parent
output = args.output.resolve()
target = args.target_dir.resolve()
if output.is_relative_to(root) or target.is_relative_to(root):
    raise SystemExit('Measurement output and target directories must be outside the source checkout')
if target.exists() and any(target.iterdir()):
    raise SystemExit('A cold target-directory measurement requires an empty target directory')
source_start = source_state(root)
if source_start['dirty'] and not args.allow_dirty:
    raise SystemExit('A candidate measurement requires clean committed source; use --allow-dirty for explicitly developmental evidence')
output.mkdir(parents=True, exist_ok=False)
env = dict(os.environ, CARGO_TARGET_DIR=str(target), CARGO_INCREMENTAL='0')
command = ['cargo', '+1.96.1', 'build', '--locked', '--release', '-p', 'rustyred-cli', '-p', 'parts-catalog']
start = time.perf_counter()
samples = 0
sampled_peak = 0
with (output / 'build.log').open('w') as log:
    process = psutil.Popen(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
    while process.poll() is None:
        sampled_peak = max(sampled_peak, sample_rss(process))
        samples += 1
        time.sleep(0.1)
    if process.wait() != 0:
        raise subprocess.CalledProcessError(process.returncode, command)
seconds = time.perf_counter() - start
peak_rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
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
source_end = source_state(root)
qualified = not source_start['dirty'] and not source_end['dirty'] and source_start == source_end
measurement = {'source_at_build_start': source_start, 'source_at_completion': source_end,
        'clean_candidate_attribution': qualified,
        'evidence_class': 'clean_candidate_measurement' if qualified else 'development_measurement',
        'host': platform.platform(), 'rustc': subprocess.check_output(['rustc', '+1.96.1', '-Vv'], cwd=root, text=True).strip(),
        'command': command, 'target_initially_empty': True, 'cargo_incremental': '0', 'build_jobs': env.get('CARGO_BUILD_JOBS'),
        'rustc_wrapper': env.get('RUSTC_WRAPPER'), 'cold_target_build_seconds': seconds,
        'maximum_child_rss_bytes': peak_rss_bytes, 'sampled_peak_process_tree_rss_bytes': sampled_peak,
        'memory_sampler': {'library': 'psutil ' + psutil.__version__, 'sample_interval_seconds': 0.1, 'samples': samples,
                           'method': 'Sum RSS of Cargo and its recursive live descendants. Shared pages may be counted more than once; brief peaks may be missed. ru_maxrss is separately the largest individual child high-water mark.'},
        'cli_binary_bytes': (target / 'release/rustyred').stat().st_size,
        'runs': runs, 'limits': 'Cargo registry/toolchain caches may remain warm; no throughput or comparative performance claim; tiny example includes process start and real fsyncs.'}
(output / 'measurements.json').write_text(json.dumps(measurement, indent=2) + '\n')
print(json.dumps(measurement, indent=2))
if not qualified and not args.allow_dirty:
    raise SystemExit('Source drifted during measurement; evidence saved as development-only, candidate attribution refused')
