#!/usr/bin/env python3
"""Capture Cargo's complete resolved closure and each declared dependency edge."""
import argparse
import json
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, default=Path('docs/extraction/dependencies.json'))
args = parser.parse_args()
root = Path(__file__).resolve().parent.parent
metadata = json.loads(subprocess.check_output(['cargo', '+1.96.1', 'metadata', '--locked', '--all-features', '--format-version', '1'], cwd=root, text=True))
workspace_ids = set(metadata['workspace_members'])

def relative(path):
    try:
        return str(Path(path).relative_to(root))
    except ValueError:
        return None

packages = []
for package in sorted(metadata['packages'], key=lambda p: (p['name'], p['version'])):
    local = package['id'] in workspace_ids
    package_files = subprocess.check_output(['cargo', '+1.96.1', 'package', '--list', '--exclude-lockfile', '--allow-dirty', '-p', package['name']], cwd=root, text=True).splitlines() if local else None
    packages.append({
        'name': package['name'], 'version': package['version'], 'source': package['source'] or 'standalone-workspace',
        'manifest': relative(package['manifest_path']) if local else None,
        'package_files': package_files, 'license': package['license'], 'rust_version': package['rust_version'], 'features': package['features'],
        'targets': [{key: target[key] for key in ['name', 'kind', 'crate_types', 'required-features'] if key in target} for target in package['targets']],
        'dependencies': [{key: value for key, value in dep.items() if key != 'path'} | ({'path': relative(dep['path'])} if dep.get('path') else {}) for dep in package['dependencies']],
    })
output = {'schema': 2, 'command': 'cargo +1.96.1 metadata --locked --all-features --format-version 1',
          'note': 'All declared normal, build, dev, optional and target-conditioned edges; Cargo resolution is not evidence that optional native features execute.',
          'toolchain': subprocess.check_output(['rustc', '+1.96.1', '-Vv'], cwd=root, text=True).strip(),
          'packages': packages, 'resolve': metadata['resolve'],
          'source_resource_markers': sorted(subprocess.run(['rg', '--line-number', '(include_str!|include_bytes!|#\\[path|env!|option_env!)', 'crates', 'vendor'], cwd=root, text=True, capture_output=True, check=False).stdout.splitlines())}
# Cargo local package IDs contain machine paths. Replace only IDs with stable name/version keys.
id_map = {p['id']: p['name'] + '@' + p['version'] + ('#registry' if p['source'] else '#workspace') for p in metadata['packages']}
output['resolve'] = {'root': id_map.get(metadata['resolve']['root']), 'nodes': [dict(node, id=id_map[node['id']], dependencies=[id_map[item] for item in node['dependencies']], deps=[dict(dep, pkg=id_map[dep['pkg']]) for dep in node['deps']]) for node in metadata['resolve']['nodes']]}
args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + '\n')
print(f'Captured {len(packages)} packages at {args.output}')
