#!/usr/bin/env python3
"""Opt-in post-publication crates.io consumer/CLI smoke; never uploads or tags."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tomllib

from package_rehearsal import executable_from_messages, network_denied


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', required=True, help='Actual published family version')
    parser.add_argument('--output', type=Path, required=True, help='New evidence directory outside this checkout')
    args = parser.parse_args()
    if os.environ.get('RUSTYRED_RUN_REGISTRY_SMOKE') != '1':
        raise SystemExit('NOT RUN: set RUSTYRED_RUN_REGISTRY_SMOKE=1 only after authorized publication is actually available')
    root = Path(__file__).resolve().parent.parent
    version = tomllib.loads((root / 'crates/rustyred/Cargo.toml').read_text())['package']['version']
    if args.version != version:
        raise SystemExit('Use the matching source version so the real consumer/example fixtures match the registry version')
    output = args.output.resolve()
    if output.is_relative_to(root):
        raise SystemExit('Registry smoke output must be outside this checkout')
    # A fresh Cargo home does not disable configuration discovered in ancestors.
    for ancestor in output.parents:
        for name in ['config', 'config.toml']:
            if (ancestor / '.cargo' / name).exists():
                raise SystemExit('Choose an output path without ancestor Cargo configuration')
    output.mkdir(parents=True, exist_ok=False)
    consumer = output / 'consumer'
    (consumer / 'src/bin').mkdir(parents=True)
    (consumer / 'Cargo.toml').write_text(
        '[package]\nname = "rustyred-registry-consumer"\nversion = "0.0.0"\nedition = "2021"\n'
        '[dependencies]\nrustyred = ' + json.dumps('=' + version) + '\nserde_json = "1"\n')
    shutil.copyfile(root / 'tests/external-consumer/src/lib.rs', consumer / 'src/lib.rs')
    shutil.copyfile(root / 'examples/parts-catalog/src/main.rs', consumer / 'src/bin/parts-catalog.rs')
    allowed = {'PATH', 'TMPDIR', 'TEMP', 'TMP', 'SYSTEMROOT', 'SDKROOT', 'MACOSX_DEPLOYMENT_TARGET'}
    env = {key: value for key, value in os.environ.items() if key in allowed}
    env.update(HOME=str(output / 'home'), CARGO_HOME=str(output / 'cargo-home'),
               XDG_CONFIG_HOME=str(output / 'home/.config'),
               RUSTUP_HOME=os.environ.get('RUSTUP_HOME', str(Path.home() / '.rustup')),
               CARGO_TARGET_DIR=str(output / 'target'), CARGO_INCREMENTAL='0',
               CARGO_BUILD_JOBS='3', RUST_TEST_THREADS='1')
    Path(env['HOME']).mkdir()
    cargo = ['cargo', '+1.96.1']
    records = []

    def invoke(command: list[str], name: str) -> str:
        result = subprocess.run(command, cwd=consumer, env=env, text=True, capture_output=True)
        (output / (name + '.stdout')).write_text(result.stdout)
        (output / (name + '.stderr')).write_text(result.stderr)
        records.append({'name': name, 'command': command, 'exit_code': result.returncode})
        (output / 'commands.json').write_text(json.dumps(records, indent=2) + '\n')
        if result.returncode:
            raise SystemExit(f'{name} failed with exit {result.returncode}; retained stdout/stderr in evidence directory')
        return result.stdout

    invoke(cargo + ['generate-lockfile'], 'resolve-public-registry')
    invoke(cargo + ['fetch', '--locked'], 'fetch-public-registry')
    metadata = json.loads(invoke(cargo + ['metadata', '--locked', '--offline', '--format-version', '1'], 'registry-metadata'))
    family = []
    for package in metadata['packages']:
        if package['name'] == 'rustyred-registry-consumer':
            continue
        if not (package['source'] or '').startswith('registry+'):
            raise SystemExit('Consumer resolved a non-registry dependency')
        if package['name'].startswith('rustyred'):
            if package['version'] != version:
                raise SystemExit('Consumer resolved a different RustyRed family version')
            family.append({'name': package['name'], 'version': package['version'], 'source': package['source']})
    messages = invoke(cargo + ['test', '--locked', '--offline', '--no-run', '--message-format=json'], 'consumer-build')
    binaries = executable_from_messages(messages, name='rustyred_registry_consumer', test=True)
    for index, binary in enumerate(binaries):
        invoke(network_denied([binary]), f'consumer-runtime-{index}')
    messages = invoke(cargo + ['build', '--locked', '--offline', '--bin', 'parts-catalog', '--message-format=json'], 'example-build')
    example = executable_from_messages(messages, name='parts-catalog', test=False)[0]
    for index in range(2):
        invoke(network_denied([example, str(output / 'parts.rr')]), f'example-runtime-{index}')
    invoke(cargo + ['install', '--locked', '--registry', 'crates-io', '--version', '=' + version,
                    '--root', str(output / 'installed'), 'rustyred-cli'], 'registry-cli-install')
    cli = output / 'installed/bin/rustyred'
    db = output / 'cli.rr'
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    operations = [('help', ['--help']), ('version', ['--version']), ('create', ['create', str(db)]),
                  ('inspect', ['inspect', str(db), '--json']), ('verify', ['verify', str(db), '--json']),
                  ('query', ['query', str(db), '--cypher', 'MATCH (n) RETURN n'])]
    for name, arguments in operations:
        before = digest(db) if db.exists() else None
        text = invoke(network_denied([str(cli), *arguments]), 'cli-' + name)
        if name == 'version' and text.strip() != 'rustyred ' + version:
            raise SystemExit('Installed CLI version differs from requested published version')
        if name not in ['help', 'version']:
            value = json.loads(text)
            if value['ok'] is not True:
                raise SystemExit(f'CLI {name} did not report success')
        if name in ['inspect', 'verify', 'query'] and before != digest(db):
            raise SystemExit(f'CLI {name} changed database bytes')
    hashes = {str(path.relative_to(output)): digest(path) for path in output.iterdir() if path.is_file()}
    receipt = {'result': 'PASS', 'evidence_class': 'actual_post_publication_registry_install',
               'version': version, 'registry_family': family, 'consumer_patch_or_path_dependencies': False,
               'fresh_home_and_cargo_home': True, 'runtime_network_denial': network_denied(['<executable>'])[:-1],
               'installed_cli_sha256': digest(cli), 'consumer_binary_sha256': {binary: digest(Path(binary)) for binary in binaries},
               'example_binary_sha256': digest(Path(example)),
               'fixture_sha256': {str(path.relative_to(root)): digest(path) for path in [root / 'tests/external-consumer/src/lib.rs', root / 'examples/parts-catalog/src/main.rs']}, 'consumer_lock_sha256': digest(consumer / 'Cargo.lock'),
               'evidence_file_sha256': hashes, 'publication': 'none; install/read-only registry operations only',
               'limits': 'PASS requires actual public registry availability. No archive/Git substitution. Runtime uses platform OS network denial; downloads/builds require public registry access. Registry dependency resolution is retained in the actual installation log.'}
    (output / 'evidence.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print('Actual registry smoke passed; evidence:', output / 'evidence.json')


if __name__ == '__main__':
    main()
