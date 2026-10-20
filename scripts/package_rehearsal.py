#!/usr/bin/env python3
"""Exercise normalized Cargo archives through Cargo's offline directory source.

No registry is uploaded or emulated. Cargo vendors external dependencies; each
unpublished package is unpacked from its actual .crate archive into that source.
The consumer resolves normal registry versions without paths or [patch].
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import tomllib

from dependency_boundary import source_problems


def run(*args: str, cwd: Path, env: dict[str, str] | None = None) -> str:
    print('+ ' + ' '.join(args), flush=True)
    result = subprocess.run(args, cwd=cwd, env=env, text=True, check=True, stdout=subprocess.PIPE)
    print(result.stdout, end='', flush=True)
    return result.stdout


def network_denied(command: list[str]) -> list[str]:
    if platform.system() == 'Darwin':
        return ['/usr/bin/sandbox-exec', '-p', '(version 1)(allow default)(deny network*)', *command]
    if platform.system() == 'Linux':
        # Only finished executables run as root. Cargo/build outputs remain
        # owned by the ordinary CI user. No credentials are passed to runtime.
        return ['sudo', '-n', 'unshare', '--net', '--', *command]
    raise RuntimeError('No qualified runtime network sandbox for this host')


def executable_from_messages(output: str, *, name: str, test: bool) -> list[str]:
    executables = []
    for line in output.splitlines():
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (message.get('reason') == 'compiler-artifact' and message.get('executable')
                and message.get('target', {}).get('name') == name
                and message.get('profile', {}).get('test') == test):
            executables.append(message['executable'])
    if not executables:
        raise RuntimeError(f'Cargo did not report an executable for {name} (test={test})')
    return sorted(set(executables))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New evidence directory outside the checkout')
    parser.add_argument('--allow-dirty', action='store_true', help='Development rehearsal only; release artifacts refuse dirty state')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    output = args.output.resolve()
    if output.is_relative_to(root):
        raise SystemExit('The rehearsal output must be outside the source checkout')
    output.mkdir(parents=True, exist_ok=False)
    cargo = ['cargo', '+1.96.1']
    metadata = json.loads(subprocess.check_output(cargo + ['metadata', '--format-version', '1', '--no-deps'], cwd=root, text=True))
    packages = [p for p in metadata['packages'] if p.get('publish') != []]
    excluded = [p['name'] for p in metadata['packages'] if p.get('publish') == []]
    command = cargo + ['package', '--workspace', '--locked', '--no-verify']
    for name in excluded:
        command += ['--exclude', name]
    if args.allow_dirty:
        command += ['--allow-dirty']
    run(*command, cwd=root)
    target = Path(os.environ.get('CARGO_TARGET_DIR', str(root / 'target')))
    source = output / 'directory-source'
    run(*cargo, 'vendor', '--locked', '--versioned-dirs', str(source), cwd=root)
    archives = output / 'archives'
    archives.mkdir()
    evidence = {'toolchain': run('rustc', '+1.96.1', '-Vv', cwd=root).strip(),
                'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
                'dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=root)),
                'route': 'Cargo directory source from cargo vendor plus normalized .crate archives', 'packages': []}
    for package in packages:
        label = package['name'] + '-' + package['version']
        archive = target / 'package' / (label + '.crate')
        shutil.copy2(archive, archives)
        with tarfile.open(archive, 'r:gz') as bundle:
            # Cargo creates these archives, but still reject traversal/link entries.
            for entry in bundle.getmembers():
                path = Path(entry.name)
                if path.is_absolute() or '..' in path.parts or entry.issym() or entry.islnk():
                    raise RuntimeError(f'Unsafe archive member: {entry.name}')
            bundle.extractall(source, filter='data')
        directory = source / label
        problems = source_problems(directory)
        if problems:
            raise RuntimeError('Packaged source boundary failed: ' + '; '.join(problems))
        normalized = tomllib.loads((directory / 'Cargo.toml').read_text())
        if 'workspace' in normalized or 'patch' in normalized:
            raise RuntimeError(f'Unnormalized manifest: {label}')
        for key in ['dependencies', 'build-dependencies', 'dev-dependencies']:
            tables = [normalized.get(key, {})] + [t.get(key, {}) for t in normalized.get('target', {}).values()]
            for table in tables:
                for name, dep in table.items():
                    if isinstance(dep, dict) and ('path' in dep or 'git' in dep):
                        raise RuntimeError(f'Non-registry dependency in {label}: {name}')
        files = {str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in sorted(directory.rglob('*')) if path.is_file()}
        checksum = hashlib.sha256(archive.read_bytes()).hexdigest()
        (directory / '.cargo-checksum.json').write_text(json.dumps({'files': files, 'package': checksum}) + '\n')
        evidence['packages'].append({'name': package['name'], 'version': package['version'], 'sha256': checksum, 'files': sorted(files)})
    consumer = output / 'consumer'
    (consumer / '.cargo').mkdir(parents=True)
    (consumer / 'src').mkdir()
    (consumer / '.cargo/config.toml').write_text('[source.crates-io]\nreplace-with = "rehearsal"\n[source.rehearsal]\ndirectory = ' + json.dumps(str(source)) + '\n')
    version = next(p['version'] for p in packages if p['name'] == 'rustyred')
    (consumer / 'Cargo.toml').write_text('[package]\nname = "rustyred-packaged-consumer"\nversion = "0.0.0"\nedition = "2021"\n[dependencies]\nrustyred = "=' + version + '"\nserde_json = "1"\n')
    shutil.copy2(root / 'tests/external-consumer/src/lib.rs', consumer / 'src/lib.rs')
    (consumer / 'src/bin').mkdir()
    shutil.copy2(root / 'examples/parts-catalog/src/main.rs', consumer / 'src/bin/parts-catalog.rs')
    # Keep the already installed public toolchain, but remove credentials from the
    # child process and give Cargo a fresh home with no maintainer configuration.
    allowed_env = {'PATH', 'TMPDIR', 'TEMP', 'TMP', 'SYSTEMROOT', 'SDKROOT',
                   'MACOSX_DEPLOYMENT_TARGET', 'CARGO_TARGET_DIR', 'CARGO_INCREMENTAL',
                   'CARGO_BUILD_JOBS', 'CARGO_PROFILE_DEV_DEBUG', 'CARGO_PROFILE_TEST_DEBUG'}
    env = {key: value for key, value in os.environ.items() if key in allowed_env}
    env['RUSTUP_HOME'] = os.environ.get('RUSTUP_HOME', str(Path.home() / '.rustup'))
    env['HOME'] = str(output / 'runtime-home')
    Path(env['HOME']).mkdir()
    env['XDG_CONFIG_HOME'] = str(output / 'runtime-home/.config')
    env['CARGO_HOME'] = str(output / 'cargo-home')
    env['CARGO_NET_OFFLINE'] = 'true'
    # Ordinary outside consumers do not run the engine's private timing fixtures.
    # Keep Cargo's default test/dev optimization so tests, example and install
    # share the same dependency build instead of forcing a second codegen profile.
    evidence['consumer_profiles'] = {'test': 'Cargo default opt-level 0', 'build_install': 'Cargo default opt-level 0'}
    for name in sorted(package['name'] for package in packages if package['name'] != 'rustyred-graphblas'):
        run(*cargo, 'check', '--offline', '--all-targets', '--manifest-path', str(source / (name + '-' + version) / 'Cargo.toml'), cwd=consumer, env=env)
    (output / 'archive-targets-checked.json').write_text(json.dumps({
        'revision': evidence['revision'], 'dirty': evidence['dirty'],
        'phase': 'standalone_archive_all_targets', 'result': 'PASS',
        'command': 'cargo +1.96.1 check --offline --all-targets',
        'checked': sorted(package['name'] for package in packages if package['name'] != 'rustyred-graphblas'),
        'excluded': {'rustyred-graphblas': 'separate native prerequisites and runtime qualification required'},
        'packages': evidence['packages'],
    }, indent=2) + '\n')
    messages = run(*cargo, 'test', '--offline', '--no-run', '--message-format=json', cwd=consumer, env=env)
    for binary in executable_from_messages(messages, name='rustyred_packaged_consumer', test=True):
        run(*network_denied([binary]), cwd=consumer, env=env)
    evidence['runtime_network_denial'] = network_denied(['<executable>'])[:-1]
    messages = run(*cargo, 'build', '--offline', '--bin', 'parts-catalog', '--message-format=json', cwd=consumer, env=env)
    example = executable_from_messages(messages, name='parts-catalog', test=False)[0]
    for _ in range(2):
        run(*network_denied([example, str(output / 'parts.rr')]), cwd=consumer, env=env)
    tree = run(*cargo, 'tree', '--offline', '--prefix', 'none', cwd=consumer, env=env)
    (output / 'consumer-tree.txt').write_text(tree)
    run(*cargo, 'install', '--config', str(consumer / '.cargo/config.toml'), '--debug', '--offline', '--version', '=' + version, '--root', str(output / 'installed'), 'rustyred-cli', cwd=consumer, env=env)
    cli = output / 'installed/bin/rustyred'
    db = output / 'cli.rr'
    for arguments in [('create', str(db)), ('inspect', str(db)), ('verify', str(db)), ('query', str(db), '--cypher', 'MATCH (n) RETURN n')]:
        run(*network_denied([str(cli), *arguments]), cwd=consumer, env=env)
    (output / 'evidence.json').write_text(json.dumps(evidence, indent=2) + '\n')
    print(f'Package rehearsal passed; evidence: {output / "evidence.json"}')


if __name__ == '__main__':
    main()
