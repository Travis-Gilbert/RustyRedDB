#!/usr/bin/env python3
"""Prove the boundary refuses private code even in an unreferenced module."""
from pathlib import Path
import subprocess
import tempfile

script = Path(__file__).resolve().with_name('dependency_boundary.py')
with tempfile.TemporaryDirectory(prefix='rustyred-boundary-') as temporary:
    root = Path(temporary)
    (root / 'src').mkdir()
    (root / 'Cargo.toml').write_text('[package]\nname = "boundary-fixture"\nversion = "0.0.0"\nedition = "2021"\n')
    (root / 'src/lib.rs').write_text('// Standalone generic fixture.\n')
    command = ['python3', str(script), '--manifest', str(root / 'Cargo.toml')]
    subprocess.run(command, check=True)
    (root / 'src/unused.rs').write_text('use theorem_blocks::Block;\n')
    result = subprocess.run(command, text=True, capture_output=True)
    assert result.returncode == 1 and 'private source import/resource' in result.stderr, result.stderr
    (root / 'src/unused.rs').unlink()
    (root / 'Cargo.toml').write_text('[package]\nname = "theorem-harness-fixture"\nversion = "0.0.0"\nedition = "2021"\n')
    result = subprocess.run(command, text=True, capture_output=True)
    assert result.returncode == 1 and 'name contains theorem-harness' in result.stderr, result.stderr
print('Dependency boundary accepted generic source and refused private manifest/unreferenced source fixtures')
