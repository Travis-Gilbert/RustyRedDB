// Builds the wasm module and its wasm-bindgen glue into ./pkg.
//
// Needs the wasm32-unknown-unknown target and wasm-bindgen-cli at the version
// the workspace pins (see crates/rustyred-log/Cargo.toml).
import { execFileSync } from 'node:child_process'
import { existsSync, rmSync, statSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const here = dirname(fileURLToPath(import.meta.url))
const pkg = join(here, '..', 'pkg')
const workspace = join(here, '..', '..', '..', '..')
const run = (command, args) => execFileSync(command, args, { cwd: workspace, stdio: 'inherit' })

run('cargo', [
  'build', '--locked', '--release', '-p', 'rustyred-wasm',
  '--target', 'wasm32-unknown-unknown',
])
const metadata = JSON.parse(
  execFileSync('cargo', ['metadata', '--format-version', '1', '--no-deps'], {
    cwd: workspace,
    maxBuffer: 1 << 28,
  }).toString(),
)
const wasm = join(
  metadata.target_directory, 'wasm32-unknown-unknown', 'release', 'rustyred_wasm.wasm',
)
if (existsSync(pkg)) rmSync(pkg, { recursive: true })
run('wasm-bindgen', [wasm, '--target', 'web', '--out-dir', pkg])
console.log(`pkg/rustyred_wasm_bg.wasm: ${statSync(join(pkg, 'rustyred_wasm_bg.wasm')).size} bytes`)
