# Extraction baseline

- Plan pin: `d6a7062c14c11c6838362bda307e267b2297c684`
- Extracted commit: `1acf012c3d2ffbc5f3e9a559fe9e282aa9ed46de`
- Toolchain used for the extract: rustc 1.96.1, from the source `rust-toolchain.toml`
- The plan pin is not an ancestor of the checkout that was extracted. This branch has continued storage work (OPFS commit semantics, bounded growth, history import) that is not in the pin, and the pin is not contained here. The extraction uses the committed tree at `1acf012c3d2ffbc5f3e9a559fe9e282aa9ed46de`, not uncommitted product files.
- Uncommitted Theorem working-tree changes were not copied.
- Private Theorem git history was not copied. This repository starts as a new history with this extraction record.
