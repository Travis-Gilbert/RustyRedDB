// egglog-add-primitive2.0.0 needs full host-side syn expression parsing.
// Keep activation explicit under Cargo resolver2 instead of inheriting it
// from unrelated workspace procedural macros. No runtime code is generated.
fn main() {
    println!("cargo:rerun-if-changed=build.rs");
}
