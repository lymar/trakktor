//! A stderr heads-up before a cold burn GPU run.
//!
//! The first time the burn runtime executes on a GPU, its kernels are
//! compiled and autotuned, which makes that run several times slower than
//! usual; the tuning result is written to a persistent cache and every later
//! run starts fast. Detecting the cold case up front and saying so keeps the
//! one-time slowness expected rather than alarming.
//!
//! The check mirrors how the kernel library (cubecl) resolves its default
//! cache root: the `target/` directory of the nearest enclosing Cargo
//! project, searching upward from the working directory, or the user cache
//! directory under `cubecl/` outside any project. A non-default cache
//! configuration is not followed; the notice then errs on announcing a warm
//! run as cold, which only costs one spurious line on stderr.

use std::path::PathBuf;

/// Prints a one-line stderr notice when no burn autotune cache exists yet —
/// the GPU run ahead compiles and autotunes kernels. Call before loading a
/// burn model on a GPU device (the CPU backend does not autotune).
pub(crate) fn announce_cold_gpu_start() {
    if !autotune_cache_exists() {
        eprintln!(
            "first burn GPU run on this machine: compiling and autotuning \
             kernels — this run is several times slower; the result is \
             cached, later runs start fast"
        );
    }
}

/// Whether the autotune cache directory exists and is non-empty.
fn autotune_cache_exists() -> bool {
    let Some(root) = cache_root() else {
        return false;
    };
    std::fs::read_dir(root.join("autotune"))
        .is_ok_and(|mut entries| entries.next().is_some())
}

/// The default cubecl cache root: `<nearest Cargo project>/target`, or the
/// user cache directory joined with `cubecl` when running outside a project.
fn cache_root() -> Option<PathBuf> {
    let mut dir = std::env::current_dir().ok()?;
    loop {
        if dir.join("Cargo.toml").exists() {
            return Some(dir.join("target"));
        }
        if !dir.pop() {
            break;
        }
    }
    Some(user_cache_dir()?.join("cubecl"))
}

/// The per-user cache directory (`~/Library/Caches` on macOS, XDG cache on
/// other Unix).
fn user_cache_dir() -> Option<PathBuf> {
    if cfg!(target_os = "macos") {
        return Some(
            PathBuf::from(std::env::var_os("HOME")?).join("Library/Caches"),
        );
    }
    match std::env::var_os("XDG_CACHE_HOME") {
        Some(xdg) if !xdg.is_empty() => Some(PathBuf::from(xdg)),
        _ => Some(PathBuf::from(std::env::var_os("HOME")?).join(".cache")),
    }
}
