//! A stderr heads-up before a burn GPU run, and suppression of the harmless
//! autotune panic noise that comes with it.
//!
//! burn compiles and autotunes its GPU kernels **the first time it sees a
//! given kernel shape**, which makes that run several times slower; the tuning
//! result is written to a persistent cache and later runs of the same shape
//! start fast. The cache is keyed by kernel shape, not by machine — a model
//! whose shapes have not been tuned yet re-autotunes even when the cache holds
//! another model's kernels — so every burn GPU run gets a heads-up: a stronger
//! one when the cache is entirely empty (the whole tuning pass is ahead), a
//! softer one otherwise (only this model's uncached shapes tune). This keeps
//! the one-time slowness expected rather than alarming, regardless of which
//! engine warmed the cache first.
//!
//! While autotuning, the kernel library (cubecl) benchmarks candidate GPU
//! kernels, and for some matmul shapes a candidate tiling divides by zero and
//! panics. The autotune harness catches that panic (`catch_unwind`), discards
//! the candidate, and picks a working kernel — the result is correct — but the
//! panic message still reaches stderr and looks alarming. We install a panic
//! hook that drops exactly those autotune-internal panics and passes every
//! other panic through unchanged (see [`suppress_autotune_panic_noise`]).
//!
//! The cache check mirrors how cubecl resolves its default cache root: the
//! `target/` directory of the nearest enclosing Cargo project, searching
//! upward from the working directory, or the user cache directory under
//! `cubecl/` outside any project. A non-default cache configuration is not
//! followed; the notice then errs on the stronger wording, which only costs
//! phrasing, not correctness.

use std::{path::PathBuf, sync::Once};

/// Prepares for a burn GPU run: silences the cubecl autotune panic noise
/// (once) and prints a one-line notice that kernels not yet cached are
/// compiled and autotuned on this run (stronger wording when the whole cache
/// is empty). Call before loading a burn model on a GPU device — every GPU
/// run, not only the first: a model whose kernel shapes are uncached
/// autotunes even when the cache holds another model's. The CPU backend does
/// not autotune, so this is not called there.
pub(crate) fn announce_cold_gpu_start() {
    suppress_autotune_panic_noise();
    if autotune_cache_exists() {
        eprintln!(
            "burn GPU: any kernels not yet cached for this model are compiled \
             and autotuned now (slower); already-tuned kernels run at full \
             speed, and this model's are cached after this run"
        );
    } else {
        eprintln!(
            "first burn GPU run on this machine: compiling and autotuning \
             kernels — this run is several times slower; the result is \
             cached, later runs start fast"
        );
    }
}

/// Installs, once, a panic hook that drops the panics cubecl emits while
/// benchmarking candidate GPU kernels during autotuning. Those panics come
/// from source files reached only inside the autotune harness's
/// `catch_unwind`, so silencing their message changes no control flow — the
/// candidate is still discarded and a working kernel chosen. Every other
/// panic (including any real cubecl compute failure, which is not
/// autotune-internal) is delegated to the previous hook unchanged, so genuine
/// crashes still surface.
fn suppress_autotune_panic_noise() {
    static INSTALL: Once = Once::new();
    INSTALL.call_once(|| {
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            if let Some(location) = info.location() {
                let file = location.file();
                let autotune_internal = file.contains("cubek-matmul") ||
                    (file.contains("cubecl-runtime") &&
                        file.contains("tune"));
                if autotune_internal {
                    return;
                }
            }
            previous(info);
        }));
    });
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
