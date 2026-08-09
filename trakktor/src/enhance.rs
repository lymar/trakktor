//! `enhance`: clean up a speech recording.
//!
//! A thin driver over `trakktor_core::enhance`: resolve the model (downloading
//! and converting it on first use), load it on the chosen runtime and device,
//! run the recording through it, and write the result in the format the output
//! path asks for.

use std::{path::PathBuf, time::Instant};

use trakktor_core::{
    audio::encode::Format,
    enhance::{
        EnhanceModel, EnhanceOptions, EnhanceProgress, Enhanced, Precision,
        gtcrn, mpsenet,
        resemble::{self, EnhancerSettings, Method},
        unipase,
    },
};

use crate::{
    asr::progress,
    cli::{
        DeviceArg, EnhanceGtcrnArgs, EnhanceMpsenetArgs,
        EnhanceResembleDenoiseArgs, EnhanceResembleEnhanceArgs,
        EnhanceUnipaseArgs, PrecisionArg, RuntimeArg, SolverArg,
    },
    error::CliError,
    output,
};

/// The verb the live line reports under.
const VERB: &str = "enhancing";

/// Runs `enhance unipase`.
pub(crate) fn run_unipase(
    args: &EnhanceUnipaseArgs,
    models_dir: &std::path::Path,
    json: bool,
    pretty: bool,
) -> Result<(), CliError> {
    let precision = match args.precision {
        PrecisionArg::F16 => Precision::F16,
        PrecisionArg::F32 => Precision::F32,
    };
    let metal = matches!(args.device, DeviceArg::Metal);
    let burn = matches!(args.runtime, RuntimeArg::Burn);

    // Where the result goes is settled before anything expensive starts: this
    // is hours of work on a long recording, and finding out at the end that
    // the directory cannot be created would throw all of it away.
    let path = output_path(&args.audio, args.output.as_ref());
    prepare_output(&path)?;

    let resolved = resolve(models_dir, &args.model)?;
    let label = resolved
        .name
        .clone()
        .unwrap_or_else(|| resolved.dir.display().to_string());

    eprintln!("loading the model...");
    let mut model = load(&resolved, burn, metal, precision)?;

    let options = EnhanceOptions {
        sample_rate: args.sample_rate,
        plc: !args.no_plc,
    };

    let started = Instant::now();
    let mut reporter = progress::live_reporter_for(VERB, started);
    let enhanced = unipase::enhance_file(
        &args.audio,
        model.as_mut(),
        &options,
        &mut |p| {
            let p: EnhanceProgress = p;
            reporter(p.done_seconds, Some(p.total_seconds));
        },
    )?;
    progress::finish_line_for(VERB, started, enhanced.duration());

    write(&enhanced, &path, args.bitrate.as_deref())?;
    output::print_enhance(
        &enhanced,
        &path,
        format_name(&path),
        "unipase",
        &label,
        if burn { "burn" } else { "candle" },
        if metal { "metal" } else { "cpu" },
        options.plc,
        json,
        pretty,
    );
    Ok(())
}

/// Resolves the model, announcing the size of a cold install before it starts.
fn resolve(
    models_dir: &std::path::Path,
    model: &str,
) -> Result<unipase::ResolvedModel, CliError> {
    let cold = !models_dir
        .join("enhance")
        .join("unipase")
        .join("model.safetensors")
        .is_file();
    if cold && !std::path::Path::new(model).is_dir() {
        eprintln!(
            "downloading and converting the pipeline ({:.2} GB) — this \
             happens once",
            unipase::download_size() as f64 / 1e9
        );
    }
    let mut reporter = progress::download_progress();
    Ok(unipase::resolve_model(models_dir, model, &mut reporter)?)
}

/// Loads the pipeline on the runtime and device asked for.
fn load(
    resolved: &unipase::ResolvedModel,
    burn: bool,
    metal: bool,
    precision: Precision,
) -> Result<Box<dyn EnhanceModel>, CliError> {
    if burn {
        #[cfg(feature = "burn")]
        {
            if metal {
                crate::burn_notice::announce_cold_gpu_start();
                return Ok(unipase::runtime_burn::load_metal(
                    &resolved.dir,
                    precision,
                )?);
            }
            return Ok(unipase::runtime_burn::load_cpu(
                &resolved.dir,
                precision,
            )?);
        }
        #[cfg(not(feature = "burn"))]
        return Err(trakktor_core::enhance::EnhanceError::InvalidOptions(
            "this build has no burn runtime; rebuild with the `burn` feature, \
             or use `--runtime candle`"
                .into(),
        )
        .into());
    }
    let device = unipase::runtime::device(metal)?;
    Ok(Box::new(unipase::CandleModel::load(
        &resolved.dir,
        device,
        precision,
    )?))
}

/// Where the enhanced recording goes: next to the input unless asked
/// otherwise, and never on top of the input itself.
fn output_path(audio: &std::path::Path, chosen: Option<&PathBuf>) -> PathBuf {
    if let Some(path) = chosen {
        return path.clone();
    }
    let stem = audio
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "audio".into());
    let dir = audio
        .parent()
        .map(std::path::Path::to_path_buf)
        .unwrap_or_default();
    dir.join(format!("{stem}.enhanced.wav"))
}

/// The container name the output path asks for.
fn format_name(path: &std::path::Path) -> &str {
    match path
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("flac") => "flac",
        Some("wav") => "wav",
        _ => "ffmpeg",
    }
}

/// Makes sure the result will have somewhere to land.
fn prepare_output(path: &std::path::Path) -> Result<(), CliError> {
    if let Some(parent) = path.parent() &&
        !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent).map_err(|e| {
            trakktor_core::enhance::EnhanceError::Io(format!(
                "creating {}: {e}",
                parent.display()
            ))
        })?;
    }
    Ok(())
}

/// Writes the result, through ffmpeg for anything the built-in encoders do not
/// cover.
fn write(
    enhanced: &Enhanced,
    path: &std::path::Path,
    bitrate: Option<&str>,
) -> Result<(), CliError> {
    match format_name(path) {
        "wav" => enhanced.write(path, Format::Wav)?,
        "flac" => enhanced.write(path, Format::Flac)?,
        _ => enhanced.write_ffmpeg(path, bitrate)?,
    }
    Ok(())
}

/// Runs `enhance mpsenet`.
pub(crate) fn run_mpsenet(
    args: &EnhanceMpsenetArgs,
    models_dir: &std::path::Path,
    json: bool,
    pretty: bool,
) -> Result<(), CliError> {
    let precision = match args.precision {
        PrecisionArg::F16 => Precision::F16,
        PrecisionArg::F32 => Precision::F32,
    };
    let metal = matches!(args.device, DeviceArg::Metal);
    let burn = matches!(args.runtime, RuntimeArg::Burn);

    let path = output_path(&args.audio, args.output.as_ref());
    prepare_output(&path)?;

    let cold = !mpsenet::model_dir(models_dir, &args.model)
        .join("model.safetensors")
        .is_file();
    if cold && !std::path::Path::new(&args.model).is_dir() {
        let size = mpsenet::download_size(&args.model);
        if size > 0 {
            eprintln!(
                "downloading the network ({} MB) — this happens once",
                size / 1_000_000
            );
        }
    }
    let mut reporter = progress::download_progress();
    let resolved =
        mpsenet::resolve_model(models_dir, &args.model, &mut reporter)?;
    let label = resolved
        .name
        .clone()
        .unwrap_or_else(|| resolved.dir.display().to_string());

    eprintln!("loading the model...");
    let mut model = load_mpsenet(&resolved, burn, metal, precision)?;

    // Nothing to conceal: this engine ends in a mask, and a mask times
    // silence is silence.
    let options = EnhanceOptions {
        sample_rate: args.sample_rate,
        plc: false,
    };

    let started = Instant::now();
    let mut reporter = progress::live_reporter_for(VERB, started);
    let enhanced = mpsenet::enhance_file(
        &args.audio,
        model.as_mut(),
        &options,
        &mut |p: EnhanceProgress| {
            reporter(p.done_seconds, Some(p.total_seconds));
        },
    )?;
    progress::finish_line_for(VERB, started, enhanced.duration());

    write(&enhanced, &path, args.bitrate.as_deref())?;
    output::print_enhance(
        &enhanced,
        &path,
        format_name(&path),
        "mpsenet",
        &label,
        if burn { "burn" } else { "candle" },
        if metal { "metal" } else { "cpu" },
        false,
        json,
        pretty,
    );
    Ok(())
}

/// Loads MP-SENet on the runtime and device asked for.
fn load_mpsenet(
    resolved: &mpsenet::ResolvedModel,
    burn: bool,
    metal: bool,
    precision: Precision,
) -> Result<Box<dyn EnhanceModel>, CliError> {
    if burn {
        #[cfg(feature = "burn")]
        {
            if metal {
                crate::burn_notice::announce_cold_gpu_start();
                return Ok(mpsenet::runtime_burn::load_metal(
                    &resolved.dir,
                    precision,
                )?);
            }
            return Ok(mpsenet::runtime_burn::load_cpu(
                &resolved.dir,
                precision,
            )?);
        }
        #[cfg(not(feature = "burn"))]
        return Err(trakktor_core::enhance::EnhanceError::InvalidOptions(
            "this build has no burn runtime; rebuild with the `burn` feature, \
             or use `--runtime candle`"
                .into(),
        )
        .into());
    }
    let device = mpsenet::runtime::device(metal)?;
    Ok(Box::new(mpsenet::CandleModel::load(
        &resolved.dir,
        device,
        precision,
    )?))
}

/// Runs `enhance resemble-denoise`.
pub(crate) fn run_resemble_denoise(
    args: &EnhanceResembleDenoiseArgs,
    models_dir: &std::path::Path,
    json: bool,
    pretty: bool,
) -> Result<(), CliError> {
    let precision = match args.precision {
        PrecisionArg::F16 => Precision::F16,
        PrecisionArg::F32 => Precision::F32,
    };
    let metal = matches!(args.device, DeviceArg::Metal);
    let burn = matches!(args.runtime, RuntimeArg::Burn);

    let path = output_path(&args.audio, args.output.as_ref());
    prepare_output(&path)?;

    let resolved = resolve_resemble(
        models_dir,
        &args.model,
        resemble::download::DENOISER_FILE,
    )?;
    let label = resolved
        .name
        .clone()
        .unwrap_or_else(|| resolved.dir.display().to_string());

    eprintln!("loading the model...");
    let mut model = load_resemble_denoiser(&resolved, burn, metal, precision)?;

    // Nothing to conceal: this engine ends in a mask.
    let options = EnhanceOptions {
        sample_rate: args.sample_rate,
        plc: false,
    };

    let started = Instant::now();
    let mut reporter = progress::live_reporter_for(VERB, started);
    let enhanced = resemble::enhance_file(
        &args.audio,
        model.as_mut(),
        &options,
        &mut |p: EnhanceProgress| {
            reporter(p.done_seconds, Some(p.total_seconds));
        },
    )?;
    progress::finish_line_for(VERB, started, enhanced.duration());

    write(&enhanced, &path, args.bitrate.as_deref())?;
    output::print_enhance(
        &enhanced,
        &path,
        format_name(&path),
        "resemble-denoise",
        &label,
        if burn { "burn" } else { "candle" },
        if metal { "metal" } else { "cpu" },
        false,
        json,
        pretty,
    );
    Ok(())
}

/// Runs `enhance resemble-enhance`.
pub(crate) fn run_resemble_enhance(
    args: &EnhanceResembleEnhanceArgs,
    models_dir: &std::path::Path,
    json: bool,
    pretty: bool,
) -> Result<(), CliError> {
    let precision = match args.precision {
        PrecisionArg::F16 => Precision::F16,
        PrecisionArg::F32 => Precision::F32,
    };
    let metal = matches!(args.device, DeviceArg::Metal);
    let burn = matches!(args.runtime, RuntimeArg::Burn);
    let settings = resemble_settings(args)?;

    let path = output_path(&args.audio, args.output.as_ref());
    prepare_output(&path)?;

    let resolved = resolve_resemble(
        models_dir,
        &args.model,
        resemble::download::ENHANCER_FILE,
    )?;
    let label = resolved
        .name
        .clone()
        .unwrap_or_else(|| resolved.dir.display().to_string());

    eprintln!("loading the model...");
    let mut model =
        load_resemble_enhancer(&resolved, burn, metal, precision, settings)?;

    let options = EnhanceOptions {
        sample_rate: args.sample_rate,
        plc: false,
    };

    let started = Instant::now();
    let mut reporter = progress::live_reporter_for(VERB, started);
    let enhanced = resemble::enhance_file(
        &args.audio,
        model.as_mut(),
        &options,
        &mut |p: EnhanceProgress| {
            reporter(p.done_seconds, Some(p.total_seconds));
        },
    )?;
    progress::finish_line_for(VERB, started, enhanced.duration());

    write(&enhanced, &path, args.bitrate.as_deref())?;
    output::print_enhance(
        &enhanced,
        &path,
        format_name(&path),
        "resemble-enhance",
        &label,
        if burn { "burn" } else { "candle" },
        if metal { "metal" } else { "cpu" },
        false,
        json,
        pretty,
    );
    Ok(())
}

/// Validates the generative engine's knobs and collects them.
fn resemble_settings(
    args: &EnhanceResembleEnhanceArgs,
) -> Result<EnhancerSettings, CliError> {
    let invalid = |message: String| {
        CliError::from(trakktor_core::enhance::EnhanceError::InvalidOptions(
            message,
        ))
    };
    if args.nfe == 0 || args.nfe > 128 {
        return Err(invalid(format!(
            "--nfe must be between 1 and 128, got {}",
            args.nfe
        )));
    }
    for (name, value) in [
        ("--temperature", args.temperature),
        ("--denoise", args.denoise),
    ] {
        if !(0.0..=1.0).contains(&value) {
            return Err(invalid(format!(
                "{name} must be between 0 and 1, got {value}"
            )));
        }
    }
    Ok(EnhancerSettings {
        nfe: args.nfe,
        method: match args.solver {
            SolverArg::Euler => Method::Euler,
            SolverArg::Midpoint => Method::Midpoint,
            SolverArg::Rk4 => Method::Rk4,
        },
        lambda: args.denoise,
        temperature: args.temperature,
        seed: args.seed,
    })
}

/// Resolves the shared checkpoint, announcing a cold install before it starts.
fn resolve_resemble(
    models_dir: &std::path::Path,
    model: &str,
    wanted: &str,
) -> Result<resemble::ResolvedModel, CliError> {
    let cold = !resemble::model_dir(models_dir).join(wanted).is_file();
    if cold && !std::path::Path::new(model).is_dir() {
        eprintln!(
            "downloading and converting the pipeline ({:.2} GB) — this \
             happens once, and covers both resemble engines",
            resemble::download_size() as f64 / 1e9
        );
    }
    let mut reporter = progress::download_progress();
    Ok(resemble::resolve_model(
        models_dir,
        model,
        wanted,
        &mut reporter,
    )?)
}

/// Loads the resemble denoiser on the runtime and device asked for.
fn load_resemble_denoiser(
    resolved: &resemble::ResolvedModel,
    burn: bool,
    metal: bool,
    precision: Precision,
) -> Result<Box<dyn EnhanceModel>, CliError> {
    if burn {
        #[cfg(feature = "burn")]
        {
            if metal {
                crate::burn_notice::announce_cold_gpu_start();
                return Ok(resemble::runtime_burn::load_denoiser_metal(
                    &resolved.dir,
                    precision,
                )?);
            }
            return Ok(resemble::runtime_burn::load_denoiser_cpu(
                &resolved.dir,
                precision,
            )?);
        }
        #[cfg(not(feature = "burn"))]
        return Err(no_burn());
    }
    let device = resemble::runtime::device(metal)?;
    Ok(Box::new(resemble::DenoiserModel::load(
        &resolved.dir,
        device,
        precision,
    )?))
}

/// Loads the resemble enhancer on the runtime and device asked for.
fn load_resemble_enhancer(
    resolved: &resemble::ResolvedModel,
    burn: bool,
    metal: bool,
    precision: Precision,
    settings: EnhancerSettings,
) -> Result<Box<dyn EnhanceModel>, CliError> {
    if burn {
        #[cfg(feature = "burn")]
        {
            if metal {
                crate::burn_notice::announce_cold_gpu_start();
                return Ok(resemble::runtime_burn::load_enhancer_metal(
                    &resolved.dir,
                    precision,
                    settings,
                )?);
            }
            return Ok(resemble::runtime_burn::load_enhancer_cpu(
                &resolved.dir,
                precision,
                settings,
            )?);
        }
        #[cfg(not(feature = "burn"))]
        return Err(no_burn());
    }
    let device = resemble::runtime::device(metal)?;
    Ok(Box::new(resemble::EnhancerModel::load(
        &resolved.dir,
        device,
        precision,
        settings,
    )?))
}

/// The failure a build without the burn runtime reports.
#[cfg(not(feature = "burn"))]
fn no_burn() -> CliError {
    trakktor_core::enhance::EnhanceError::InvalidOptions(
        "this build has no burn runtime; rebuild with the `burn` feature, or \
         use `--runtime candle`"
            .into(),
    )
    .into()
}

/// Runs `enhance gtcrn`.
pub(crate) fn run_gtcrn(
    args: &EnhanceGtcrnArgs,
    models_dir: &std::path::Path,
    json: bool,
    pretty: bool,
) -> Result<(), CliError> {
    let path = output_path(&args.audio, args.output.as_ref());
    prepare_output(&path)?;

    let cold = !models_dir
        .join("enhance")
        .join("gtcrn")
        .join("model.safetensors")
        .is_file();
    if cold && !std::path::Path::new(&args.model).is_dir() {
        eprintln!(
            "downloading the network ({} KB) — this happens once",
            gtcrn::download_size() / 1024
        );
    }
    let mut reporter = progress::download_progress();
    let resolved =
        gtcrn::resolve_model(models_dir, &args.model, &mut reporter)?;
    let label = resolved
        .name
        .clone()
        .unwrap_or_else(|| resolved.dir.display().to_string());

    // This engine streams: its chunks continue one another rather than
    // overlap, so the driver speaks to it directly instead of through the
    // one-window-at-a-time seam.
    let mut model = gtcrn::CandleModel::load(&resolved.dir)?;
    let options = EnhanceOptions {
        sample_rate: args.sample_rate,
        plc: false,
    };

    let started = Instant::now();
    let mut reporter = progress::live_reporter_for(VERB, started);
    let enhanced = gtcrn::enhance_file(
        &args.audio,
        &mut model,
        &options,
        &mut |p: EnhanceProgress| {
            reporter(p.done_seconds, Some(p.total_seconds));
        },
    )?;
    progress::finish_line_for(VERB, started, enhanced.duration());

    write(&enhanced, &path, args.bitrate.as_deref())?;
    output::print_enhance(
        &enhanced,
        &path,
        format_name(&path),
        "gtcrn",
        &label,
        "candle",
        "cpu",
        false,
        json,
        pretty,
    );
    Ok(())
}
