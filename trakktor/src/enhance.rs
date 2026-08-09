//! `enhance`: clean up a speech recording.
//!
//! A thin driver over `trakktor_core::enhance`: resolve the model (downloading
//! and converting it on first use), load it on the chosen runtime and device,
//! run the recording through it, and write the result in the format the output
//! path asks for.

use std::{path::PathBuf, time::Instant};

use trakktor_core::{
    audio::encode::Format,
    enhance::unipase::{
        self, EnhanceOptions, Precision, ResolvedModel, download_size,
        enhance_file,
    },
};

use crate::{
    asr::progress,
    cli::{DeviceArg, EnhanceUnipaseArgs, PrecisionArg, RuntimeArg},
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
    let path = output_path(args);
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
    let enhanced =
        enhance_file(&args.audio, model.as_mut(), &options, &mut |p| {
            let p: unipase::EnhanceProgress = p;
            reporter(p.done_seconds, Some(p.total_seconds));
        })?;
    progress::finish_line_for(VERB, started, enhanced.duration());

    write(&enhanced, &path, args.bitrate.as_deref())?;
    output::print_enhance(
        &enhanced,
        &path,
        format_name(&path),
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
) -> Result<ResolvedModel, CliError> {
    let cold = !models_dir
        .join("enhance")
        .join("unipase")
        .join("model.safetensors")
        .is_file();
    if cold && !std::path::Path::new(model).is_dir() {
        eprintln!(
            "downloading and converting the pipeline ({:.2} GB) — this \
             happens once",
            download_size() as f64 / 1e9
        );
    }
    let mut reporter = progress::download_progress();
    Ok(unipase::resolve_model(models_dir, model, &mut reporter)?)
}

/// Loads the pipeline on the runtime and device asked for.
fn load(
    resolved: &ResolvedModel,
    burn: bool,
    metal: bool,
    precision: Precision,
) -> Result<Box<dyn unipase::EnhanceModel>, CliError> {
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
        return Err(unipase::UnipaseError::InvalidOptions(
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
fn output_path(args: &EnhanceUnipaseArgs) -> PathBuf {
    if let Some(path) = &args.output {
        return path.clone();
    }
    let stem = args
        .audio
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "audio".into());
    let dir = args
        .audio
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
            unipase::UnipaseError::Io(format!(
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
    enhanced: &unipase::Enhanced,
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
