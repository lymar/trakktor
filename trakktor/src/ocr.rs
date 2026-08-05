//! `ocr paddle` and `ocr vl`: flag mapping, model resolution, and the page
//! loop.

use std::{
    io::{IsTerminal, Write},
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use trakktor_core::ocr::{
    OcrError, Page, Quad,
    figures::{self, Figure},
    layout::{
        Region,
        detect::{self, Detector},
        post,
    },
    markdown,
    paddle::{
        crop::{self, Crop},
        db::Params,
        image::Page as RawPage,
        model,
        pipeline::{Device, Engine, Options},
    },
    vl,
};

use crate::{
    cli::{
        OcrFormatArg, OcrLayoutArgs, OcrPaddleArgs, OcrTaskArg, OcrVlArgs,
        RuntimeArg,
    },
    error::CliError,
    output::OcrModels,
};

/// Runs one OCR pass: resolve the models, then read every page in turn.
pub(crate) fn run_paddle(
    args: &OcrPaddleArgs,
    model_dir: &Path,
    json: bool,
    pretty: bool,
) -> Result<(), CliError> {
    if args.lang.eq_ignore_ascii_case("list") {
        crate::output::print_ocr_languages(
            &model::LANGUAGES
                .iter()
                .map(|(code, name)| (*code, *name))
                .collect::<Vec<_>>(),
            json,
            pretty,
        );
        return Ok(());
    }

    // A flag that silently changed nothing would be worse than a refusal.
    if matches!(args.runtime, RuntimeArg::Burn) {
        return Err(OcrError::InvalidOptions(
            "the classic engine has no burn runtime; use `--runtime candle`, \
             or `ocr vl` for the engine that has one"
                .into(),
        )
        .into());
    }

    if args.pages.is_empty() {
        return Err(OcrError::NoPages.into());
    }

    // Fail on a missing page before downloading anything.
    for page in &args.pages {
        if !page.is_file() {
            return Err(OcrError::PageNotFound {
                path: page.display().to_string(),
            }
            .into());
        }
    }

    let options = Options {
        language: args.lang.clone(),
        detection: args
            .det_model
            .clone()
            .unwrap_or_else(|| model::DEFAULT_DETECTION.to_string()),
        recognition: args.rec_model.clone(),
        orientation: args.textline_orientation,
        limit_side_len: args.limit_side_len,
        params: Params {
            thresh: args.thresh,
            box_thresh: args.box_thresh,
            unclip_ratio: args.unclip_ratio,
            ..Params::default()
        },
        drop_score: args.drop_score,
        // Illustrations are only worth looking for when there is somewhere to
        // put them: a Markdown run writing to a file.
        figures: matches!(args.format, OcrFormatArg::Md) && args.out.is_some(),
    };

    let engine = Engine::load(
        model_dir,
        device(args.device),
        options,
        &mut crate::asr::progress::download_progress(),
    )?;
    let marker = (!args.no_layout)
        .then(|| {
            load_layout(
                model_dir,
                args.layout_model.as_deref(),
                device(args.device),
                RuntimeArg::Candle,
            )
        })
        .transpose()?;

    let mut pages: Vec<Page> = Vec::with_capacity(args.pages.len());
    let mut figures: Vec<Vec<Quad>> = Vec::with_capacity(args.pages.len());
    let mut regions: Vec<Vec<Region>> = Vec::with_capacity(args.pages.len());
    for (index, path) in args.pages.iter().enumerate() {
        let read = engine.read_file(path, index + 1)?;
        if let Some(dir) = &args.crops {
            write_crops(dir, index + 1, &read.crops)?;
        }
        let marked = match &marker {
            None => Vec::new(),
            Some(marker) => marker.detect_file(path)?,
        };
        // With a layout model the pictures are its own: it says what a picture
        // is, where the raster path only says where ink stands that no text
        // box covers.
        let found: Vec<Figure> = if marked.is_empty() {
            read.figures.clone()
        } else {
            marked
                .iter()
                .filter(|r| r.label.is_pictorial())
                .map(figure_of)
                .collect()
        };
        if !found.is_empty() {
            let raster = RawPage::load(path)?;
            write_figures(args.out.as_deref(), index + 1, &raster, &found)?;
        }
        figures.push(found.iter().map(Figure::quad).collect());
        regions.push(marked);
        pages.push(read.page);
    }

    let markdown = matches!(args.format, OcrFormatArg::Md);
    let (detection, recognition) = engine.models();
    crate::output::print_ocr(
        &pages,
        &figures,
        &regions,
        Some(&args.lang),
        OcrModels {
            detection,
            recognition,
            layout: marker.as_ref().map(Detector::model),
        },
        markdown,
        args.out.as_deref(),
        json,
        pretty,
    )
    .map_err(CliError::from)
}

/// Resolves and loads the layout model.
fn load_layout(
    model_dir: &Path,
    model: Option<&str>,
    device: Device,
    runtime: RuntimeArg,
) -> Result<Detector, CliError> {
    let options = detect::Options {
        model: model.map(str::to_string).unwrap_or_else(|| {
            trakktor_core::ocr::layout::model::DEFAULT_MODEL.to_string()
        }),
        post: post::Settings::default(),
    };
    let runtime = match runtime {
        RuntimeArg::Candle => detect::Runtime::Candle,
        RuntimeArg::Burn => detect::Runtime::Burn,
    };
    Detector::load(
        model_dir,
        device,
        runtime,
        options,
        &mut crate::asr::progress::download_progress(),
    )
    .map_err(CliError::from)
}

/// A labelled picture as the figure writer wants it.
fn figure_of(region: &Region) -> Figure {
    let (x0, y0, x1, y1) = region.bounds();
    Figure {
        x: x0.max(0.0) as u32,
        y: y0.max(0.0) as u32,
        width: (x1 - x0).max(1.0) as u32,
        height: (y1 - y0).max(1.0) as u32,
    }
}

/// Runs one `ocr vl` pass.
///
/// The shape is the classic engine's — resolve, then read page by page — with
/// one difference that matters to whoever is watching: a page here is tens of
/// seconds, not one, so the block counter goes to stderr as it goes.
pub(crate) fn run_vl(
    args: &OcrVlArgs,
    model_dir: &Path,
    json: bool,
    pretty: bool,
) -> Result<(), CliError> {
    if args.pages.is_empty() {
        return Err(OcrError::NoPages.into());
    }
    for page in &args.pages {
        if !page.is_file() {
            return Err(OcrError::PageNotFound {
                path: page.display().to_string(),
            }
            .into());
        }
    }

    let options = vl::Options {
        model: args
            .model
            .clone()
            .unwrap_or_else(|| vl::model::DEFAULT_MODEL.to_string()),
        detection: args
            .det_model
            .clone()
            .unwrap_or_else(|| vl::pipeline::DEFAULT_DETECTION.to_string()),
        limit_side_len: args.limit_side_len,
        params: Params {
            thresh: args.thresh,
            box_thresh: args.box_thresh,
            unclip_ratio: args.unclip_ratio,
            ..Params::default()
        },
        task: match args.task {
            OcrTaskArg::Ocr => vl::Task::Ocr,
            OcrTaskArg::Table => vl::Task::Table,
            OcrTaskArg::Formula => vl::Task::Formula,
            OcrTaskArg::Chart => vl::Task::Chart,
        },
        limits: vl::Limits {
            max_tokens: args.max_tokens,
            ..vl::Limits::default()
        },
        blocks: vl::blocks::Settings {
            gap: args.block_gap,
            overlap: args.block_overlap,
            height: args.block_height,
            row_gap: args.block_row_gap,
            max_lines: args.block_lines,
            padding: args.block_padding,
            ..vl::blocks::Settings::default()
        },
        whole_page: args.whole_page,
        drop_score: args.drop_score,
        figures: matches!(args.format, OcrFormatArg::Md) && args.out.is_some(),
    };

    let runtime = match args.runtime {
        RuntimeArg::Candle => vl::pipeline::Runtime::Candle,
        RuntimeArg::Burn => vl::pipeline::Runtime::Burn,
    };
    if runtime == vl::pipeline::Runtime::Burn &&
        matches!(args.device, crate::cli::DeviceArg::Metal)
    {
        announce_burn_gpu();
    }

    let mut engine = vl::Engine::load(
        model_dir,
        device(args.device),
        runtime,
        options,
        &mut crate::asr::progress::download_progress(),
    )?;

    let marker = (!args.no_layout)
        .then(|| {
            load_layout(
                model_dir,
                args.layout_model.as_deref(),
                device(args.device),
                args.runtime,
            )
        })
        .transpose()?;

    let mut pages: Vec<Page> = Vec::with_capacity(args.pages.len());
    let mut figures: Vec<Vec<Quad>> = Vec::with_capacity(args.pages.len());
    let mut regions: Vec<Vec<Region>> = Vec::with_capacity(args.pages.len());
    let started = Instant::now();
    for (index, path) in args.pages.iter().enumerate() {
        let number = index + 1;
        let mut report = block_progress(started, number, args.pages.len());
        let marked = match &marker {
            None => Vec::new(),
            Some(marker) => marker.detect_file(path)?,
        };
        let read =
            engine.read_file_marked(path, number, &marked, &mut report)?;
        if let Some(dir) = &args.crops {
            write_blocks(dir, number, &read)?;
        }
        let found: Vec<Figure> = if marked.is_empty() {
            read.figures.clone()
        } else {
            marked
                .iter()
                .filter(|r| r.label.is_pictorial())
                .map(figure_of)
                .collect()
        };
        if !found.is_empty() {
            let raster = RawPage::load(path)?;
            write_figures(args.out.as_deref(), number, &raster, &found)?;
        }
        figures.push(found.iter().map(Figure::quad).collect());
        regions.push(marked);
        pages.push(read.page);
    }
    finish_progress();

    let (detection, recognition) = engine.models();
    crate::output::print_ocr(
        &pages,
        &figures,
        &regions,
        None,
        OcrModels {
            detection,
            recognition,
            layout: marker.as_ref().map(Detector::model),
        },
        matches!(args.format, OcrFormatArg::Md),
        args.out.as_deref(),
        json,
        pretty,
    )
    .map_err(CliError::from)
}

/// Runs `ocr layout`: the markup stage on its own.
pub(crate) fn run_layout(
    args: &OcrLayoutArgs,
    model_dir: &Path,
    json: bool,
    pretty: bool,
) -> Result<(), CliError> {
    if args.pages.is_empty() {
        return Err(OcrError::NoPages.into());
    }
    for page in &args.pages {
        if !page.is_file() {
            return Err(OcrError::PageNotFound {
                path: page.display().to_string(),
            }
            .into());
        }
    }
    if matches!(args.runtime, RuntimeArg::Burn) &&
        matches!(args.device, crate::cli::DeviceArg::Metal)
    {
        announce_burn_gpu();
    }

    let mut options = detect::Options {
        model: args.model.clone().unwrap_or_else(|| {
            trakktor_core::ocr::layout::model::DEFAULT_MODEL.to_string()
        }),
        post: post::Settings::default(),
    };
    options.post.threshold = args.threshold;
    let runtime = match args.runtime {
        RuntimeArg::Candle => detect::Runtime::Candle,
        RuntimeArg::Burn => detect::Runtime::Burn,
    };
    let marker = Detector::load(
        model_dir,
        device(args.device),
        runtime,
        options,
        &mut crate::asr::progress::download_progress(),
    )?;

    let mut pages = Vec::with_capacity(args.pages.len());
    for (index, path) in args.pages.iter().enumerate() {
        let raster = RawPage::load(path)?;
        let regions = marker.detect(&raster)?;
        if let Some(dir) = &args.crops {
            write_regions(dir, index + 1, &raster, &regions)?;
        }
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| path.display().to_string());
        pages.push((index + 1, name, raster.width, raster.height, regions));
    }

    crate::output::print_ocr_layout(&pages, marker.model(), json, pretty);
    Ok(())
}

/// Writes one page's block pictures.
fn write_regions(
    dir: &Path,
    page: usize,
    raster: &RawPage,
    regions: &[Region],
) -> Result<(), CliError> {
    std::fs::create_dir_all(dir).map_err(|source| OcrError::Write {
        path: dir.display().to_string(),
        source,
    })?;
    for (index, region) in regions.iter().enumerate() {
        let png = figures::crop_png(
            &raster.bgr,
            raster.width as usize,
            raster.height as usize,
            &figure_of(region),
        )?;
        let path = dir.join(format!(
            "p{page:03}-b{index:02}-{}.png",
            region.label.name()
        ));
        std::fs::write(&path, png).map_err(|source| OcrError::Write {
            path: path.display().to_string(),
            source,
        })?;
    }
    Ok(())
}

/// The device the run asks for. Whether this build can serve it is the
/// engine's to say, so that the answer is the same however trakktor is
/// embedded.
fn device(chosen: crate::cli::DeviceArg) -> Device {
    match chosen {
        crate::cli::DeviceArg::Cpu => Device::Cpu,
        crate::cli::DeviceArg::Metal => Device::Metal,
    }
}

/// The kernel compile/autotune heads-up before a burn GPU run (builds with
/// the `burn` feature).
#[cfg(feature = "burn")]
fn announce_burn_gpu() { crate::burn_notice::announce_cold_gpu_start(); }

/// Without the `burn` feature the load fails with its own message; there is
/// nothing to announce.
#[cfg(not(feature = "burn"))]
fn announce_burn_gpu() {}

/// The live stderr line of a `ocr vl` run.
///
/// A generative reader spends tens of seconds on a page with nothing to show
/// for it until the page is done, which is indistinguishable from being stuck.
/// The line says which block of which page is being read and how long the run
/// has been going; it is rewritten in place on a terminal and, off one,
/// printed only when the block advances.
fn block_progress(
    started: Instant,
    page: usize,
    pages: usize,
) -> impl FnMut(usize, usize) {
    let interactive = std::io::stderr().is_terminal();
    let mut last: Option<Instant> = None;
    move |block: usize, blocks: usize| {
        let now = Instant::now();
        let too_soon = last.is_some_and(|at| {
            now.duration_since(at) < Duration::from_millis(250)
        });
        // The first block of a page always prints: it is the only sign that
        // the page changed.
        if too_soon && block > 0 {
            return;
        }
        last = Some(now);
        let elapsed = started.elapsed().as_secs();
        let where_ = if pages > 1 {
            format!("page {page}/{pages}, ")
        } else {
            String::new()
        };
        let line = format!(
            "reading {where_}block {}/{blocks} · {}:{:02}",
            block + 1,
            elapsed / 60,
            elapsed % 60
        );
        let mut stderr = std::io::stderr();
        if interactive {
            let _ = write!(stderr, "\r\u{1b}[2K{line}");
        } else {
            let _ = writeln!(stderr, "{line}");
        }
        let _ = stderr.flush();
    }
}

/// Clears the live line so the result does not land on top of it.
fn finish_progress() {
    if std::io::stderr().is_terminal() {
        let mut stderr = std::io::stderr();
        let _ = write!(stderr, "\r\u{1b}[2K");
        let _ = stderr.flush();
    }
}

/// Writes one page's block pictures.
fn write_blocks(
    dir: &Path,
    page: usize,
    read: &vl::Read,
) -> Result<(), CliError> {
    std::fs::create_dir_all(dir).map_err(|source| OcrError::Write {
        path: dir.display().to_string(),
        source,
    })?;
    for index in 0..read.crops() {
        let path: PathBuf = dir.join(format!("p{page:03}-b{index:04}.png"));
        std::fs::write(&path, read.crop_png(index)?).map_err(|source| {
            OcrError::Write {
                path: path.display().to_string(),
                source,
            }
        })?;
    }
    Ok(())
}

/// The directory illustration crops go in, next to the Markdown they are
/// linked from. The name is also what the Markdown spells, so the two agree by
/// construction.
pub(crate) const IMAGE_DIR: &str = "imgs";

/// Writes one page's illustrations next to the Markdown file.
fn write_figures(
    out: Option<&Path>,
    page: usize,
    raster: &RawPage,
    found: &[Figure],
) -> Result<(), CliError> {
    let Some(out) = out else { return Ok(()) };
    let dir = out
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(IMAGE_DIR);
    std::fs::create_dir_all(&dir).map_err(|source| OcrError::Write {
        path: dir.display().to_string(),
        source,
    })?;
    for (index, figure) in found.iter().enumerate() {
        let png = figures::crop_png(
            &raster.bgr,
            raster.width as usize,
            raster.height as usize,
            figure,
        )?;
        let path = dir.join(markdown::figure_name(page, index));
        std::fs::write(&path, png).map_err(|source| OcrError::Write {
            path: path.display().to_string(),
            source,
        })?;
    }
    Ok(())
}

/// Writes the straightened line crops of one page.
fn write_crops(
    dir: &Path,
    page: usize,
    crops: &[Crop],
) -> Result<(), CliError> {
    std::fs::create_dir_all(dir).map_err(|source| OcrError::Write {
        path: dir.display().to_string(),
        source,
    })?;
    for (index, crop) in crops.iter().enumerate() {
        let path: PathBuf = dir.join(format!("p{page:03}-l{index:04}.png"));
        let png = crop::to_png(crop)?;
        std::fs::write(&path, png).map_err(|source| OcrError::Write {
            path: path.display().to_string(),
            source,
        })?;
    }
    Ok(())
}
