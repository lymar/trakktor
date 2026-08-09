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
        download as layout_download, post,
    },
    markdown,
    overlay::{self, Shape, Weight},
    paddle::{
        crop::{self, Crop},
        db::Thresholds,
        download as paddle_download,
        image::Page as RawPage,
        model::{self, Quality},
        pipeline::{Device, Engine, Options},
    },
    preprocess::{self, Prepared, Preprocessor},
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
            &model::catalogue(quality(args.quality)),
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
        quality: quality(args.quality),
        detection: args.det_model.clone(),
        recognition: args.rec_model.clone(),
        orientation: args.textline_orientation,
        limit_side_len: args.limit_side_len,
        thresholds: Thresholds {
            thresh: args.thresh,
            box_thresh: args.box_thresh,
            unclip_ratio: args.unclip_ratio,
        },
        drop_score: args.drop_score,
        // Illustrations are only worth looking for when there is somewhere to
        // put them: a Markdown run writing to a file.
        figures: matches!(args.format, OcrFormatArg::Md) && args.out.is_some(),
    };

    let layout_model = (!args.no_layout).then(|| {
        args.layout_model.clone().unwrap_or_else(|| {
            trakktor_core::ocr::layout::model::DEFAULT_MODEL.to_string()
        })
    });
    let preprocess = args.preprocess.options();
    announce_downloads(
        &paddle_download::pending(model_dir, &options),
        layout_model
            .as_deref()
            .map(|name| layout_download::pending(model_dir, name))
            .unwrap_or_default()
            .into_iter()
            .chain(preprocess.pending(model_dir))
            .collect(),
    );

    let preprocessor = load_preprocessor(model_dir, preprocess, args.device)?;
    let engine = Engine::load(
        model_dir,
        device(args.device),
        options,
        &mut crate::asr::progress::download_progress(),
    )?;
    let marker = layout_model
        .as_deref()
        .map(|name| {
            load_layout(
                model_dir,
                Some(name),
                args.layout_threshold,
                device(args.device),
                RuntimeArg::Candle,
            )
        })
        .transpose()?;

    let mut pages: Vec<Page> = Vec::with_capacity(args.pages.len());
    let mut figures: Vec<Vec<Quad>> = Vec::with_capacity(args.pages.len());
    let mut regions: Vec<Vec<Region>> = Vec::with_capacity(args.pages.len());
    let mut prepared: Vec<Prepared> = Vec::with_capacity(args.pages.len());
    for (index, path) in args.pages.iter().enumerate() {
        // One decode of the page serves every stage: the markup runs first,
        // because a line the detector glued across a boundary between two
        // regions is taken apart before it is read.
        let photograph = RawPage::load(path)?;
        // Boxes are drawn on the file the caller named, so the page as it
        // arrived is kept when — and only when — there is a drawing to do.
        let original = (args.boxes.is_some() && preprocess.any())
            .then(|| photograph.clone());
        let stage = prepare(&preprocessor, photograph)?;
        write_rectified(
            args.preprocess.rectified.as_deref(),
            index + 1,
            args.pages.len(),
            &stage,
        )?;
        let raster = &stage.page;
        let marked = match &marker {
            None => Vec::new(),
            Some(marker) => marker.detect(raster)?,
        };
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| path.display().to_string());
        let read =
            engine.read_page_marked(raster, index + 1, &name, &marked)?;
        if let Some(dir) = &args.crops {
            write_crops(dir, index + 1, &read.crops)?;
        }
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
            write_figures(args.out.as_deref(), index + 1, raster, &found)?;
        }
        if let Some(target) = &args.boxes {
            let mut shapes = line_shapes(&read.page);
            if let Some(page) = &original {
                for shape in &mut shapes {
                    shape.quad = stage.locate(&shape.quad);
                }
                write_boxes(
                    target,
                    index + 1,
                    args.pages.len(),
                    page,
                    &shapes,
                )?;
            } else {
                write_boxes(
                    target,
                    index + 1,
                    args.pages.len(),
                    raster,
                    &shapes,
                )?;
            }
        }
        figures.push(found.iter().map(Figure::quad).collect());
        regions.push(marked);
        pages.push(read.page);
        prepared.push(stage);
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
        &prepared,
        markdown,
        args.out.as_deref(),
        json,
        pretty,
    )
    .map_err(CliError::from)
}

/// Loads the preprocessing stage, or nothing when nothing was asked for.
fn load_preprocessor(
    model_dir: &Path,
    options: preprocess::Options,
    device_arg: crate::cli::DeviceArg,
) -> Result<Option<Preprocessor>, CliError> {
    if !options.any() {
        return Ok(None);
    }
    let resolved = device(device_arg).resolve()?;
    Ok(Some(Preprocessor::load(
        model_dir,
        options,
        &resolved,
        &mut crate::asr::progress::download_progress(),
    )?))
}

/// Runs the stage over one page, or hands it straight back.
fn prepare(
    preprocessor: &Option<Preprocessor>,
    page: RawPage,
) -> Result<Prepared, CliError> {
    match preprocessor {
        None => Ok(Prepared::untouched(page)),
        Some(preprocessor) => Ok(preprocessor.run(page)?),
    }
}

/// Resolves and loads the layout model.
fn load_layout(
    model_dir: &Path,
    model: Option<&str>,
    threshold: Option<f32>,
    device: Device,
    runtime: RuntimeArg,
) -> Result<Detector, CliError> {
    let options = detect::Options {
        model: model.map(str::to_string).unwrap_or_else(|| {
            trakktor_core::ocr::layout::model::DEFAULT_MODEL.to_string()
        }),
        post: post::Settings {
            threshold,
            ..post::Settings::default()
        },
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
        thresholds: Thresholds {
            thresh: args.thresh,
            box_thresh: args.box_thresh,
            unclip_ratio: args.unclip_ratio,
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

    let layout_model = (!args.no_layout).then(|| {
        args.layout_model.clone().unwrap_or_else(|| {
            trakktor_core::ocr::layout::model::DEFAULT_MODEL.to_string()
        })
    });
    let mut reading = vl::download::pending(model_dir, &options.model);
    reading.extend(paddle_download::pending_named(
        model_dir,
        &options.detection,
    ));
    let preprocess = args.preprocess.options();
    announce_downloads(
        &reading,
        layout_model
            .as_deref()
            .map(|name| layout_download::pending(model_dir, name))
            .unwrap_or_default()
            .into_iter()
            .chain(preprocess.pending(model_dir))
            .collect(),
    );

    let preprocessor = load_preprocessor(model_dir, preprocess, args.device)?;
    let mut engine = vl::Engine::load(
        model_dir,
        device(args.device),
        runtime,
        options,
        &mut crate::asr::progress::download_progress(),
    )?;

    let marker = layout_model
        .as_deref()
        .map(|name| {
            load_layout(
                model_dir,
                Some(name),
                args.layout_threshold,
                device(args.device),
                args.runtime,
            )
        })
        .transpose()?;

    let mut pages: Vec<Page> = Vec::with_capacity(args.pages.len());
    let mut figures: Vec<Vec<Quad>> = Vec::with_capacity(args.pages.len());
    let mut regions: Vec<Vec<Region>> = Vec::with_capacity(args.pages.len());
    let mut prepared: Vec<Prepared> = Vec::with_capacity(args.pages.len());
    let started = Instant::now();
    for (index, path) in args.pages.iter().enumerate() {
        let number = index + 1;
        let mut report = block_progress(started, number, args.pages.len());
        let photograph = RawPage::load(path)?;
        let original = (args.boxes.is_some() && preprocess.any())
            .then(|| photograph.clone());
        let stage = prepare(&preprocessor, photograph)?;
        write_rectified(
            args.preprocess.rectified.as_deref(),
            number,
            args.pages.len(),
            &stage,
        )?;
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| path.display().to_string());
        let marked = match &marker {
            None => Vec::new(),
            Some(marker) => marker.detect(&stage.page)?,
        };
        let read = engine.read_page_marked(
            &stage.page,
            number,
            &name,
            &marked,
            &mut report,
        )?;
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
            write_figures(args.out.as_deref(), number, &stage.page, &found)?;
        }
        if let Some(target) = &args.boxes {
            let mut shapes = block_shapes(&read);
            if let Some(page) = &original {
                for shape in &mut shapes {
                    shape.quad = stage.locate(&shape.quad);
                }
                write_boxes(target, number, args.pages.len(), page, &shapes)?;
            } else {
                write_boxes(
                    target,
                    number,
                    args.pages.len(),
                    &stage.page,
                    &shapes,
                )?;
            }
        }
        figures.push(found.iter().map(Figure::quad).collect());
        regions.push(marked);
        pages.push(read.page);
        prepared.push(stage);
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
        &prepared,
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

    let model = args.model.clone().unwrap_or_else(|| {
        trakktor_core::ocr::layout::model::DEFAULT_MODEL.to_string()
    });
    announce_downloads(&[], layout_download::pending(model_dir, &model));

    let marker = load_layout(
        model_dir,
        Some(&model),
        args.threshold,
        device(args.device),
        args.runtime,
    )?;

    let mut pages = Vec::with_capacity(args.pages.len());
    for (index, path) in args.pages.iter().enumerate() {
        let raster = RawPage::load(path)?;
        let regions = marker.detect(&raster)?;
        if let Some(dir) = &args.crops {
            write_regions(dir, index + 1, &raster, &regions)?;
        }
        if let Some(target) = &args.boxes {
            write_boxes(
                target,
                index + 1,
                args.pages.len(),
                &raster,
                &region_shapes(&regions),
            )?;
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

/// Which end of the model catalog the run draws its defaults from.
fn quality(chosen: crate::cli::OcrQualityArg) -> Quality {
    match chosen {
        crate::cli::OcrQualityArg::Best => Quality::Best,
        crate::cli::OcrQualityArg::Fast => Quality::Fast,
    }
}

/// Names what a run is about to download, and how much of it, before the first
/// byte moves.
///
/// The engine reads with the strongest models the catalog has rather than the
/// cheapest, so a first run fetches around a hundred megabytes for the reading
/// and another hundred and twenty-nine for the markup. Whoever typed one short
/// command should learn that from the command, not from watching a progress
/// line for a minute and guessing what it is doing.
fn announce_downloads(
    reading: &[(&str, u64)],
    markup: Vec<(&'static str, u64)>,
) {
    let all: Vec<(&str, u64)> = reading.iter().copied().chain(markup).collect();
    if all.is_empty() {
        return;
    }
    let total: u64 = all.iter().map(|(_, bytes)| bytes).sum();
    let names: Vec<String> = all
        .iter()
        .map(|(name, bytes)| format!("{name} {}", megabytes(*bytes)))
        .collect();
    eprintln!(
        "fetching {} of models on first use: {}",
        megabytes(total),
        names.join(", ")
    );
}

/// A byte count as whole or tenths of a megabyte, the unit model sizes are
/// published in.
fn megabytes(bytes: u64) -> String {
    let mb = bytes as f64 / 1_000_000.0;
    if mb < 10.0 {
        format!("{mb:.1} MB")
    } else {
        format!("{:.0} MB", mb.round())
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

/// What the classic engine's overlay shows: one outline per reported line,
/// numbered as the result numbers it, so the picture, the JSON and the line
/// crops all count the same lines.
fn line_shapes(page: &Page) -> Vec<Shape> {
    page.lines
        .iter()
        .enumerate()
        .map(|(index, line)| Shape {
            quad: line.quad,
            color: overlay::score_color(line.score),
            weight: Weight::Thick,
            caption: Some(index.to_string()),
        })
        .collect()
}

/// What the generative engine's overlay shows: the blocks it read, over the
/// line boxes they were assembled from. Two layers because there are two
/// questions — what the detector found, and how it was grouped — and a wrong
/// reading here is nearly always the second one.
fn block_shapes(read: &vl::Read) -> Vec<Shape> {
    let detected = read.detected.iter().map(|quad| Shape {
        quad: *quad,
        color: overlay::CONTEXT,
        weight: Weight::Thin,
        caption: None,
    });
    let blocks = read.blocks.iter().map(|block| Shape {
        quad: block.quad,
        color: match block.crop {
            Some(_) => overlay::CONFIDENT,
            None => overlay::DOUBTFUL,
        },
        weight: Weight::Thick,
        // A dropped block is the one thing the result cannot show: its text is
        // simply not there, and without the picture neither is the reason.
        caption: Some(match block.crop {
            Some(at) => at.to_string(),
            None => "DROPPED".to_string(),
        }),
    });
    detected.chain(blocks).collect()
}

/// What the layout overlay shows: every region with its number and its label.
fn region_shapes(regions: &[Region]) -> Vec<Shape> {
    regions
        .iter()
        .enumerate()
        .map(|(index, region)| Shape {
            quad: region.quad,
            color: overlay::score_color(region.score),
            weight: Weight::Thick,
            caption: Some(format!("{index} {}", region.label.name())),
        })
        .collect()
}

/// Writes one page with the shapes drawn over it.
///
/// The target is the file itself for a single-page run and a directory of
/// `pNNN.png` for a longer one: one page is the debugging case, and being able
/// to name that file is the point of it.
fn write_boxes(
    target: &Path,
    page: usize,
    pages: usize,
    raster: &RawPage,
    shapes: &[Shape],
) -> Result<(), CliError> {
    let png = overlay::draw_png(
        &raster.bgr,
        raster.width as usize,
        raster.height as usize,
        shapes,
    )?;
    let path = if pages > 1 || target.is_dir() {
        std::fs::create_dir_all(target).map_err(|source| OcrError::Write {
            path: target.display().to_string(),
            source,
        })?;
        target.join(format!("p{page:03}.png"))
    } else {
        target.to_path_buf()
    };
    std::fs::write(&path, png).map_err(|source| OcrError::Write {
        path: path.display().to_string(),
        source,
    })?;
    Ok(())
}

/// Writes the page the reading was actually done on.
///
/// Only when the stage changed something: a run that asked for straightening
/// and got a page that needed none should not be handed a copy of its own
/// file.
fn write_rectified(
    target: Option<&Path>,
    page: usize,
    pages: usize,
    prepared: &Prepared,
) -> Result<(), CliError> {
    let Some(target) = target else {
        return Ok(());
    };
    if !prepared.changed() {
        return Ok(());
    }
    let raster = &prepared.page;
    let png = overlay::draw_png(
        &raster.bgr,
        raster.width as usize,
        raster.height as usize,
        &[],
    )?;
    let path = if pages > 1 || target.is_dir() {
        std::fs::create_dir_all(target).map_err(|source| OcrError::Write {
            path: target.display().to_string(),
            source,
        })?;
        target.join(format!("p{page:03}.png"))
    } else {
        target.to_path_buf()
    };
    std::fs::write(&path, png).map_err(|source| OcrError::Write {
        path: path.display().to_string(),
        source,
    })?;
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
