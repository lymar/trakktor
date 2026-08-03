//! `ocr paddle`: flag mapping, model resolution, and the page loop.

use std::path::{Path, PathBuf};

use trakktor_core::ocr::{
    OcrError, Page, Quad,
    figures::{self, Figure},
    markdown,
    paddle::{
        crop::{self, Crop},
        db::Params,
        image::Page as RawPage,
        model,
        pipeline::{Device, Engine, Options},
    },
};

use crate::{
    cli::{OcrFormatArg, OcrPaddleArgs},
    error::CliError,
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
        device(args),
        options,
        &mut crate::asr::progress::download_progress(),
    )?;

    let mut pages: Vec<Page> = Vec::with_capacity(args.pages.len());
    let mut figures: Vec<Vec<Quad>> = Vec::with_capacity(args.pages.len());
    for (index, path) in args.pages.iter().enumerate() {
        let read = engine.read_file(path, index + 1)?;
        if let Some(dir) = &args.crops {
            write_crops(dir, index + 1, &read.crops)?;
        }
        if !read.figures.is_empty() {
            let raster = RawPage::load(path)?;
            write_figures(
                args.out.as_deref(),
                index + 1,
                &raster,
                &read.figures,
            )?;
        }
        figures.push(read.figures.iter().map(Figure::quad).collect());
        pages.push(read.page);
    }

    let markdown = matches!(args.format, OcrFormatArg::Md);
    let (detection, recognition) = engine.models();
    crate::output::print_ocr(
        &pages,
        &figures,
        &args.lang,
        detection,
        recognition,
        markdown,
        args.out.as_deref(),
        json,
        pretty,
    )
    .map_err(CliError::from)
}

/// The device the run asks for. Whether this build can serve it is the
/// engine's to say, so that the answer is the same however trakktor is
/// embedded.
fn device(args: &OcrPaddleArgs) -> Device {
    match args.device {
        crate::cli::DeviceArg::Cpu => Device::Cpu,
        crate::cli::DeviceArg::Metal => Device::Metal,
    }
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
