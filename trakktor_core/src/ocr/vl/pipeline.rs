//! The engine end to end: detection from one model, reading from another.
//!
//! `vl` is not a self-contained pipeline and does not pretend to be one. It
//! borrows the classic engine's detector — 4.7 MB, a fraction of a second, and
//! **indifferent to the writing system**, so it finds Tibetan lines no classic
//! recognizer could read — and spends the 1.92 GB model only on reading. The
//! division is by strength: detection where it is cheap, recognition where it
//! is the only thing that works.
//!
//! What the run then does with the result is the domain's business, not this
//! module's: the pages it produces go through the same reading order, layout
//! analysis, Markdown assembly and illustration extraction as the classic
//! engine's.

use std::path::Path;

use candle_core::{DType, Tensor};
use candle_nn::VarBuilder;
use image::RgbImage;
use tokenizers::Tokenizer;

use super::{
    blocks::{self, Settings as BlockSettings},
    config::{ImageConfig, ModelConfig, TOKENIZER_FILE, WEIGHTS_FILE},
    download,
    ernie::Decoder,
    generate::{Answer, Limits, Reader, Task},
    image as picture, model,
    vision::{Projector, Tower},
};
use crate::{
    download::Progress,
    ocr::{
        error::OcrError,
        figures::{self, Figure},
        paddle::{
            artifact::Artifact,
            db::{self, Params},
            det::Detector,
            download as paddle_download,
            image::{self as raw, LimitType, Page as RawPage},
            model as paddle_model,
            net::Loader,
            pipeline::{Device, sort_boxes},
        },
        page::{Line, Page, Quad},
    },
};

/// How a run is configured.
#[derive(Debug, Clone)]
pub struct Options {
    /// The checkpoint to read with, or a path to a directory holding one.
    pub model: String,
    /// The detector that finds the lines, as in the classic engine.
    pub detection: String,
    /// Longest side the page is resized to before detection.
    pub limit_side_len: usize,
    /// Detection post-processing thresholds.
    pub params: Params,
    /// What to ask the model for.
    pub task: Task,
    /// When to stop generating.
    pub limits: Limits,
    /// How the detected lines are grouped.
    pub blocks: BlockSettings,
    /// Read the page in one call instead of block by block. The right choice
    /// when the page *is* one block, and a way into a repeat loop when it is
    /// not.
    pub whole_page: bool,
    /// Blocks the model read with less confidence than this are dropped.
    pub drop_score: f32,
    /// Also look for illustrations.
    pub figures: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            model: model::DEFAULT_MODEL.to_string(),
            detection: paddle_model::DEFAULT_DETECTION.to_string(),
            limit_side_len: DEFAULT_LIMIT_SIDE_LEN,
            params: Params::default(),
            task: Task::Ocr,
            limits: Limits::default(),
            blocks: BlockSettings::default(),
            whole_page: false,
            drop_score: DEFAULT_DROP_SCORE,
            figures: false,
        }
    }
}

/// The default longest side of the detector's input.
///
/// Higher than the classic engine's, and on purpose: there the detector's
/// output *is* the answer's geometry, so a coarse page merely loses small
/// type; here a missed line also means a block that never reaches the reader.
pub const DEFAULT_LIMIT_SIDE_LEN: usize = 1440;

/// The default confidence floor. Lower than the classic engine's: this score
/// is a mean token probability, and a correct reading of an unusual script
/// sits lower on that scale than a correct reading of Latin prose.
pub const DEFAULT_DROP_SCORE: f32 = 0.3;

/// A loaded engine.
pub struct Engine {
    detector: Detector,
    reader: Reader,
    detection_name: String,
    model_name: String,
    options: Options,
}

impl Engine {
    /// Resolves the models named by `options`, downloading what is missing,
    /// and loads them onto `device`.
    pub fn load(
        models_dir: &Path,
        device: Device,
        options: Options,
        progress: Progress<'_>,
    ) -> Result<Self, OcrError> {
        let device = device.resolve()?;
        // Half precision on a GPU: the weights are published in bfloat16 and
        // the three precisions were measured to produce the same text, so this
        // is memory and speed for nothing. On the CPU candle has no fast half
        // path, so full precision is the faster choice there.
        let dtype = if device.is_cpu() {
            DType::F32
        } else {
            DType::F16
        };

        let detection = paddle_download::resolve(
            models_dir,
            &options.detection,
            &mut *progress,
        )?;
        let checkpoint =
            download::resolve(models_dir, &options.model, progress)?;

        let detector = {
            let artifact = Artifact::load(&detection.dir)?;
            Detector::load(&Loader::new(&artifact, &device))?
        };

        let cfg = ModelConfig::load(&checkpoint.dir)?;
        let image_cfg = ImageConfig::load(&checkpoint.dir)?;
        let tokenizer = Tokenizer::from_file(
            checkpoint.dir.join(TOKENIZER_FILE),
        )
        .map_err(|e| {
            OcrError::Artifact(format!("cannot read the tokenizer: {e}"))
        })?;

        let weights = checkpoint.dir.join(WEIGHTS_FILE);
        // SAFETY: the weights are memory-mapped and must not change while the
        // engine is loaded; they are ours, under the model directory, and
        // verified against a digest when they were fetched.
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights], dtype, &device)?
        };
        let tower = Tower::load(&cfg.vision, vb.pp("visual"))?;
        let projector =
            Projector::load(&cfg.vision, cfg.hidden_size, vb.pp("mlp_AR"))?;
        let decoder = Decoder::load(&cfg, vb)?;

        let reader = Reader::new(
            tower, projector, decoder, tokenizer, cfg, image_cfg, device, dtype,
        )?;

        Ok(Self {
            detector,
            reader,
            detection_name: detection
                .name
                .map(str::to_string)
                .unwrap_or_else(|| options.detection.clone()),
            model_name: checkpoint
                .name
                .map(str::to_string)
                .unwrap_or_else(|| options.model.clone()),
            options,
        })
    }

    /// The models this engine is running, for the result envelope.
    pub fn models(&self) -> (&str, &str) {
        (&self.detection_name, &self.model_name)
    }

    pub fn options(&self) -> &Options { &self.options }

    /// Reads one page image.
    ///
    /// `report` is called before each block is read, with the block's index and
    /// how many there are — a page is tens of seconds and a caller that shows
    /// nothing for that long looks stuck.
    pub fn read_page(
        &self,
        page: &RawPage,
        number: usize,
        source: &str,
        report: &mut dyn FnMut(usize, usize),
    ) -> Result<Read, OcrError> {
        let quads = self.detect(page)?;
        let size = (page.width, page.height);

        let regions: Vec<blocks::Block> = if self.options.whole_page ||
            self.options.task.wants_whole_page()
        {
            vec![blocks::whole(&quads, size)]
        } else {
            let ink = blocks::ink_profile(&page.bgr, size);
            blocks::assemble(&quads, size, Some(&ink), &self.options.blocks)
        };

        let mut lines = Vec::new();
        let mut crops = Vec::new();
        for (index, region) in regions.iter().enumerate() {
            report(index, regions.len());
            let cut = blocks::cut(&page.bgr, size, region);
            let answer = self.read_block(&cut)?;
            if answer.score < self.options.drop_score || answer.text.is_empty()
            {
                continue;
            }
            lines.extend(place(&answer, region, &quads));
            crops.push(cut);
        }

        let illustrations = if self.options.figures {
            figures::find(
                &page.bgr,
                page.width as usize,
                page.height as usize,
                &quads,
                &figures::Settings::default(),
            )
        } else {
            Vec::new()
        };

        Ok(Read {
            figures: illustrations,
            page: Page {
                number,
                source: source.to_string(),
                width: page.width,
                height: page.height,
                lines,
            },
            crops,
        })
    }

    /// Reads a page image from a file.
    pub fn read_file(
        &self,
        path: &Path,
        number: usize,
        report: &mut dyn FnMut(usize, usize),
    ) -> Result<Read, OcrError> {
        let page = RawPage::load(path)?;
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| path.display().to_string());
        self.read_page(&page, number, &name, report)
    }

    /// Reads one already-cut picture.
    pub fn read_block(&self, picture: &RgbImage) -> Result<Answer, OcrError> {
        let prepared = picture::prepare(picture, self.reader.image_config())?;
        Ok(self.reader.read(
            &prepared,
            self.options.task,
            &self.options.limits,
        )?)
    }

    /// Runs the detector and returns the line boxes in reading order.
    fn detect(&self, page: &RawPage) -> Result<Vec<Quad>, OcrError> {
        let input = raw::detector_input(
            page,
            self.options.limit_side_len,
            LimitType::Max,
            &raw::DETECTOR_NORMALIZE,
        );
        let tensor = Tensor::from_vec(
            input.data,
            (1, raw::CHANNELS, input.height, input.width),
            self.reader.device(),
        )?
        .to_dtype(DType::F32)?;
        let map = self.detector.forward(&tensor)?;
        let probabilities = map.flatten_all()?.to_vec1::<f32>()?;

        let mut boxes = db::boxes_from_bitmap(
            &probabilities,
            input.width,
            input.height,
            (page.width, page.height),
            (input.ratio_height as f32, input.ratio_width as f32),
            &self.options.params,
        );
        sort_boxes(&mut boxes);
        Ok(boxes.into_iter().map(|(quad, _)| quad).collect())
    }
}

/// Turns one block's answer into result lines.
///
/// When the model returned as many lines as the detector found boxes, the two
/// are matched up in order. When it did not — and it often does not, because
/// this version reflows the text and joins words broken across lines — every
/// line gets the block's own rectangle. Reporting a box that was not measured
/// would be worse than reporting a coarse one.
fn place(answer: &Answer, region: &blocks::Block, quads: &[Quad]) -> Vec<Line> {
    let texts: Vec<&str> = answer
        .text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect();
    let aligned = texts.len() == region.lines.len();
    texts
        .iter()
        .enumerate()
        .map(|(at, text)| Line {
            text: (*text).to_string(),
            score: answer.score,
            quad: if aligned {
                quads[region.lines[at]]
            } else {
                region.rect.quad()
            },
            rotated: false,
            truncated: answer.truncated,
        })
        .collect()
}

/// One page's result, plus the block pictures the model actually read.
pub struct Read {
    pub page: Page,
    /// The blocks, as they were handed to the model. The unit is the block,
    /// not the line — which is what makes them worth writing out: they show
    /// exactly what the model was asked to make sense of.
    crops: Vec<RgbImage>,
    pub figures: Vec<Figure>,
}

impl Read {
    /// How many blocks the model was given.
    pub fn crops(&self) -> usize { self.crops.len() }

    /// One block's picture, encoded as PNG.
    pub fn crop_png(&self, index: usize) -> Result<Vec<u8>, OcrError> {
        use image::{ImageEncoder, codecs::png::PngEncoder};

        let picture = self.crops.get(index).ok_or_else(|| {
            OcrError::Runtime(format!("no block {index} on this page"))
        })?;
        let mut png = Vec::new();
        PngEncoder::new(&mut png)
            .write_image(
                picture.as_raw(),
                picture.width(),
                picture.height(),
                image::ExtendedColorType::Rgb8,
            )
            .map_err(|e| {
                OcrError::Runtime(format!("encoding a block picture: {e}"))
            })?;
        Ok(png)
    }
}
