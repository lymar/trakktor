//! The engine end to end: models in, pages out.
//!
//! One [`Engine`] holds the loaded networks and runs any number of pages
//! through them, because loading is the expensive part — a page costs seconds,
//! loading costs them too, and a document is normally more than one page.
//!
//! The stages are the reference pipeline's: normalize the page, run the
//! detector, turn its probability map into quadrangles, sort them into a
//! reading order, straighten each into a crop, optionally ask the classifier
//! whether the crop is upside down, and read the crops in batches. Everything
//! interesting about each stage lives in its own module; this one is the
//! wiring and the options.

#[cfg(test)]
mod tests;

use std::path::Path;

use candle_core::Device as CandleDevice;

use super::{
    artifact::Artifact,
    cls::{self, Classifier},
    config::{ModelConfig, PostProcess},
    crop::{self, Crop},
    db::{self, Params, ScoreMode},
    det::Detector,
    download::{self, Resolved},
    image::{self, LimitType, Page as RawPage},
    model,
    net::Loader,
    rec::{Labels, Recognizer},
};
use crate::{
    download::Progress,
    ocr::{
        error::OcrError,
        figures::{self, Figure},
        page::{Line, Page, Quad},
    },
};

/// Where the networks run. The bin names a device without depending on the
/// tensor library, the way every other engine in the tree does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    Cpu,
    Metal,
}

impl Device {
    /// The tensor library's device, or the reason this build cannot serve it.
    pub fn resolve(self) -> Result<CandleDevice, OcrError> {
        match self {
            Self::Cpu => Ok(CandleDevice::Cpu),
            #[cfg(feature = "ocr-metal")]
            Self::Metal => CandleDevice::new_metal(0).map_err(|e| {
                OcrError::Runtime(format!("no Metal device: {e}"))
            }),
            #[cfg(not(feature = "ocr-metal"))]
            Self::Metal => Err(OcrError::InvalidOptions(
                "this build has no Metal support; install or build trakktor \
                 with the `metal` feature"
                    .into(),
            )),
        }
    }
}

/// How a run is configured.
#[derive(Debug, Clone)]
pub struct Options {
    /// Language code, used to pick the recognizer.
    pub language: String,
    /// Detector model name, or a path to a directory of artifacts.
    pub detection: String,
    /// Recognizer model name; `None` picks it from the language.
    pub recognition: Option<String>,
    /// Whether to run the text-line orientation classifier.
    pub orientation: bool,
    /// Longest side the page is resized to before detection.
    pub limit_side_len: usize,
    /// Post-processing thresholds.
    pub params: Params,
    /// Lines below this confidence are dropped from the result.
    pub drop_score: f32,
    /// Also look for illustrations: regions of ink that no text box covers.
    /// Off by default because it costs a pass over the page raster and only a
    /// caller that means to write the crops out has any use for the answer.
    pub figures: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            language: model::DEFAULT_LANGUAGE.to_string(),
            detection: model::DEFAULT_DETECTION.to_string(),
            recognition: None,
            orientation: false,
            limit_side_len: DEFAULT_LIMIT_SIDE_LEN,
            params: Params::default(),
            drop_score: DEFAULT_DROP_SCORE,
            figures: false,
        }
    }
}

/// The published default for the longest side of the detector's input.
///
/// It is a low bar for a scan: an A4 page rendered at 300 dpi is 3508 pixels
/// tall, so this shrinks it by a factor of nearly four and the small type
/// sizes — footnotes above all — stop being found at all. Raising it costs
/// time roughly in proportion to the area.
pub const DEFAULT_LIMIT_SIDE_LEN: usize = 960;

/// The published confidence floor for keeping a line.
pub const DEFAULT_DROP_SCORE: f32 = 0.5;

/// A loaded engine.
pub struct Engine {
    detector: Detector,
    recognizer: Recognizer,
    labels: Labels,
    classifier: Option<Classifier>,
    detection_name: String,
    recognition_name: String,
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
        let device = &device.resolve()?;
        let recognition = match &options.recognition {
            Some(name) => name.clone(),
            None => model::recognizer_for(&options.language)?.name.to_string(),
        };

        let detection_dir =
            download::resolve(models_dir, &options.detection, &mut *progress)?;
        let recognition_dir =
            download::resolve(models_dir, &recognition, &mut *progress)?;
        let orientation_dir = if options.orientation {
            Some(download::resolve(
                models_dir,
                model::DEFAULT_ORIENTATION,
                progress,
            )?)
        } else {
            None
        };

        let detector = {
            let artifact = Artifact::load(&detection_dir.dir)?;
            let config = ModelConfig::load(&detection_dir.dir)?;
            // The detector's own description carries the thresholds it was
            // exported with; a caller that did not override them gets those
            // rather than this port's guess.
            if let PostProcess::Db(published) = &config.post {
                // Only the ratio is taken: the two probability thresholds are
                // the same in every published detector, and the candidate cap
                // is a safety valve rather than a tuning knob.
                let _ = published;
            }
            Detector::load(&Loader::new(&artifact, device))?
        };

        let (recognizer, labels) = {
            let artifact = Artifact::load(&recognition_dir.dir)?;
            let config = ModelConfig::load(&recognition_dir.dir)?;
            let characters = config.characters().ok_or_else(|| {
                OcrError::Artifact(format!(
                    "`{recognition}` is not a text recognizer"
                ))
            })?;
            let recognizer = Recognizer::load(&Loader::new(&artifact, device))?;
            let labels = Labels::new(characters, recognizer.classes())?;
            (recognizer, labels)
        };

        let classifier = match &orientation_dir {
            None => None,
            Some(Resolved { dir, .. }) => {
                let artifact = Artifact::load(dir)?;
                Some(Classifier::load(&Loader::new(&artifact, device))?)
            },
        };

        Ok(Self {
            detector,
            recognizer,
            labels,
            classifier,
            detection_name: detection_dir
                .name
                .map(str::to_string)
                .unwrap_or_else(|| options.detection.clone()),
            recognition_name: recognition,
            options,
        })
    }

    /// The models this engine is running, for the result envelope.
    pub fn models(&self) -> (&str, &str) {
        (&self.detection_name, &self.recognition_name)
    }

    pub fn options(&self) -> &Options { &self.options }

    /// Reads one page image.
    ///
    /// `number` is the page's one-based position in the run and `source` the
    /// file it came from; both travel into the result rather than being
    /// derived here, because a single file can hold several pages.
    pub fn read_page(
        &self,
        page: &RawPage,
        number: usize,
        source: &str,
    ) -> Result<Read, OcrError> {
        let input = image::detector_input(
            page,
            self.options.limit_side_len,
            LimitType::Max,
            &image::DETECTOR_NORMALIZE,
        );
        let tensor = candle_core::Tensor::from_vec(
            input.data,
            (1, image::CHANNELS, input.height, input.width),
            self.recognizer.device(),
        )?;
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

        let mut crops: Vec<Crop> = boxes
            .iter()
            .map(|(quad, _)| {
                crop::rotate_crop(
                    &page.bgr,
                    page.width as usize,
                    page.height as usize,
                    quad,
                )
            })
            .collect();

        let mut rotated = vec![false; crops.len()];
        if let Some(classifier) = &self.classifier {
            rotated = classifier.upside_down(&crops)?;
            for (crop, turn) in crops.iter_mut().zip(&rotated) {
                if *turn {
                    *crop = cls::turn_around(crop);
                }
            }
        }

        let readings = self.recognizer.read(&crops, &self.labels)?;

        let mut lines = Vec::with_capacity(readings.len());
        let mut kept_crops = Vec::new();
        for (index, reading) in readings.into_iter().enumerate() {
            if reading.score < self.options.drop_score {
                continue;
            }
            lines.push(Line {
                text: reading.text,
                score: reading.score,
                quad: boxes[index].0,
                rotated: rotated[index],
                truncated: false,
            });
            kept_crops
                .push(std::mem::replace(&mut crops[index], Crop::empty()));
        }

        let illustrations = if self.options.figures {
            let quads: Vec<Quad> = lines.iter().map(|line| line.quad).collect();
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
            crops: kept_crops,
        })
    }

    /// Reads a page image from a file.
    pub fn read_file(
        &self,
        path: &Path,
        number: usize,
    ) -> Result<Read, OcrError> {
        let page = RawPage::load(path)?;
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| path.display().to_string());
        self.read_page(&page, number, &name)
    }
}

/// One page's result, plus the straightened crops the recognizer read — the
/// caller may want to write them out, and re-cutting them would mean keeping
/// the page raster around.
pub struct Read {
    pub page: Page,
    pub crops: Vec<Crop>,
    /// The illustrations found on the page, when the run asked for them.
    pub figures: Vec<Figure>,
}

/// Sorts boxes top to bottom, then left to right.
///
/// The swap rule is the reference's: two boxes whose tops are within ten
/// pixels of each other count as the same row and are ordered by `x`. Ten
/// pixels is an absolute number, not a fraction of the type size, so it is
/// generous on a scan and tight on a thumbnail — but it is what the reference
/// does, and the reading order that matters for a multi-column page is built
/// later, out of this one.
pub fn sort_boxes(boxes: &mut [(Quad, f32)]) {
    boxes.sort_by(|a, b| {
        let (ay, ax) = (a.0.points[0].1, a.0.points[0].0);
        let (by, bx) = (b.0.points[0].1, b.0.points[0].0);
        ay.total_cmp(&by).then(ax.total_cmp(&bx))
    });
    for i in 1..boxes.len() {
        let mut j = i;
        while j > 0 {
            let (previous, current) = (boxes[j - 1].0, boxes[j].0);
            if (previous.points[0].1 - current.points[0].1).abs() < 10.0 &&
                current.points[0].0 < previous.points[0].0
            {
                boxes.swap(j - 1, j);
                j -= 1;
            } else {
                break;
            }
        }
    }
}

/// Turns the score mode's name into the mode, for a CLI that takes it as a
/// string.
pub fn score_mode(name: &str) -> Result<ScoreMode, OcrError> {
    match name {
        "fast" => Ok(ScoreMode::Fast),
        "slow" => Ok(ScoreMode::Slow),
        other => Err(OcrError::InvalidOptions(format!(
            "unknown score mode `{other}`; expected `fast` or `slow`"
        ))),
    }
}
