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
    db::{self, Params, ScoreMode, Thresholds},
    det::Detector,
    download::{self, Resolved},
    image::{self, LimitType, Page as RawPage},
    model::{self, Quality},
    net::Loader,
    rec::{Labels, Recognizer},
};
use crate::{
    download::Progress,
    ocr::{
        error::OcrError,
        figures::{self, Figure},
        layout::region::{self, Region},
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
    /// Which end of the catalog the models come from when they are not named
    /// outright.
    pub quality: Quality,
    /// Detector model name, or a path to a directory of artifacts; `None`
    /// takes the one `quality` asks for.
    pub detection: Option<String>,
    /// Recognizer model name; `None` picks it from the language and `quality`.
    pub recognition: Option<String>,
    /// Whether to run the text-line orientation classifier.
    pub orientation: bool,
    /// Longest side the page is resized to before detection.
    pub limit_side_len: usize,
    /// Post-processing thresholds the run names. What it does not name comes
    /// from the detector's own description — see [`Thresholds`].
    pub thresholds: Thresholds,
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
            quality: Quality::default(),
            detection: None,
            recognition: None,
            orientation: false,
            limit_side_len: DEFAULT_LIMIT_SIDE_LEN,
            thresholds: Thresholds::default(),
            drop_score: DEFAULT_DROP_SCORE,
            figures: false,
        }
    }
}

impl Options {
    /// The detector this run will load: the one it names, or the one its
    /// quality asks for.
    pub fn detector(&self) -> &str {
        match &self.detection {
            Some(named) => named,
            None => model::detector(self.quality),
        }
    }

    /// The recognizer this run will load: the one it names, or the strongest
    /// the catalog has for its language.
    pub fn recognizer(&self) -> Result<&str, OcrError> {
        match &self.recognition {
            Some(named) => Ok(named),
            None => {
                Ok(model::recognizer_for(&self.language, self.quality)?.name)
            },
        }
    }
}

/// The published default for the longest side of the detector's input.
///
/// It is a low bar for a scan: an A4 page rendered at 300 dpi is 3508 pixels
/// tall, so this shrinks it by a factor of nearly four, and small type can stop
/// being found at all. Raising it costs time roughly in proportion to the area.
///
/// How much it costs to leave it here depends on which detector is running.
/// The small one finds several lines more of a dense page at 1920 than at 960;
/// the large one — what [`Quality::Best`] picks — finds at 960 about what the
/// small one finds at 1920, so raising it there is paying twice for the same
/// lines.
pub const DEFAULT_LIMIT_SIDE_LEN: usize = 960;

/// The published confidence floor for keeping a line.
pub const DEFAULT_DROP_SCORE: f32 = 0.5;

/// A loaded engine.
pub struct Engine {
    detector: Detector,
    /// The detector's declared thresholds with the run's own laid over them.
    params: Params,
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
        let detection = options.detector().to_string();
        let recognition = options.recognizer()?.to_string();

        let detection_dir =
            download::resolve(models_dir, &detection, &mut *progress)?;
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

        let (detector, params) = {
            let artifact = Artifact::load(&detection_dir.dir)?;
            let config = ModelConfig::load(&detection_dir.dir)?;
            // The detector's own description carries the thresholds it was
            // calibrated with; a caller that named none gets those rather
            // than another generation's.
            let PostProcess::Db(published) = &config.post else {
                return Err(OcrError::Artifact(format!(
                    "`{detection}` is not a text detector"
                )));
            };
            let params = Params::resolved(published, options.thresholds);
            (Detector::load(&Loader::new(&artifact, device))?, params)
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
            params,
            recognizer,
            labels,
            classifier,
            detection_name: detection_dir
                .name
                .map(str::to_string)
                .unwrap_or(detection),
            recognition_name: recognition,
            options,
        })
    }

    /// The models this engine is running, for the result envelope.
    pub fn models(&self) -> (&str, &str) {
        (&self.detection_name, &self.recognition_name)
    }

    pub fn options(&self) -> &Options { &self.options }

    /// The post-processing thresholds this engine resolved: the detector's own
    /// unless the run named otherwise.
    pub fn params(&self) -> &Params { &self.params }

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
        self.read_page_marked(page, number, source, &[])
    }

    /// Reads one page image, with a layout model's regions.
    ///
    /// The regions do one thing here that the geometry cannot do afterwards:
    /// they take apart a line the detector glued across a boundary between
    /// them — the two columns of a page set close together, most often — and
    /// have each piece read on its own ([`region::split`]). Everything else
    /// the labels are good for happens later, on the result.
    ///
    /// An empty `marked` is the same run as [`read_page`](Self::read_page).
    pub fn read_page_marked(
        &self,
        page: &RawPage,
        number: usize,
        source: &str,
        marked: &[Region],
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
            &self.params,
        );
        sort_boxes(&mut boxes);
        let cuts = straddling(&mut boxes, marked);

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

        // A cut line was read whole *and* in pieces, and only one of the two
        // survives. The pieces win when every one of them came back — a
        // straddling line normally reads better in pieces than glued, since
        // neither half is two columns of text any more — and the whole line
        // stands when any piece did not, because a page must not lose text to
        // a guess about its structure.
        let mut dropped = vec![false; boxes.len()];
        for cut in &cuts {
            let read = cut.pieces.clone().all(|at| {
                readings[at].score >= self.options.drop_score &&
                    !readings[at].text.trim().is_empty()
            });
            if read {
                dropped[cut.whole] = true;
            } else {
                cut.pieces.clone().for_each(|at| dropped[at] = true);
            }
        }

        let mut kept: Vec<(Line, Crop)> = Vec::with_capacity(readings.len());
        for (index, reading) in readings.into_iter().enumerate() {
            if dropped[index] || reading.score < self.options.drop_score {
                continue;
            }
            kept.push((
                Line {
                    text: reading.text,
                    score: reading.score,
                    quad: boxes[index].0,
                    rotated: rotated[index],
                    truncated: false,
                },
                std::mem::replace(&mut crops[index], Crop::empty()),
            ));
        }
        // The pieces were appended after every box the detector found, so they
        // have to take their places in the reading order. Sorting a list that
        // is already in it changes nothing.
        sort_in_reading_order(&mut kept, |(line, _)| line.quad);
        let (lines, kept_crops): (Vec<Line>, Vec<Crop>) =
            kept.into_iter().unzip();

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
                reflowed: false,
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
        self.read_file_marked(path, number, &[])
    }

    /// Reads a page image from a file, with a layout model's regions.
    pub fn read_file_marked(
        &self,
        path: &Path,
        number: usize,
        marked: &[Region],
    ) -> Result<Read, OcrError> {
        let page = RawPage::load(path)?;
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| path.display().to_string());
        self.read_page_marked(&page, number, &name, marked)
    }
}

/// A line the layout took apart: where the glued line sits among the detected
/// boxes, and where its pieces do.
struct Cut {
    /// The line as the detector drew it, spanning the boundary.
    whole: usize,
    /// Its pieces, appended after every box the detector found.
    pieces: std::ops::Range<usize>,
}

/// Cuts every box that straddles a boundary between two regions, appending the
/// pieces to `boxes` and reporting which came from what.
///
/// The glued line is left in place rather than replaced: both it and its pieces
/// go to the recognizer in the same batch, and which of them is kept is decided
/// on what came back. Reading one extra crop for each straddling line is a
/// page's worth of nothing — there are one or two of them where there are any
/// at all — and it saves the second pass a fallback would otherwise need.
fn straddling(boxes: &mut Vec<(Quad, f32)>, regions: &[Region]) -> Vec<Cut> {
    let mut cuts = Vec::new();
    if regions.is_empty() {
        return cuts;
    }
    // The range is fixed before the loop, so what the loop appends is not
    // itself looked at again.
    for index in 0..boxes.len() {
        let (quad, score) = boxes[index];
        let pieces = region::split(quad.bounds(), regions);
        if pieces.len() < 2 {
            continue;
        }
        let start = boxes.len();
        boxes.extend(
            pieces
                .iter()
                .map(|(from, to)| (quad.slice(*from, *to), score)),
        );
        cuts.push(Cut {
            whole: index,
            pieces: start..boxes.len(),
        });
    }
    cuts
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
    sort_in_reading_order(boxes, |(quad, _)| *quad);
}

/// [`sort_boxes`], for anything a box can be carried in.
fn sort_in_reading_order<T>(items: &mut [T], quad: impl Fn(&T) -> Quad) {
    let corner = |item: &T| quad(item).points[0];
    items.sort_by(|a, b| {
        let (ax, ay) = corner(a);
        let (bx, by) = corner(b);
        ay.total_cmp(&by).then(ax.total_cmp(&bx))
    });
    for i in 1..items.len() {
        let mut j = i;
        while j > 0 {
            let (previous, current) =
                (corner(&items[j - 1]), corner(&items[j]));
            if (previous.1 - current.1).abs() < 10.0 && current.0 < previous.0 {
                items.swap(j - 1, j);
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
