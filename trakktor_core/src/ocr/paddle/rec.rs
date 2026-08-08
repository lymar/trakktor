//! The text recognizer: one of two networks, chosen by the artifact itself.
//!
//! A recognizer takes a straightened text-line crop and returns one
//! probability distribution per time step, which a greedy CTC pass turns into
//! text. Either network is a backbone whose strides work on one axis at a time
//! — reading a line means keeping its length and spending its height — under
//! the [small transformer and the projection](head) both of them share.
//!
//! What differs is the backbone and what it costs. The [small](mobile) one is
//! the PP-LCNetV3 family — eight megabytes for an alphabet-bound model — and
//! eleven of the catalog's twelve alphabets are published in that size alone.
//! The [large](server) one is the PP-HGNetV2 the large detector carries, and
//! upstream publishes it for the Han alphabet only, where it costs eighty-four
//! megabytes against seventeen and about two tenths of a second more per line.
//!
//! Which of the two an artifact holds is not a property of its name — a caller
//! may hand `--rec-model` a directory — so it is taken from the graph, which
//! names its own backbone.
//!
//! Four things about a recognizer are easy to get wrong and are pinned here:
//!
//! - **The height is exactly 48.** Both backbones halve it four times and the
//!   closing pooling divides what is left by three, so 48 is the one height
//!   that leaves a single row for the sequence to be read off. Any other height
//!   leaves a feature map that cannot be a sequence at all.
//! - **A crop is resized to a width its batch chooses, not its own.** The whole
//!   batch is padded to one width, and a crop read across a wide, mostly empty
//!   tensor reads differently from the same crop read across a snug one. The
//!   batching policy is therefore part of the result rather than an
//!   implementation detail; it lives in [`Recognizer::read`].
//! - **The padding is mid-grey, not black.** Crops are normalized to `[-1, 1]`
//!   and the padding is left at zero, which is where the value 127 lands.
//! - **The softmax is inside the exported graph.** [`Recognizer::forward`]
//!   returns probabilities; a second softmax would flatten them and quietly
//!   ruin every confidence the pipeline reports.

pub mod head;
pub mod mobile;
pub mod server;
#[cfg(test)]
mod tests;

use candle_core::{Device, Tensor};

use self::head::Head;
use super::{
    artifact::Artifact,
    crop::Crop,
    image::{CHANNELS, resize_linear},
    net::Loader,
};
use crate::ocr::error::OcrError;

/// The height every crop is resized to.
pub const HEIGHT: usize = 48;

/// The width a batch is padded to when nothing in it is wider than the shape
/// the models were exported for — 320 stands for an aspect ratio of `320 / 48`.
pub const WIDTH: usize = 320;

/// The widest a batch is ever padded to. A pathological crop — a printed rule
/// mistaken for a line of text — would otherwise size the whole batch, and the
/// exported graphs are not built for that either.
pub const MAX_WIDTH: usize = 3200;

/// Crops per forward pass.
pub const BATCH: usize = 6;

/// The CTC blank: class zero, the only ignored class, and it emits nothing.
pub const BLANK: usize = 0;

/// The narrowest input the network can read — below this the closing pooling
/// has nothing to pool. Every batch is far wider, but the check is cheap and
/// the alternative is an unhelpful failure deep inside a convolution.
const MIN_WIDTH: usize = 8;

/// The backbone each network declares itself with.
const MOBILE_BACKBONE: &str = "PPLCNetV3";
const SERVER_BACKBONE: &str = "PPHGNetV2";

/// A loaded recognizer. Both backbones are boxed: a network holds its own
/// weights, so the two differ in size by more than an enum should carry.
#[derive(Debug)]
enum Backbone {
    Mobile(Box<mobile::Net>),
    Server(Box<server::Net>),
}

impl Backbone {
    fn forward(&self, x: &Tensor) -> Result<Tensor, OcrError> {
        match self {
            Self::Mobile(net) => net.forward(x),
            Self::Server(net) => net.forward(x),
        }
    }
}

/// The recognizer: a backbone, the head that reads its output as a sequence,
/// and the device they live on.
#[derive(Debug)]
pub struct Recognizer {
    device: Device,
    backbone: Backbone,
    head: Head,
}

impl Recognizer {
    /// Loads whichever of the two the artifact holds.
    pub fn load(loader: &Loader) -> Result<Self, OcrError> {
        let artifact = loader.artifact();
        let classes = artifact.output_classes().ok_or_else(|| {
            OcrError::Artifact(
                "the recognizer does not declare how many classes it reads"
                    .into(),
            )
        })?;

        let (backbone, naming) = if artifact.has_module(MOBILE_BACKBONE) {
            (
                Backbone::Mobile(Box::new(mobile::Net::load(loader)?)),
                mobile::NAMING,
            )
        } else if artifact.has_module(SERVER_BACKBONE) {
            (
                Backbone::Server(Box::new(server::Net::load(loader)?)),
                server::NAMING,
            )
        } else {
            return Err(unknown_backbone(artifact));
        };

        Ok(Self {
            device: loader.device().clone(),
            backbone,
            head: Head::load(loader, naming, classes)?,
        })
    }

    /// How many classes the head reads, the blank and the space included.
    pub fn classes(&self) -> usize { self.head.classes() }

    /// The device the weights live on.
    pub fn device(&self) -> &Device { &self.device }

    /// Runs the network over a batch of normalized crops.
    ///
    /// The input is `[batch, 3, 48, width]` in `[-1, 1]`, blue channel first;
    /// the result is `[batch, steps, classes]` of probabilities — the softmax
    /// is inside the exported graph, so do not apply it again.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor, OcrError> {
        let (_, _, height, width) = x.dims4()?;
        if height != HEIGHT {
            return Err(OcrError::Runtime(format!(
                "the recognizer reads crops {HEIGHT} pixels tall, got {height}"
            )));
        }
        if width < MIN_WIDTH {
            return Err(OcrError::Runtime(format!(
                "the recognizer needs at least {MIN_WIDTH} pixels of width, \
                 got {width}"
            )));
        }

        let pooled = self.backbone.forward(x)?;
        let logits = self.head.forward(&pooled)?;
        Ok(candle_nn::ops::softmax_last_dim(&logits)?)
    }

    /// Reads a page's crops and returns one reading per crop, in the order the
    /// crops were given.
    ///
    /// Crops reach the network sorted by aspect ratio, in groups of [`BATCH`],
    /// because a group is padded to a single width and that width changes what
    /// the network reads: pack a short crop together with a long one and the
    /// short one is read across a tensor that is mostly padding. Sorting keeps
    /// crops of similar shape together, and the sort is stable, so crops of
    /// equal ratio keep page order and a run repeats itself.
    ///
    /// A crop with no pixels is never read; its reading is the empty one.
    pub fn read(
        &self,
        crops: &[Crop],
        labels: &Labels,
    ) -> Result<Vec<Reading>, OcrError> {
        if labels.len() != self.classes() {
            return Err(OcrError::Artifact(format!(
                "the label table holds {} classes but the recognizer reads {}",
                labels.len(),
                self.classes()
            )));
        }
        let mut readings = vec![Reading::default(); crops.len()];
        let mut order: Vec<usize> =
            (0..crops.len()).filter(|i| readable(&crops[*i])).collect();
        order.sort_by(|a, b| {
            ratio(&crops[*a])
                .partial_cmp(&ratio(&crops[*b]))
                .expect("a readable crop has a finite aspect ratio")
        });

        for group in order.chunks(BATCH) {
            let batch: Vec<&Crop> = group.iter().map(|i| &crops[*i]).collect();
            let input = batch_tensor(&batch, &self.device)?;
            let probabilities = self.forward(&input)?;
            let (_, steps, classes) = probabilities.dims3()?;
            let flat = probabilities.flatten_all()?.to_vec1::<f32>()?;
            for (row, index) in group.iter().enumerate() {
                let from = row * steps * classes;
                let rows = &flat[from..from + steps * classes];
                readings[*index] = decode(rows, classes, labels);
            }
        }
        Ok(readings)
    }
}

/// A recognizer this port does not implement — a newer generation, most
/// likely, since the two here share their published names with everything else
/// in the family.
fn unknown_backbone(artifact: &Artifact) -> OcrError {
    OcrError::Artifact(format!(
        "this recognizer is built on {}, and trakktor runs {MOBILE_BACKBONE} \
         and {SERVER_BACKBONE}",
        artifact
            .modules()
            .first()
            .map_or("no module the graph names", String::as_str),
    ))
}

/// Whether a crop can be read at all: it has an extent, and enough bytes to
/// back the extent it claims.
///
/// A quadrangle that straightens into nothing is the caller's to drop, but a
/// resize must never be asked to read past the end of one.
pub(super) fn readable(crop: &Crop) -> bool {
    crop.width > 0 &&
        crop.height > 0 &&
        crop.bgr.len() >= crop.width * crop.height * CHANNELS
}

/// How many times wider than tall a crop is. Meaningless for a crop that is
/// not [`readable`], which is why nothing asks one.
fn ratio(crop: &Crop) -> f64 { crop.width as f64 / crop.height as f64 }

/// The width a batch of crops is padded to.
///
/// The widest crop in the batch decides, but never below the shape the models
/// were exported for and never above [`MAX_WIDTH`]. The multiplication
/// truncates, so an aspect ratio a hair under a whole pixel does not buy one.
pub fn batch_width(crops: &[&Crop]) -> usize {
    let mut widest = WIDTH as f64 / HEIGHT as f64;
    for crop in crops {
        if readable(crop) {
            widest = widest.max(ratio(crop));
        }
    }
    ((HEIGHT as f64 * widest) as usize).min(MAX_WIDTH)
}

/// Turns a batch of crops into the tensor the network reads: `[batch, 3, 48,
/// width]`, normalized to `[-1, 1]`, blue channel first.
///
/// A crop keeps its aspect ratio until it would run past the batch width, at
/// which point it is squeezed to fit; whatever is left over on the right stays
/// at zero, which on this scale is mid-grey rather than black.
pub fn batch_tensor(
    crops: &[&Crop],
    device: &Device,
) -> Result<Tensor, OcrError> {
    let width = batch_width(crops);
    let mut data = vec![0f32; crops.len() * CHANNELS * HEIGHT * width];
    for (n, crop) in crops.iter().enumerate() {
        if !readable(crop) {
            continue;
        }
        let scaled = (HEIGHT as f64 * ratio(crop)).ceil() as usize;
        let taken = scaled.clamp(1, width);
        let resized = resize_linear(
            &crop.bgr,
            crop.width,
            crop.height,
            CHANNELS,
            taken,
            HEIGHT,
        );
        for y in 0..HEIGHT {
            for x in 0..taken {
                for c in 0..CHANNELS {
                    let mut value =
                        f32::from(resized[(y * taken + x) * CHANNELS + c]) /
                            255.0;
                    value -= 0.5;
                    value /= 0.5;
                    data[((n * CHANNELS + c) * HEIGHT + y) * width + x] = value;
                }
            }
        }
    }
    let shape = (crops.len(), CHANNELS, HEIGHT, width);
    Ok(Tensor::from_vec(data, shape, device)?)
}

/// What the recognizer made of one crop.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Reading {
    pub text: String,
    /// Mean probability of the steps that were kept, and zero when none were.
    pub score: f32,
}

/// The class table a CTC recognizer decodes against: the blank, the model's
/// own character dictionary, and the space.
///
/// The order is not a convention this port chose — the models were trained
/// against it. The blank is class zero and emits nothing, the dictionary
/// follows in file order, and the space is appended *after* the dictionary
/// rather than sorted into it, so it is always the last class. Getting the
/// order wrong shifts every character the model emits.
#[derive(Debug, Clone)]
pub struct Labels {
    table: Vec<String>,
}

impl Labels {
    /// Builds the table from a model's own dictionary and checks it against
    /// the class count its graph declares.
    ///
    /// The two live in different files, and a mismatch means they do not
    /// belong together — worth reporting here rather than as text that decodes
    /// into the wrong alphabet.
    pub fn new(
        characters: &[String],
        classes: usize,
    ) -> Result<Self, OcrError> {
        if characters.len() + 2 != classes {
            return Err(OcrError::Artifact(format!(
                "the character dictionary holds {} entries, which with the \
                 blank and the space makes {}, but the recognizer reads {} \
                 classes",
                characters.len(),
                characters.len() + 2,
                classes
            )));
        }
        let mut table = Vec::with_capacity(classes);
        table.push(String::new());
        table.extend(characters.iter().cloned());
        table.push(" ".to_string());
        Ok(Self { table })
    }

    /// How many classes the table covers.
    pub fn len(&self) -> usize { self.table.len() }

    pub fn is_empty(&self) -> bool { self.table.is_empty() }

    /// What a class emits. The blank, and anything the table does not cover,
    /// emits nothing.
    pub fn text(&self, class: usize) -> &str {
        self.table
            .get(class)
            .map(String::as_str)
            .unwrap_or_default()
    }
}

/// Greedy CTC decoding of one crop's probabilities, laid out step after step.
///
/// At every step the likeliest class wins, ties going to the lower class the
/// way the reference's arg-max does. Repeats collapse *before* blanks are
/// dropped — so two identical steps in a row are one character, while the same
/// two separated by a blank are two — and the score is the mean probability of
/// the steps that survive both rules.
pub fn decode(
    probabilities: &[f32],
    classes: usize,
    labels: &Labels,
) -> Reading {
    let mut text = String::new();
    let mut total = 0f64;
    let mut kept = 0usize;
    let mut previous = usize::MAX;
    for step in probabilities.chunks_exact(classes) {
        let mut best = 0usize;
        for (class, probability) in step.iter().enumerate() {
            if *probability > step[best] {
                best = class;
            }
        }
        let repeated = best == previous;
        previous = best;
        if repeated || best == BLANK {
            continue;
        }
        text.push_str(labels.text(best));
        total += f64::from(step[best]);
        kept += 1;
    }
    Reading {
        text,
        score: if kept == 0 {
            0.0
        } else {
            (total / kept as f64) as f32
        },
    }
}
