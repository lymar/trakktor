//! The stage end to end: a model directory in, labelled regions out.
//!
//! What the network returns is three hundred boxes in the coordinates of the
//! square it was fed, with a logit per class. Turning those into regions is the
//! graph's own tail, done here on the host where it can be read: the sigmoid,
//! the ranking *across* classes (a query may win under one label and lose under
//! another, so the top three hundred are taken over the whole `300 × classes`
//! grid rather than per query), the centre-size-to-corners conversion and the
//! scaling back onto the page.

#[cfg(test)]
mod tests;

use std::path::Path;

use super::{
    config::LayoutConfig,
    download::{self, Resolved},
    image::{self, Prepared},
    model,
    post::{self, Raw},
    region::Region,
};
use crate::{
    download::Progress,
    ocr::{
        error::OcrError,
        paddle::{
            artifact::Artifact, image::Page as RawPage, net::Loader,
            pipeline::Device,
        },
    },
};

/// Which runtime runs the network.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Runtime {
    Candle,
    Burn,
}

/// How a run of this stage is configured.
#[derive(Debug, Clone)]
pub struct Options {
    /// Model name, or a path to a directory of artifacts.
    pub model: String,
    /// Post-processing thresholds.
    pub post: post::Settings,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            model: model::DEFAULT_MODEL.to_string(),
            post: post::Settings::default(),
        }
    }
}

/// The loaded stage.
pub struct Detector {
    net: Net,
    side: usize,
    name: String,
    options: Options,
}

/// The network, on whichever runtime was asked for.
enum Net {
    Candle(super::net::Net),
    #[cfg(feature = "ocr-burn")]
    Burn(super::net_burn::Net),
}

impl Detector {
    /// Resolves the model named by `options`, downloading it on first use, and
    /// loads it onto `device`.
    pub fn load(
        models_dir: &Path,
        device: Device,
        runtime: Runtime,
        options: Options,
        progress: Progress<'_>,
    ) -> Result<Self, OcrError> {
        let resolved: Resolved =
            download::resolve(models_dir, &options.model, progress)?;
        let config = LayoutConfig::load(&resolved.dir)?;
        config.check_labels()?;
        let side = config.target_size[0];
        if config.target_size[0] != config.target_size[1] {
            return Err(OcrError::Artifact(format!(
                "this model wants a {}×{} input; the port implements the \
                 square one it was exported with",
                config.target_size[1], config.target_size[0]
            )));
        }

        let net = match runtime {
            Runtime::Candle => {
                let artifact = Artifact::load(&resolved.dir)?;
                let device = device.resolve()?;
                let loader = Loader::new(&artifact, &device);
                Net::Candle(super::net::Net::load(
                    &loader,
                    side,
                    config.labels.len(),
                )?)
            },
            Runtime::Burn => burn_net(&resolved.dir, device, side, &config)?,
        };

        Ok(Self {
            net,
            side,
            name: resolved
                .name
                .map(str::to_string)
                .unwrap_or_else(|| config.model_name.clone()),
            options,
        })
    }

    /// The model this stage is running, for the result envelope.
    pub fn model(&self) -> &str { &self.name }

    pub fn options(&self) -> &Options { &self.options }

    /// Marks up one page.
    pub fn detect(&self, page: &RawPage) -> Result<Vec<Region>, OcrError> {
        let prepared = image::prepare(page, self.side);
        let raw = self.run(&prepared)?;
        Ok(post::regions(
            &raw,
            (page.width, page.height),
            &self.options.post,
        ))
    }

    /// Marks up a page image from a file.
    pub fn detect_file(&self, path: &Path) -> Result<Vec<Region>, OcrError> {
        self.detect(&RawPage::load(path)?)
    }

    /// Runs the network and decodes its output into page-pixel boxes.
    fn run(&self, prepared: &Prepared) -> Result<Vec<Raw>, OcrError> {
        let prediction = match &self.net {
            Net::Candle(net) => {
                let input = super::net::input(
                    &prepared.data,
                    prepared.side,
                    net.device(),
                )?;
                net.forward(&input)?
            },
            #[cfg(feature = "ocr-burn")]
            Net::Burn(net) => net.forward(&prepared.data, prepared.side)?,
        };
        Ok(decode(&prediction, prepared))
    }
}

/// The graph's tail: rank every `(query, class)` pair, keep the best three
/// hundred, and put the boxes back on the page.
fn decode(
    prediction: &super::net::Prediction,
    prepared: &Prepared,
) -> Vec<Raw> {
    /// How many boxes the network is asked for, which is also how many pairs
    /// survive the ranking.
    const TOP: usize = 300;

    let classes = prediction.classes;
    let queries = prediction.boxes.len() / 4;
    let mut ranked: Vec<(usize, f32)> = prediction
        .logits
        .iter()
        .enumerate()
        .map(|(at, logit)| (at, 1.0 / (1.0 + (-logit).exp())))
        .collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
    ranked.truncate(TOP.min(ranked.len()));

    // The page the boxes belong on is the one the graph reconstructs from the
    // two shape inputs, and the rounding is the graph's.
    let (scale_h, scale_w) = prepared.scale_factor();
    let width = (prepared.side as f32 / scale_w + 0.5).floor();
    let height = (prepared.side as f32 / scale_h + 0.5).floor();

    ranked
        .into_iter()
        .filter_map(|(at, score)| {
            let query = at / classes;
            let class = at % classes;
            if query >= queries {
                return None;
            }
            let box_ = &prediction.boxes[query * 4..query * 4 + 4];
            let (cx, cy, w, h) = (box_[0], box_[1], box_[2], box_[3]);
            Some(Raw {
                class,
                score,
                bounds: (
                    (cx - 0.5 * w) * width,
                    (cy - 0.5 * h) * height,
                    (cx + 0.5 * w) * width,
                    (cy + 0.5 * h) * height,
                ),
            })
        })
        .collect()
}

/// The network on the burn runtime.
#[cfg(feature = "ocr-burn")]
fn burn_net(
    dir: &Path,
    device: Device,
    side: usize,
    config: &LayoutConfig,
) -> Result<Net, OcrError> {
    Ok(Net::Burn(super::net_burn::Net::load(
        dir,
        device,
        side,
        config.labels.len(),
    )?))
}

/// Without the `ocr-burn` feature there is no burn runtime to load.
#[cfg(not(feature = "ocr-burn"))]
fn burn_net(
    _dir: &Path,
    _device: Device,
    _side: usize,
    _config: &LayoutConfig,
) -> Result<Net, OcrError> {
    Err(OcrError::InvalidOptions(
        "this build has no burn runtime; install or build trakktor with the \
         `burn` feature"
            .into(),
    ))
}
