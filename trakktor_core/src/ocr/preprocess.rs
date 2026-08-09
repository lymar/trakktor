//! Page preprocessing: turning a photograph into something a scanner could
//! have produced.
//!
//! A scan and a photograph of the same page differ in five ways, and this
//! stage answers two of them: **which way up the page is**, and **what shape
//! it is**. The other three — uneven light, the desk around the sheet, a
//! spread that is really two pages — are not handled here, and saying so is
//! part of the contract; see the domain's design notes for what each costs.
//!
//! Both models are ports of PaddleOCR's document pre-processor and both are
//! off by default, because on a scan they do nothing: the classifier answers
//! "upright" and the unwarper reproduces the page. They earn their time only
//! on photographs.
//!
//! ## The order, and why
//!
//! Orientation first, unwarping second, which is upstream's order too. The
//! classifier was trained on whole upright pages and reads a page lying on its
//! side as one of four right angles; the unwarper has no such prior and
//! straightens a page whichever way it faces. Turning first also costs
//! nothing: a right-angle turn moves whole pixels and resamples nothing, so
//! the unwarper still sees the photograph rather than a resampled copy of it.
//!
//! ## What comes out, and where the boxes are
//!
//! Straightening moves every pixel, so a box found afterwards is a box of the
//! straightened page and not of the photograph the caller handed in. The stage
//! therefore returns a [`Prepared`] page **together with the map back**, and
//! the engines put every quadrangle they report through it. What a caller gets
//! in `quad` is always a place on their own file — which is the only answer
//! that is any use for pointing at the original, and the only one that
//! survives being told the page was preprocessed at all.

#[cfg(test)]
mod tests;

pub mod download;
pub mod field;
pub mod model;
pub mod orientation;
pub mod sheet;
pub mod unwarp;

use std::path::Path;

pub use field::Backmap;
pub use orientation::Reading;
pub use sheet::Sheet;

use crate::{
    download::Progress,
    ocr::{
        error::OcrError,
        paddle::{artifact::Artifact, image::Page, net::Loader},
        page::Quad,
    },
};

/// What the caller asked the stage to do.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Options {
    /// Turn the page upright first.
    pub orientation: bool,
    /// Straighten the page.
    pub unwarp: bool,
    /// Find the sheet in the frame and cut it out first.
    pub sheet: bool,
}

impl Options {
    /// Whether anything at all is asked for.
    pub fn any(self) -> bool { self.orientation || self.unwarp || self.sheet }

    /// What the stage has yet to download, in the shape the engines announce
    /// their own downloads in.
    pub fn pending(self, models_dir: &Path) -> Vec<(&'static str, u64)> {
        let mut pending = Vec::new();
        if self.orientation {
            pending.extend(download::pending(models_dir, model::ORIENTATION));
        }
        if self.unwarp {
            pending.extend(download::pending(models_dir, model::UNWARP));
        }
        pending
    }
}

/// A page after the stage has had it, and everything needed to talk about the
/// original again.
#[derive(Debug)]
pub struct Prepared {
    /// The page the rest of the pipeline reads.
    pub page: Page,
    /// The turn that was applied, when one was.
    pub turn: Option<Reading>,
    /// The sheet that was cut out of the frame, when one was found.
    pub sheet: Option<Sheet>,
    /// The map from the straightened page back to the photograph, when the
    /// page was straightened.
    pub backmap: Option<Backmap>,
    /// The size of the page as it came in, which stays the page's size in the
    /// output whatever the stage did to it.
    pub original: (u32, u32),
}

impl Prepared {
    /// A page nothing was done to.
    pub fn untouched(page: Page) -> Self {
        let original = (page.width, page.height);
        Self {
            page,
            turn: None,
            sheet: None,
            backmap: None,
            original,
        }
    }

    /// Whether the stage changed anything at all.
    pub fn changed(&self) -> bool {
        self.turn.is_some() || self.sheet.is_some() || self.backmap.is_some()
    }

    /// Puts a read page back on the file the caller handed in: its size, and
    /// every line's quadrangle.
    ///
    /// A caller who preprocessed a page still gets boxes on their own
    /// photograph, and a size that matches the file they can open. Everything
    /// that had to happen in the straightened page's coordinates — cropping a
    /// line, cutting out an illustration, sorting into reading order — has
    /// happened by the time this is called.
    pub fn relocate(&self, page: &mut crate::ocr::page::Page) {
        if !self.changed() {
            return;
        }
        let (width, height) = self.original;
        page.width = width;
        page.height = height;
        for line in &mut page.lines {
            line.quad = self.locate(&line.quad);
        }
    }

    /// Puts a quadrangle of the prepared page back on the page the caller
    /// handed in.
    ///
    /// Both steps are undone in the order they were applied, back to front:
    /// the straightening first, because it is what the quadrangle is
    /// expressed in, and then the turn.
    pub fn locate(&self, quad: &Quad) -> Quad {
        let quad = match &self.backmap {
            None => *quad,
            Some(backmap) => backmap.unmap(quad),
        };
        let quad = match &self.sheet {
            None => quad,
            Some(sheet) => {
                let mut points = quad.points;
                for point in &mut points {
                    *point = sheet.source(point.0, point.1);
                }
                Quad::new(points)
            },
        };
        match self.turn {
            None => quad,
            Some(Reading { turn, .. }) => {
                let (width, height) = self.original;
                let mut points = quad.points;
                for point in &mut points {
                    *point = orientation::unturn(
                        *point,
                        turn,
                        width as f32,
                        height as f32,
                    );
                }
                Quad::new(points)
            },
        }
    }
}

/// The stage: the models it was asked for, loaded once and run per page.
#[derive(Debug)]
pub struct Preprocessor {
    orientation: Option<orientation::Classifier>,
    sheet: bool,
    unwarp: Option<unwarp::Unwarper>,
}

impl Preprocessor {
    /// Loads whichever models the options ask for, downloading them on first
    /// use. With nothing asked for, nothing is loaded.
    pub fn load(
        models_dir: &Path,
        options: Options,
        device: &candle_core::Device,
        progress: Progress<'_>,
    ) -> Result<Self, OcrError> {
        let orientation = if options.orientation {
            let resolved = download::resolve(
                models_dir,
                model::ORIENTATION,
                &mut *progress,
            )?;
            let artifact = Artifact::load(&resolved.dir)?;
            let loader = Loader::new(&artifact, device);
            Some(orientation::Classifier::load(&loader)?)
        } else {
            None
        };
        let unwarp = if options.unwarp {
            let resolved =
                download::resolve(models_dir, model::UNWARP, &mut *progress)?;
            let artifact = Artifact::load(&resolved.dir)?;
            let loader = Loader::new(&artifact, device);
            Some(unwarp::Unwarper::load(&loader)?)
        } else {
            None
        };
        Ok(Self {
            orientation,
            sheet: options.sheet,
            unwarp,
        })
    }

    /// Whether this preprocessor does anything.
    pub fn idle(&self) -> bool {
        self.orientation.is_none() && !self.sheet && self.unwarp.is_none()
    }

    /// Runs the stage over one page.
    pub fn run(&self, page: Page) -> Result<Prepared, OcrError> {
        let original = (page.width, page.height);
        let mut prepared = Prepared::untouched(page);

        if let Some(classifier) = &self.orientation {
            if let Some(reading) = classifier.read(&prepared.page)? {
                if reading.turn != 0 {
                    prepared.page =
                        orientation::turn(&prepared.page, reading.turn);
                    prepared.turn = Some(reading);
                }
            }
        }

        // The sheet is cut out before the page is straightened, and the order
        // is not a preference: the unwarper reads the whole frame, so a sheet
        // that is a third of the picture reaches it as a third of 712 by 488
        // and comes back no straighter than it went in.
        if self.sheet {
            if let Some(found) = sheet::find(&prepared.page) {
                prepared.page = found.apply(&prepared.page);
                prepared.sheet = Some(found);
            }
        }

        if let Some(unwarper) = &self.unwarp {
            let field = unwarper.field(&prepared.page)?;
            let backmap = Backmap::new(
                field,
                prepared.page.width as usize,
                prepared.page.height as usize,
            );
            prepared.page = backmap.apply(&prepared.page);
            prepared.backmap = Some(backmap);
        }

        prepared.original = original;
        Ok(prepared)
    }
}
