//! Operations on a PDF as a document: pages are the unit, the file is the
//! container.
//!
//! Nothing here reads or interprets content — extracting text belongs to
//! [`crate::convert`] (the text layer) and [`crate::ocr`] (recognition). What
//! lives here is document surgery: today, cutting a page range out into a new
//! PDF that carries everything those pages need — fonts, images, shared
//! resources — and nothing they do not.

mod catalog;
pub mod cut;
pub mod error;

pub use cut::{Cut, Options, cut};
pub use error::PdfError;

/// A document-level structure the cut had to drop, by its stable output name.
///
/// Deleting a page strips every reference to it, and a structure that pointed
/// at the page would survive that mutilated rather than whole. Better to drop
/// it honestly — and say so — than to hand over an outline whose entries lead
/// nowhere.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dropped {
    /// The bookmark tree (`/Outlines`): an entry led to a removed page.
    Outlines,
    /// User-facing page numbering (`/PageLabels`): labels number positions,
    /// and after a cut the positions have moved.
    PageLabels,
    /// The logical structure tree of a tagged PDF (`/StructTreeRoot`).
    StructTree,
    /// The document's "open at page N" action (`/OpenAction`).
    OpenAction,
    /// Interactive form fields whose widgets all sat on removed pages.
    FormFields,
    /// Named destinations that led to removed pages.
    NamedDestinations,
}

impl Dropped {
    /// The stable string that goes into the output.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Dropped::Outlines => "outlines",
            Dropped::PageLabels => "page_labels",
            Dropped::StructTree => "struct_tree",
            Dropped::OpenAction => "open_action",
            Dropped::FormFields => "form_fields",
            Dropped::NamedDestinations => "named_destinations",
        }
    }
}
