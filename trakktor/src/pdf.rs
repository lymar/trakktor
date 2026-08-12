//! `pdf cut`: a thin driver over `trakktor_core::pdf`.

use std::path::{Path, PathBuf};

use trakktor_core::{pages::Selection, pdf};

use crate::{cli::PdfCutArgs, error::CliError};

/// Cuts the selected pages into a new PDF and prints the result.
pub(crate) fn run_cut(
    args: &PdfCutArgs,
    json: bool,
    pretty: bool,
) -> Result<(), CliError> {
    // A page spec that does not parse should not cost the file being read.
    let pages = Selection::parse(&args.pages).map_err(pdf::PdfError::from)?;

    let out = args.out.clone().unwrap_or_else(|| default_out(&args.file));
    let cut = pdf::cut(
        &args.file,
        &out,
        &pdf::Options {
            pages,
            password: args.password.clone(),
        },
    )?;

    let source = args
        .file
        .file_name()
        .unwrap_or(args.file.as_os_str())
        .to_string_lossy();
    crate::output::print_pdf_cut(&cut, &source, &out, json, pretty);
    Ok(())
}

/// The output path when `--out` does not name one: the input's name with
/// `.cut.pdf`, in the current directory.
fn default_out(input: &Path) -> PathBuf {
    let stem = input
        .file_stem()
        .unwrap_or(input.as_os_str())
        .to_string_lossy();
    PathBuf::from(format!("{stem}.cut.pdf"))
}
