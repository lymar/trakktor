//! `convert pdf`: a thin driver over `trakktor_core::convert`.

use trakktor_core::convert::{
    ConvertError,
    pdf::{self, Selection},
};

use crate::{cli::ConvertPdfArgs, error::CliError};

/// Converts the text layer of a PDF and prints the result.
pub(crate) fn run_pdf(
    args: &ConvertPdfArgs,
    json: bool,
    pretty: bool,
) -> Result<(), CliError> {
    // A page range that does not parse should not cost the file being read.
    let pages = args
        .pages
        .as_deref()
        .map(Selection::parse)
        .transpose()
        .map_err(ConvertError::from)?;

    let converted = pdf::convert(
        &args.file,
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
    crate::output::print_convert_pdf(
        &converted,
        &source,
        args.out.as_deref(),
        json,
        pretty,
    )
    .map_err(CliError::from)?;
    Ok(())
}

/// Writes the Markdown where `--out` asks for it.
pub(crate) fn write_out(
    path: &std::path::Path,
    markdown: &str,
) -> Result<(), ConvertError> {
    std::fs::write(path, markdown).map_err(|source| ConvertError::Io {
        path: path.display().to_string(),
        source,
    })
}
