use std::{fmt, sync::Arc};

use mime_guess::{Mime, mime};

pub mod audio;

pub struct PreviewData {
    pub data: Arc<[u8]>,
    pub mime: Mime,
    pub text: Option<String>,
}

impl fmt::Debug for PreviewData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let bs = bytesize::ByteSize::b(self.data.len() as u64);
        f.debug_struct("PreviewData")
            .field("data", &bs)
            .field("mime", &self.mime)
            .field("text", &self.text)
            .finish()
    }
}

pub async fn render_preview(preview: PreviewData) -> anyhow::Result<()> {
    log::info!(preview:?; "Rendering preview");

    if preview.mime.type_() == mime::AUDIO {
        audio::render_audio_preview(preview).await?;
    } else {
        anyhow::bail!("Unsupported mime type for preview: {}", preview.mime);
    }

    Ok(())
}
