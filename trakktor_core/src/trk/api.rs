use std::sync::Arc;

use crate::{artifact::Artifact, preview::{PreviewData, render_preview}, trk::Trk};

impl Trk {
    pub async fn read_file(&self, path: &str) -> anyhow::Result<Artifact> {
        let file_path = self.cfg.work_dir.join(path);
        if !file_path.exists() {
            anyhow::bail!("File not found: {}", file_path.display());
        }
        let content = tokio::fs::read(&file_path).await?;
        let mime = mime_guess::from_path(&file_path)
            .first_or_octet_stream();
        let artifact = Artifact::builder()
            .data(content.into())
            .mime(mime)
            .original_name(
                file_path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("unknown")
                    .to_string(),
            )
            .build();
        log::debug!(artifact:?, path, file_path:?; 
            "read_file: loaded successfully");
        Ok(artifact)
    }

    pub async fn preview(&self, artifact: &Artifact, text: Option<&str>) -> anyhow::Result<()> {
        let _lock = self.preview_lock.lock().await;
        
        render_preview(PreviewData {
            data: Arc::clone(&artifact.data),
            mime: artifact.mime.clone().unwrap_or(
                mime_guess::mime::APPLICATION_OCTET_STREAM
            ),
            text: text.map(|s| s.to_string()),
        }).await?;
        Ok(())
    }
}
