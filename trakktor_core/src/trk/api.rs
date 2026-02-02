use crate::{artifact::Artifact, trk::Trk};

impl Trk {
    pub async fn read_file(&self, path: &str) -> anyhow::Result<Artifact> {
        let file_path = self.cfg.work_dir.join(path);
        if !file_path.exists() {
            anyhow::bail!("File not found: {}", file_path.display());
        }
        let content = tokio::fs::read(&file_path).await?;
        let mime = mime_guess::from_path(&file_path)
            .first_or_octet_stream()
            .to_string();
        let artifact = Artifact::builder()
            .data(content)
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
}
