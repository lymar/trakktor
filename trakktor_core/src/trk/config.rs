use std::path::{Path, PathBuf};

use bon::bon;

pub struct TrkConfig {
    pub(crate) js_file: PathBuf,
    pub(crate) work_dir: PathBuf,
}

#[bon]
impl TrkConfig {
    #[builder]
    pub fn new<T: AsRef<Path>>(js_file: T, work_dir: Option<PathBuf>) -> Self {
        let work_dir = work_dir.unwrap_or_else(|| {
            std::env::current_dir()
                .expect("Failed to get current working directory")
        });
        Self {
            js_file: js_file.as_ref().to_path_buf(),
            work_dir,
        }
    }
}
