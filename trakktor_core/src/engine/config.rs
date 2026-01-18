use std::path::PathBuf;

use bon::Builder;

#[derive(Builder)]
#[builder(on(String, into))]
pub struct EngineConfig {
    #[builder(into)]
    pub js_file: PathBuf,
}
