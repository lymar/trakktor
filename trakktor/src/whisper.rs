use serde::Serialize;

use crate::{aws::batch::ContainerEnvs, job::JobUid};

const VERSION_TAG: &str = "1";
const DEV_VERSION_TAG: &str = "dev";
const IMAGE_NAME: &str = "ghcr.io/lymar/trakktor/whisper";
const LARGE_MODEL: &str = "large-v3";

#[derive(Debug, Clone, Copy)]
pub enum Model {
    Large,
}

impl Model {
    pub fn get_name(&self) -> &str {
        match self {
            Model::Large => LARGE_MODEL,
        }
    }
}

pub fn make_image_name(model: Model, is_dev: bool) -> String {
    format!(
        "{}:{}-{}",
        IMAGE_NAME,
        model.get_name(),
        if is_dev { DEV_VERSION_TAG } else { VERSION_TAG }
    )
}

/// Arguments for a Whisper job passed to the container as environment
/// variables.
#[derive(Debug, Serialize)]
pub struct WhisperJobArgs<'a> {
    #[serde(rename = "TRK_JOB_UID")]
    pub job_uid: &'a JobUid,
    #[serde(rename = "TRK_INPUT_FILE")]
    pub input_file: &'a str,
    #[serde(rename = "TRK_LANGUAGE")]
    pub language: &'a str,
}

impl<'a> WhisperJobArgs<'a> {
    /// Convert the arguments into a list of environment variables.
    pub fn environments(&self) -> ContainerEnvs {
        let sv = serde_json::to_value(&self).expect("Failed to serialize");
        let serde_json::Value::Object(vm) = sv else {
            panic!("Expected object");
        };
        ContainerEnvs(
            vm.into_iter()
                .map(|(k, v)| {
                    let serde_json::Value::String(vs) = v else {
                        panic!("Expected string");
                    };
                    (k, vs)
                })
                .collect::<Vec<_>>(),
        )
    }
}

#[test]
fn whisper_job_args_test() -> anyhow::Result<()> {
    let jid = JobUid::new();

    let mut envs = WhisperJobArgs {
        job_uid: &jid,
        input_file: "input.mp3",
        language: "en",
    }
    .environments()
    .0;
    envs.sort_by(|a, b| a.0.cmp(&b.0));

    assert_eq!(
        vec![
            ("TRK_INPUT_FILE".to_string(), "input.mp3".to_string()),
            ("TRK_JOB_UID".to_string(), jid.to_string()),
            ("TRK_LANGUAGE".to_string(), "en".to_string()),
        ],
        envs
    );

    Ok(())
}
