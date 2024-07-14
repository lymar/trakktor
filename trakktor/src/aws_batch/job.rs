use std::sync::Arc;

use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Debug, PartialEq, Eq, Clone, Serialize, Deserialize, Hash)]
pub struct JobUid(Arc<str>);

impl std::fmt::Display for JobUid {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl JobUid {
    pub fn new() -> Self {
        loop {
            let uid = URL_SAFE_NO_PAD.encode(uuid::Uuid::new_v4().as_bytes());
            if !uid.chars().next().unwrap().is_ascii_alphanumeric() {
                continue;
            }
            return Self(uid.into());
        }
    }

    pub fn parse_job_uid(s: &str) -> Result<Self, String> {
        let mk_err = || "Invalid job ID".to_string();
        uuid::Uuid::from_slice(
            &URL_SAFE_NO_PAD.decode(s).map_err(|_| mk_err())?,
        )
        .map_err(|_| mk_err())?;

        Ok(JobUid(s.to_string().into()))
    }
}

impl AsRef<str> for JobUid {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

#[test]
fn new_job_id_test() {
    let job_id = JobUid::new();
    println!("{}", job_id);
    assert_eq!(job_id.0.len(), 22);
}

#[test]
fn parse_job_id_test() {
    let job_id = JobUid::parse_job_uid("BAtQ5-omTm6ZSRTg2AfFKQ").unwrap();
    assert_eq!(job_id.to_string(), "BAtQ5-omTm6ZSRTg2AfFKQ");
    assert!(JobUid::parse_job_uid("BAtQ5-omTm6ZSRTg2AfFK").is_err())
}

#[derive(
    Debug,
    Serialize,
    Deserialize,
    PartialEq,
    Eq,
    Clone,
    Copy,
    strum_macros::Display,
)]
pub enum JobType {
    Transcribe,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq, Clone, Copy)]
pub struct JobInfo {
    #[serde(rename = "t")]
    pub job_type: JobType,
    #[serde(rename = "s")]
    pub start_time: DateTime<Utc>,
}

const JOB_INFO_SUFFIX: &str = ".🚜-info";
pub const JOB_DONE_FLAG: &str = "done.🚜-flag";

impl JobInfo {
    pub fn serialize(&self) -> Box<str> {
        (URL_SAFE_NO_PAD.encode(
            rmp_serde::to_vec(self).expect("Failed to serialize job info"),
        ) + JOB_INFO_SUFFIX)
            .into()
    }

    pub fn check_suffix(serialized: &str) -> bool {
        serialized.ends_with(JOB_INFO_SUFFIX)
    }

    pub fn deserialize(serialized: &str) -> anyhow::Result<Self> {
        if !JobInfo::check_suffix(serialized) {
            anyhow::bail!("Invalid job info suffix");
        }
        let bytes = URL_SAFE_NO_PAD.decode(
            serialized[..serialized.len() - JOB_INFO_SUFFIX.len()].as_bytes(),
        )?;
        Ok(rmp_serde::from_slice(&bytes)?)
    }
}

#[test]
fn job_info_serialize_test() -> anyhow::Result<()> {
    let job_info = JobInfo {
        job_type: JobType::Transcribe,
        start_time: Utc::now(),
    };
    let serialized = job_info.serialize();
    println!("{}", serialized);
    assert!(JobInfo::check_suffix(&serialized));
    let deserialized = JobInfo::deserialize(&serialized)?;
    assert_eq!(job_info, deserialized);
    Ok(())
}

pub const JOB_IN_PREFIX: &str = "in/";

/// Make a storage key for the job input file.
pub fn make_input_storage_key(job_id: &JobUid, file: &str) -> Box<str> {
    format!("{}/{}{}", job_id, JOB_IN_PREFIX, file).into()
}

/// Make a storage key for the job info object.
pub fn make_info_storage_key(job_id: &JobUid, job_info: &JobInfo) -> Box<str> {
    format!("{}/{}", job_id, job_info.serialize()).into()
}

pub const JOB_OUT_PREFIX: &str = "out/";

/// Make a storage key prefix for the job output files.
pub fn make_output_storage_prefix(job_id: &JobUid) -> Box<str> {
    format!("{}/{}", job_id, JOB_OUT_PREFIX).into()
}
