use anyhow::{anyhow, Context};

use crate::{
    app_config::AppConfigProvider,
    aws::{
        batch::submit_job,
        cloudformation::{load_gpu_stack_outputs, StackId},
        config::{AwsConfigProvider, CloudFormationStackProvider, S3Provider},
        s3::{put_object, upload_file},
    },
    job::{
        make_info_storage_key, make_input_storage_key, JobInfo, JobType, JobUid,
    },
    whisper::WhisperJobArgs,
};

#[derive(clap::Args, Debug)]
pub struct TranscribeJobArgs {
    /// File to transcribe.
    pub file: std::path::PathBuf,
    /// The language of the audio.
    pub language: Box<str>,
}

impl TranscribeJobArgs {
    fn get_file_name(&self) -> anyhow::Result<&str> {
        Ok(self
            .file
            .file_name()
            .ok_or_else(|| anyhow!("Unable to get file name"))?
            .to_str()
            .ok_or_else(|| anyhow!("Invalid file name"))?)
    }
}

#[tracing::instrument(level = "info", skip(config))]
pub async fn run_transcribe_job(
    config: &(impl AwsConfigProvider
          + S3Provider
          + CloudFormationStackProvider
          + AppConfigProvider),
    job: &TranscribeJobArgs,
) -> anyhow::Result<()> {
    crate::aws::cloudformation::manage_cloudformation_stacks(
        config,
        [StackId::Base, StackId::GpuBatch].into(),
    )
    .await?;

    let jid = JobUid::new();
    tracing::info!(job_id = %jid, "Starting transcription job.");

    let file_name = job.get_file_name().with_context(|| {
        format!("Could not get file name: {}", job.file.display())
    })?;

    let start_time = chrono::Utc::now();

    upload_file(config, &job.file, &make_input_storage_key(&jid, &file_name))
        .await?;

    let job_info = JobInfo {
        job_type: JobType::Transcribe,
        start_time,
    };

    put_object(config, b"", &make_info_storage_key(&jid, &job_info)).await?;

    let stack_outputs = load_gpu_stack_outputs(config).await?;
    tracing::debug!(?stack_outputs, "Loaded GPU stack outputs.");

    submit_job(
        config,
        jid.clone(),
        &stack_outputs.job_queue,
        &stack_outputs.whisper_large_job,
        WhisperJobArgs {
            job_uid: &jid,
            input_file: &file_name,
            language: &job.language,
        }
        .environments(),
    )
    .await?;

    tracing::info!(job_id = %jid, "Transcription job submitted.");

    Ok(())
}
