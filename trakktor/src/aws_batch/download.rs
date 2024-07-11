use crate::aws_batch::{
    config::{AwsConfigProvider, S3Provider},
    job::{make_output_storage_prefix, JobUid, JOB_DONE_FLAG},
    s3::{download_folder, list_objects},
};

#[derive(clap::Args, Debug)]
pub struct DownloadArgs {
    /// Job ID to download.
    #[arg(value_parser = JobUid::parse_job_uid)]
    pub job_id: JobUid,
    /// Directory to download to. If not specified, the current directory is
    /// used.
    pub out_path: Option<std::path::PathBuf>,
}

#[tracing::instrument(level = "info", skip(config))]
pub async fn download_job_result(
    config: &(impl AwsConfigProvider + S3Provider),
    args: &DownloadArgs,
) -> anyhow::Result<()> {
    let objs = list_objects(config, &args.job_id.to_string())
        .await?
        .collect::<Vec<_>>();

    if objs.is_empty() {
        anyhow::bail!("Job not found.");
    }

    tracing::debug!(?objs, "Listed objects.");

    if !objs.iter().any(|i| i.ends_with(JOB_DONE_FLAG)) {
        anyhow::bail!("Job not finished yet.");
    }

    let pfx = make_output_storage_prefix(&args.job_id);

    download_folder(
        config,
        objs.into_iter().filter(|o| o.starts_with(pfx.as_ref())),
        &pfx,
        args.out_path
            .as_deref()
            .unwrap_or(std::path::Path::new(".")),
    )
    .await?;

    Ok(())
}
