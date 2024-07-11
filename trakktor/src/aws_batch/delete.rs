use std::sync::Arc;

use tokio::{sync::Semaphore, task::JoinHandle};
use tracing::{info_span, Instrument};

use crate::aws_batch::{
    config::{AwsConfigProvider, S3Provider},
    job::JobUid,
    s3::delete_dir,
};

#[derive(clap::Args, Debug)]
pub struct DeleteArgs {
    /// List of job IDs to delete.
    pub job_ids: Vec<String>,
}

const PARALLEL_REQS: usize = 8;

#[tracing::instrument(level = "debug", skip_all)]
pub async fn do_delete(
    config: Arc<impl AwsConfigProvider + S3Provider + Sync + Send + 'static>,
    args: &DeleteArgs,
) -> anyhow::Result<()> {
    let jids: Vec<JobUid> = args
        .job_ids
        .iter()
        .map(|j| JobUid::parse_job_uid(j).map_err(|e| anyhow::anyhow!(e)))
        .collect::<Result<Vec<JobUid>, _>>()?;

    let par_sem = Arc::new(Semaphore::new(PARALLEL_REQS));

    let mut reqs: Vec<JoinHandle<anyhow::Result<()>>> = Vec::new();

    for d in jids {
        reqs.push(tokio::spawn({
            let config = Arc::clone(&config);
            let span = info_span!("delete job", id = d.as_ref());
            let par_sem = Arc::clone(&par_sem);
            async move {
                let _permit = par_sem.acquire().await?;
                tracing::info!("deleting...");
                delete_dir(&*config, d.as_ref()).await?;
                Ok(())
            }
            .instrument(span)
        }));
    }

    for req in reqs {
        req.await??;
    }

    Ok(())
}
