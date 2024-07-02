use std::sync::Arc;

use aws_sdk_batch::{
    types::{ContainerOverrides, JobSummary, KeyValuePair, KeyValuesPair},
    Client,
};
use tokio::{sync::Semaphore, task::JoinHandle};
use tracing::{info_span, Instrument};

use super::config::{AwsConfigProvider, CloudFormationStackProvider};
use crate::job::JobUid;

const PARALLEL_REQS: usize = 8;

/// Environment variables to be passed to the job container.
#[derive(Debug)]
pub struct ContainerEnvs(pub Vec<(String, String)>);

#[tracing::instrument(level = "debug", skip(config))]
pub async fn submit_job(
    config: &(impl AwsConfigProvider + CloudFormationStackProvider),
    uid: JobUid,
    queue: &str,
    definition: &str,
    envs: ContainerEnvs,
) -> anyhow::Result<()> {
    let client = Client::new(config.get_aws_config());

    client
        .submit_job()
        .job_name(uid.to_string())
        .job_queue(queue)
        .job_definition(definition)
        .container_overrides(
            ContainerOverrides::builder()
                .set_environment(Some(
                    envs.0
                        .into_iter()
                        .map(|(k, v)| {
                            KeyValuePair::builder().name(k).value(v).build()
                        })
                        .collect(),
                ))
                .build(),
        )
        .send()
        .await?;

    Ok(())
}

#[tracing::instrument(level = "debug", skip_all)]
pub async fn load_jobs(
    config: &(impl AwsConfigProvider + CloudFormationStackProvider),
    batch_queues: impl IntoIterator<Item = impl AsRef<str> + Send + 'static>,
) -> anyhow::Result<Vec<Vec<JobSummary>>> {
    let client = Client::new(config.get_aws_config());

    let par_sem = Arc::new(Semaphore::new(PARALLEL_REQS));

    let mut chunks: Vec<JoinHandle<anyhow::Result<Vec<JobSummary>>>> =
        Vec::new();

    for queue in batch_queues {
        let par_sem = Arc::clone(&par_sem);
        let client = client.clone();
        let span = info_span!("list jobs", queue = queue.as_ref());
        chunks.push(tokio::spawn(
            async move {
                let _permit = par_sem.acquire().await?;
                let jobs = client
                    .list_jobs()
                    .job_queue(queue.as_ref())
                    // AWS returns empty list if `AFTER_CREATED_AT` is not set.
                    .filters(
                        KeyValuesPair::builder()
                            .name("AFTER_CREATED_AT")
                            .values("0")
                            .build(),
                    )
                    .into_paginator()
                    .send()
                    .collect::<Result<Vec<_>, _>>()
                    .await?
                    .into_iter()
                    .filter_map(|res| res.job_summary_list)
                    .flatten();

                Ok(jobs.collect())
            }
            .instrument(span),
        ));
    }

    let mut res = vec![];

    for c in chunks {
        res.push(c.await??);
    }

    Ok(res)
}
