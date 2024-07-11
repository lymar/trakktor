use std::{collections::HashMap, sync::Arc};

use anyhow::bail;
use aws_sdk_batch::types::JobSummary;
use chrono::{DateTime, Local};
use duration_str::HumanFormat;
use tracing::{info_span, Instrument};

use crate::aws_batch::{
    cloudformation::load_all_batch_jobs,
    config::{AwsConfigProvider, CloudFormationStackProvider, S3Provider},
    job::{JobInfo, JobUid, JOB_DONE_FLAG, JOB_IN_PREFIX, JOB_OUT_PREFIX},
    s3::list_objects,
};

#[derive(Debug, strum_macros::Display)]
enum JobStatus {
    Unknown,
    Done,
    InProgress,
    Failed,
}

impl Default for JobStatus {
    fn default() -> Self { Self::Unknown }
}

#[derive(Debug, Default)]
struct JobDisplayInfo<'a> {
    in_files: Vec<&'a str>,
    out_files: Vec<&'a str>,
    status: JobStatus,
    duration: Option<std::time::Duration>,
}

#[derive(Debug)]
struct JobDisplayFull<'a> {
    uid: JobUid,
    display_info: JobDisplayInfo<'a>,
    job_info: JobInfo,
}

const IND: &str = "    ";

#[tracing::instrument(level = "debug", skip(config))]
pub async fn list_all_jobs(
    config: Arc<
        impl AwsConfigProvider
            + S3Provider
            + CloudFormationStackProvider
            + Sync
            + Send
            + 'static,
    >,
) -> anyhow::Result<()> {
    println!();

    let list_obj_task = {
        let config = Arc::clone(&config);
        tokio::spawn(
            async move {
                anyhow::Result::<Vec<_>>::Ok(
                    list_objects(&*config, "").await?.collect::<Vec<_>>(),
                )
            }
            .instrument(info_span!("list s3 objects task")),
        )
    };

    let load_jobs_task = {
        let config = Arc::clone(&config);
        tokio::spawn(
            async move { load_all_batch_jobs(&*config).await }
                .instrument(info_span!("loading batch jobs")),
        )
    };

    let s3_objs = list_obj_task.await??;
    let mut job_summaries = load_jobs_task
        .await??
        .into_iter()
        .map(|(_, j)| j.into_iter())
        .flatten()
        .filter_map(|mut j| {
            Some((std::mem::replace(&mut j.job_name, None)?, j))
        })
        .collect::<HashMap<String, JobSummary>>();

    // tracing::debug!(?job_summaries, "loaded jobs list");

    let mut jobs_map = HashMap::<JobUid, JobDisplayInfo>::new();
    let mut jobs_info = HashMap::<JobUid, JobInfo>::new();

    for o in &s3_objs {
        let Some((job_uid, rest)) = o.split_once('/') else {
            bail!("Invalid job object: {o}");
        };
        let job_uid = JobUid::parse_job_uid(job_uid)
            .map_err(|m| anyhow::anyhow!("{job_uid}: {}", m))?;
        let info = jobs_map.entry(job_uid.clone()).or_default();

        if let Some(in_file) = rest.strip_prefix(JOB_IN_PREFIX) {
            info.in_files.push(in_file);
        } else if let Some(out_file) = rest.strip_prefix(JOB_OUT_PREFIX) {
            info.out_files.push(out_file);
        } else if rest == JOB_DONE_FLAG {
            info.status = JobStatus::Done;
        } else if let Ok(ji) = JobInfo::deserialize(rest) {
            jobs_info.insert(job_uid.clone(), ji);
        } else {
            bail!("Unexpected job object: {o}");
        }
    }

    let mut jobs = jobs_map
        .into_iter()
        .filter_map(|(ji, mut info)| {
            let Some(job_info) = jobs_info.remove(&ji) else {
                tracing::error!("Job info not found for {ji}");
                return None;
            };

            if let Some(summ) = job_summaries.remove(ji.as_ref()) {
                if matches!(info.status, JobStatus::Unknown) {
                    use aws_sdk_batch::types::JobStatus as JS;
                    match summ.status {
                        Some(s)
                            if s == JS::Pending ||
                                s == JS::Runnable ||
                                s == JS::Running ||
                                s == JS::Starting ||
                                s == JS::Submitted =>
                        {
                            info.status = JobStatus::InProgress;
                        },
                        Some(s) if s == JS::Failed => {
                            info.status = JobStatus::Failed
                        },
                        Some(JS::Succeeded) => {
                            tracing::error!(
                                "Job {ji} is marked as done but no done flag \
                                 found"
                            );
                        },
                        Some(s) => {
                            tracing::warn!("Job {ji} has unknown status: {s}");
                        },
                        None => {},
                    }
                }

                if let (Some(started_at), Some(stopped_at)) =
                    (summ.started_at, summ.stopped_at)
                {
                    info.duration = Some(std::time::Duration::from_secs(
                        (stopped_at - started_at) as u64 / 1000,
                    ));
                }
            }

            Some(JobDisplayFull {
                uid: ji,
                display_info: info,
                job_info,
            })
        })
        .collect::<Vec<_>>();

    jobs.sort_by_key(|e| e.job_info.start_time);

    for JobDisplayFull {
        uid,
        display_info,
        job_info,
    } in &jobs
    {
        let local_time: DateTime<Local> = DateTime::from(job_info.start_time);
        println!("- {} -- {} ({})", uid, job_info.job_type, local_time);
        println!("{IND}status: {}", display_info.status);
        if let Some(d) = display_info.duration {
            println!("{IND}duration: {}", d.human_format());
        }
        println!("{IND}files:");
        print_list(2, display_info.in_files.iter());
        if !display_info.out_files.is_empty() {
            println!("{IND}output files:");
            print_list(2, display_info.out_files.iter());
        }
    }

    println!();

    Ok(())
}

fn print_list(
    indent_count: usize,
    data: impl IntoIterator<Item = impl AsRef<str>>,
) {
    for d in data.into_iter() {
        for _ in 0..indent_count {
            print!("{}", IND);
        }
        println!("{}", d.as_ref());
    }
}
