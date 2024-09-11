use std::{
    collections::{HashMap, HashSet},
    error::Error,
    str::FromStr,
};

use aws_sdk_batch::types::JobSummary;
use aws_sdk_cloudformation::{
    types::{Capability, Output, StackStatus, Tag},
    Client,
};
use gpu_batch::GpuBatchStackOutputs;

use super::{
    batch,
    config::{AwsConfigProvider, CloudFormationStackProvider},
    ec2::get_availability_zone_count,
};
use crate::app_config::AppConfigProvider;

mod base;
mod gpu_batch;

const TRAKKTOR_UID_TAG: &str = "trakktor:uid";
const TRAKKTOR_VERSION_TAG: &str = "trakktor:version";
const TRAKKTOR_STACK_TAG: &str = "trakktor:stack";

pub use base::get_s3_storage_name;

#[derive(
    Debug,
    Eq,
    PartialEq,
    Hash,
    Clone,
    Copy,
    strum_macros::EnumString,
    strum_macros::Display,
)]
pub enum StackId {
    Base,
    GpuBatch,
}

impl StackId {
    pub fn get_stack_name(
        &self,
        config: &impl CloudFormationStackProvider,
    ) -> Box<str> {
        match self {
            StackId::Base => config.get_base_stack_name(),
            StackId::GpuBatch => config.get_gpu_batch_stack_name(),
        }
    }
}

#[test]
fn stack_id_str() {
    assert_eq!(StackId::Base.to_string(), "Base");
    assert_eq!(StackId::from_str("GpuBatch").unwrap(), StackId::GpuBatch);
}

/// Manages CloudFormation stacks by ensuring they are created if absent, and
/// updated if their templates have changed.
#[tracing::instrument(level = "debug", skip_all)]
pub async fn manage_cloudformation_stacks(
    config: &(impl AwsConfigProvider
          + CloudFormationStackProvider
          + AppConfigProvider),
    stacks: HashSet<StackId>,
) -> anyhow::Result<()> {
    let client = Client::new(config.get_aws_config());

    let azs_count = tokio::sync::OnceCell::new();
    let azs_count = || async {
        azs_count
            .get_or_try_init(|| async {
                get_availability_zone_count(config).await
            })
            .await
    };

    let all_stacks = StackInfo::load_all(&client).await?;

    if stacks.contains(&StackId::Base) {
        let template = base::gen_cloudformation_template(
            *azs_count().await?,
            config.get_stack_prefix(),
        );

        manage_stack(config, &all_stacks, &client, StackId::Base, &template)
            .await?;
    }

    if stacks.contains(&StackId::GpuBatch) {
        let template = gpu_batch::gen_gpu_batch_template(
            *azs_count().await?,
            &config.get_base_stack_name(),
            config.is_dev_mode(),
        );

        manage_stack(
            config,
            &all_stacks,
            &client,
            StackId::GpuBatch,
            &template,
        )
        .await?;
    }

    Ok(())
}

#[tracing::instrument(
    level = "debug",
    skip(config, all_stacks, client, template)
)]
async fn manage_stack(
    config: &impl CloudFormationStackProvider,
    all_stacks: &HashMap<Box<str>, StackInfo>,
    client: &Client,
    stack_id: StackId,
    template: &str,
) -> anyhow::Result<()> {
    let stack_name = stack_id.get_stack_name(config);
    let ver = crate::hasher::get_hash_value(template.as_bytes());

    if let Some(stack_info) = all_stacks.get(&stack_name) {
        tracing::debug!(?stack_info, "Stack has already been created");
        if stack_info.status != StackStatus::CreateComplete &&
            stack_info.status != StackStatus::UpdateComplete
        {
            anyhow::bail!(
                "Stack is in an unexpected status: {}",
                stack_info.status
            );
        }

        if stack_info.uid.as_ref() == ver {
            tracing::debug!("Stack is up to date");
        } else {
            tracing::debug!("Updating stack");
            update_stack(&client, &stack_name, &template, &ver, stack_id)
                .await?;
        }
    } else {
        tracing::debug!(?stack_name, "Creating stack");
        create_stack(&client, &stack_name, &template, &ver, stack_id).await?;
    }

    Ok(())
}

macro_rules! stack_operation {
    ($client:expr, $method:ident, $stack_name:expr, $template:expr,
        $uid:expr, $stack:expr) => {
        $client
            .$method()
            .stack_name($stack_name)
            .template_body($template)
            .capabilities(Capability::CapabilityIam)
            .capabilities(Capability::CapabilityNamedIam)
            .capabilities(Capability::CapabilityAutoExpand)
            .tags(Tag::builder().key(TRAKKTOR_UID_TAG).value($uid).build())
            .tags(
                Tag::builder()
                    .key(TRAKKTOR_STACK_TAG)
                    .value($stack.to_string())
                    .build(),
            )
            .tags(
                Tag::builder()
                    .key(TRAKKTOR_VERSION_TAG)
                    .value(env!("CARGO_PKG_VERSION"))
                    .build(),
            )
            .send()
    };
}

#[tracing::instrument(level = "debug", skip(client, template))]
async fn create_stack(
    client: &Client,
    stack_name: &str,
    template: &str,
    uid: &str,
    stack: StackId,
) -> anyhow::Result<()> {
    let creation_res = stack_operation!(
        client,
        create_stack,
        stack_name,
        template,
        uid,
        stack
    )
    .await?;

    tracing::debug!(stack_id = ?creation_res.stack_id,
        "Stack creation initiated");

    await_stack_operation_completion(&client, &stack_name).await?;

    Ok(())
}

#[tracing::instrument(level = "debug", skip(client, template))]
async fn update_stack(
    client: &Client,
    stack_name: &str,
    template: &str,
    uid: &str,
    stack: StackId,
) -> anyhow::Result<()> {
    let update_res = stack_operation!(
        client,
        update_stack,
        stack_name,
        template,
        uid,
        stack
    )
    .await?;

    tracing::debug!(stack_id = ?update_res.stack_id, "Stack update initiated");
    await_stack_operation_completion(&client, &stack_name).await?;

    Ok(())
}

#[tracing::instrument(level = "debug", skip(client))]
async fn await_stack_operation_completion(
    client: &Client,
    stack_name: &str,
) -> anyhow::Result<()> {
    loop {
        let stack = client
            .describe_stacks()
            .stack_name(stack_name)
            .send()
            .await?
            .stacks
            .into_iter()
            .flatten()
            .next()
            .ok_or_else(|| anyhow::anyhow!("Stack {} not found", stack_name))?;

        let status = stack.stack_status.ok_or_else(|| {
            anyhow::anyhow!("Stack {} status not found", stack_name)
        })?;
        let status = status.as_str();
        tracing::debug!(?status, "Stack status");

        if status == "CREATE_COMPLETE" || status == "UPDATE_COMPLETE" {
            break;
        } else if status.ends_with("_FAILED") ||
            status.ends_with("ROLLBACK_COMPLETE")
        {
            anyhow::bail!("Stack operation failed: {}", status);
        } else {
            tokio::time::sleep(std::time::Duration::from_secs(15)).await;
        }
    }

    Ok(())
}

#[derive(Debug)]
struct StackInfo {
    stack_id: StackId,
    status: StackStatus,
    uid: Box<str>,
    // This field is intended for recording the version of Trakktor that
    // created this stack, which can be viewed in the AWS console.
    #[allow(dead_code)]
    version: Box<str>,
    outputs: serde_json::Value,
}

impl StackInfo {
    #[tracing::instrument(level = "debug", skip_all)]
    async fn load_all(
        client: &Client,
    ) -> anyhow::Result<HashMap<Box<str>, Self>> {
        Ok(client
            .describe_stacks()
            .into_paginator()
            .send()
            .collect::<Result<Vec<_>, _>>()
            .await?
            .into_iter()
            .map(|s| s.stacks.unwrap_or_default().into_iter())
            .flatten()
            .filter_map(|s| {
                let mut tags = s
                    .tags?
                    .into_iter()
                    .filter_map(|t| {
                        t.key.into_iter().zip(t.value.into_iter()).next()
                    })
                    .collect::<HashMap<_, _>>();
                let outputs = outputs_to_json_obj(s.outputs);
                Some((
                    s.stack_name?.into_boxed_str(),
                    Self {
                        stack_id: StackId::from_str(
                            &tags.remove(TRAKKTOR_STACK_TAG)?,
                        )
                        .ok()?,
                        status: s.stack_status?,
                        uid: tags.remove(TRAKKTOR_UID_TAG)?.into(),
                        version: tags.remove(TRAKKTOR_VERSION_TAG)?.into(),
                        outputs,
                    },
                ))
            })
            .collect::<HashMap<_, _>>())
    }
}

/// Check if the base stack is present
#[tracing::instrument(level = "debug", skip_all)]
pub async fn verify_base_stack_presence(
    config: &(impl AwsConfigProvider + CloudFormationStackProvider),
) -> anyhow::Result<bool> {
    let stack_name = config.get_base_stack_name();
    let client = Client::new(config.get_aws_config());

    let res = client.describe_stacks().stack_name(stack_name).send().await;

    match res {
        Ok(_) => Ok(true),
        Err(err) => {
            if err
                .source()
                .map(|e| e.source())
                .flatten()
                .map(|e| e.to_string().contains("does not exist")) ==
                Some(true)
            {
                Ok(false)
            } else {
                Err(err.into())
            }
        },
    }
}

/// Convert the stack outputs to a JSON object for easier deserialization
fn outputs_to_json_obj(outputs: Option<Vec<Output>>) -> serde_json::Value {
    serde_json::Value::Object(
        outputs
            .into_iter()
            .map(Vec::into_iter)
            .flatten()
            .filter_map(|o| match (o.output_key, o.output_value) {
                (Some(k), Some(v)) => Some((k, serde_json::Value::String(v))),
                _ => None,
            })
            .collect::<serde_json::Map<_, _>>(),
    )
}

/// Load the stack outputs
#[tracing::instrument(level = "debug", skip(config))]
async fn load_stack_outputs(
    config: &(impl AwsConfigProvider + CloudFormationStackProvider),
    stack: StackId,
) -> anyhow::Result<serde_json::Value> {
    let client = Client::new(config.get_aws_config());
    let stack_name = stack.get_stack_name(config);
    let outputs = client
        .describe_stacks()
        .stack_name(stack_name.as_ref())
        .send()
        .await?
        .stacks
        .into_iter()
        .flatten()
        .next()
        .ok_or_else(|| anyhow::anyhow!("Stack {} not found", stack_name))?
        .outputs;

    Ok(outputs_to_json_obj(outputs))
}

/// Load GPU stack outputs
#[tracing::instrument(level = "debug", skip_all)]
pub async fn load_gpu_stack_outputs(
    config: &(impl AwsConfigProvider + CloudFormationStackProvider),
) -> anyhow::Result<GpuBatchStackOutputs> {
    Ok(serde_json::from_value(
        load_stack_outputs(config, StackId::GpuBatch).await?,
    )?)
}

/// Load all batch job status
#[tracing::instrument(level = "debug", skip_all)]
pub async fn load_all_batch_jobs(
    config: &(impl AwsConfigProvider + CloudFormationStackProvider),
) -> anyhow::Result<HashMap<StackId, Vec<JobSummary>>> {
    let client = Client::new(config.get_aws_config());
    let stacks = StackInfo::load_all(&client).await?;
    let mut batch_queue_names: Vec<Box<str>> = vec![];
    let mut batch_queue_ids: Vec<StackId> = vec![];
    for (
        _,
        StackInfo {
            stack_id, outputs, ..
        },
    ) in stacks
    {
        match stack_id {
            StackId::Base => {},
            StackId::GpuBatch => {
                batch_queue_names.push(
                    serde_json::from_value::<GpuBatchStackOutputs>(outputs)?
                        .job_queue
                        .into(),
                );
                batch_queue_ids.push(StackId::GpuBatch);
            },
        }
    }

    let jobs = batch::load_jobs(config, batch_queue_names).await?;

    Ok(batch_queue_ids.into_iter().zip(jobs.into_iter()).collect())
}
