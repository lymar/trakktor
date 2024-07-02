use std::{path::Path, sync::Arc, time::Duration};

use anyhow::{anyhow, bail};
use aws_config::timeout::TimeoutConfig;
use aws_sdk_s3::{
    operation::create_multipart_upload::CreateMultipartUploadOutput,
    primitives::ByteStream,
    types::{
        CompletedMultipartUpload, CompletedPart, Delete, ObjectIdentifier,
    },
    Client,
};
use aws_smithy_types::{body::SdkBody, byte_stream::Length};
use tokio::{fs::File, io::AsyncWriteExt, sync::Semaphore, task::JoinHandle};
use tracing::{info_span, Instrument};

use super::config::{AwsConfigProvider, S3Provider};

const CHUNK_SIZE: u64 = 1024 * 1024 * 5;
const PARALLEL_UPLOADS: usize = 4;
const PARALLEL_DOWNLOADS: usize = 4;

fn get_client(
    config: &impl AwsConfigProvider,
    with_long_timeout: bool,
) -> Client {
    let mut s3_config =
        aws_sdk_s3::config::Builder::from(config.get_aws_config())
            .accelerate(true);
    if with_long_timeout {
        s3_config = s3_config.timeout_config(
            TimeoutConfig::builder()
                .operation_attempt_timeout(Duration::from_secs(60 * 5))
                .build(),
        );
    }
    Client::from_conf(s3_config.build())
}

#[tracing::instrument(level = "debug", skip(config))]
pub async fn upload_file(
    config: &(impl AwsConfigProvider + S3Provider),
    file_path: &Path,
    s3_key: &str,
) -> anyhow::Result<()> {
    let file_path = Arc::new(file_path.to_owned());
    let s3_key = Arc::new(s3_key.to_string());

    let client = get_client(config, true);

    tracing::debug!("Uploading file to S3.");

    let bucket_name = Arc::new(config.get_bucket_name().to_string());

    let multipart_upload_res: CreateMultipartUploadOutput = client
        .create_multipart_upload()
        .bucket(bucket_name.as_str())
        .key(s3_key.as_str())
        .send()
        .await?;
    let upload_id = Arc::new(
        multipart_upload_res
            .upload_id()
            .ok_or_else(|| anyhow!("empty upload id"))?
            .to_string(),
    );

    let file_size = tokio::fs::metadata(file_path.as_ref()).await?.len();

    if file_size == 0 {
        bail!("Bad file size.");
    }

    let mut chunk_count = (file_size / CHUNK_SIZE) + 1;
    let mut size_of_last_chunk = file_size % CHUNK_SIZE;
    if size_of_last_chunk == 0 {
        size_of_last_chunk = CHUNK_SIZE;
        chunk_count -= 1;
    }

    let mut parts: Vec<JoinHandle<anyhow::Result<CompletedPart>>> = Vec::new();

    let par_sem = Arc::new(Semaphore::new(PARALLEL_UPLOADS));

    for chunk_index in 0..chunk_count {
        let bucket_name = Arc::clone(&bucket_name);
        let file_path = Arc::clone(&file_path);
        let s3_key = Arc::clone(&s3_key);
        let upload_id = Arc::clone(&upload_id);
        let par_sem = Arc::clone(&par_sem);
        let client = client.clone();
        let span = info_span!("chunk upload", chunk_index);
        parts.push(tokio::spawn(
            async move {
                let _permit = par_sem.acquire().await?;

                tracing::debug!("uploading");
                let this_chunk = if chunk_count - 1 == chunk_index {
                    size_of_last_chunk
                } else {
                    CHUNK_SIZE
                };
                let stream = ByteStream::read_from()
                    .path(file_path.as_ref())
                    .offset(chunk_index * CHUNK_SIZE)
                    .length(Length::Exact(this_chunk))
                    .build()
                    .await?;
                // Chunk index needs to start at 0, but part numbers start at 1.
                let part_number = (chunk_index as i32) + 1;
                let upload_part_res = client
                    .upload_part()
                    .key(s3_key.as_str())
                    .bucket(bucket_name.as_str())
                    .upload_id(upload_id.as_str())
                    .body(stream)
                    .part_number(part_number)
                    .send()
                    .await?;
                Ok(CompletedPart::builder()
                    .e_tag(upload_part_res.e_tag.unwrap_or_default())
                    .part_number(part_number)
                    .build())
            }
            .instrument(span),
        ));
    }

    let mut upload_parts: Vec<CompletedPart> = Vec::new();
    for part in parts {
        upload_parts.push(part.await??);
    }

    let completed_multipart_upload: CompletedMultipartUpload =
        CompletedMultipartUpload::builder()
            .set_parts(Some(upload_parts))
            .build();

    client
        .complete_multipart_upload()
        .bucket(config.get_bucket_name())
        .key(s3_key.as_str())
        .multipart_upload(completed_multipart_upload)
        .upload_id(upload_id.as_str())
        .send()
        .await?;

    tracing::debug!("Upload complete.");

    Ok(())
}

#[tracing::instrument(level = "debug", skip(config, data))]
pub async fn put_object(
    config: &(impl AwsConfigProvider + S3Provider),
    data: &[u8],
    s3_key: &str,
) -> anyhow::Result<()> {
    let client = get_client(config, false);

    client
        .put_object()
        .bucket(config.get_bucket_name())
        .key(s3_key)
        .body(ByteStream::new(SdkBody::from(data)))
        .send()
        .await?;

    tracing::debug!("Put object complete.");

    Ok(())
}

#[tracing::instrument(level = "debug", skip(config))]
pub async fn list_objects(
    config: &(impl AwsConfigProvider + S3Provider),
    s3_dir: &str,
) -> anyhow::Result<impl Iterator<Item = String>> {
    Ok(get_client(config, false)
        .list_objects_v2()
        .bucket(config.get_bucket_name())
        .prefix(s3_dir)
        .into_paginator()
        .send()
        .collect::<Result<Vec<_>, _>>()
        .await?
        .into_iter()
        .filter_map(|res| res.contents)
        .map(|i| i.into_iter())
        .flatten()
        .filter_map(|o| o.key)
        .filter(|k| !k.ends_with('/')))
}

#[tracing::instrument(level = "debug", skip(config, objs))]
pub async fn download_folder(
    config: &(impl AwsConfigProvider + S3Provider),
    objs: impl IntoIterator<Item = String>,
    s3_prefix: &str,
    dest_dir: &Path,
) -> anyhow::Result<()> {
    let client = get_client(config, false);
    let bucket_name = Arc::new(config.get_bucket_name().to_string());
    let dest_dir = Arc::new(dest_dir.to_path_buf());
    let s3_prefix = Arc::new(s3_prefix.to_string());
    let par_sem = Arc::new(Semaphore::new(PARALLEL_DOWNLOADS));
    let mut tasks: Vec<JoinHandle<anyhow::Result<()>>> = Vec::new();

    for obj in objs {
        let bucket_name = Arc::clone(&bucket_name);
        let s3_prefix = Arc::clone(&s3_prefix);
        let dest_dir = Arc::clone(&dest_dir);
        let client = client.clone();
        let par_sem = Arc::clone(&par_sem);
        let span = info_span!("download object", obj);

        tasks.push(tokio::spawn(
            async move {
                let _permit = par_sem.acquire().await?;

                let dest_path = dest_dir.join(
                    obj.strip_prefix(s3_prefix.as_ref())
                        .expect("unexpected object prefix"),
                );
                tracing::debug!(?dest_path, "downloading");

                if let Some(parent) = dest_path.parent() {
                    tokio::fs::create_dir_all(parent).await?;
                }
                let mut file = File::create(&dest_path).await?;

                let mut object = client
                    .get_object()
                    .bucket(bucket_name.as_str())
                    .key(obj)
                    .send()
                    .await?;

                while let Some(bytes) = object.body.try_next().await? {
                    file.write_all(&bytes).await?;
                }

                Ok(())
            }
            .instrument(span),
        ));
    }

    for task in tasks {
        task.await??;
    }

    Ok(())
}

#[tracing::instrument(level = "debug", skip_all)]
pub async fn delete_dir(
    config: &(impl AwsConfigProvider + S3Provider),
    s3_dir: &str,
) -> anyhow::Result<()> {
    let client = get_client(config, false);

    let objects = client
        .list_objects_v2()
        .bucket(config.get_bucket_name())
        .prefix(s3_dir)
        .into_paginator()
        .send()
        .collect::<Result<Vec<_>, _>>()
        .await?
        .into_iter()
        .filter_map(|res| res.contents)
        .map(|i| i.into_iter())
        .flatten();

    let mut delete_objects: Vec<ObjectIdentifier> = vec![];
    for obj in objects {
        let obj_id = ObjectIdentifier::builder()
            .set_key(Some(obj.key().unwrap().to_string()))
            .build()?;
        delete_objects.push(obj_id);
    }

    if !delete_objects.is_empty() {
        get_client(config, false)
            .delete_objects()
            .bucket(config.get_bucket_name())
            .delete(
                Delete::builder()
                    .set_objects(Some(delete_objects))
                    .build()?,
            )
            .send()
            .await?;
    } else {
        tracing::info!("No objects to delete.");
    }

    Ok(())
}
