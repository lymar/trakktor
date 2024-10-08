use super::config::AwsConfigProvider;

#[tracing::instrument(level = "debug", skip_all)]
pub async fn get_availability_zone_count(
    aws_cfg_provider: &impl AwsConfigProvider,
) -> anyhow::Result<usize> {
    let client = aws_sdk_ec2::Client::new(aws_cfg_provider.get_aws_config());
    let resp = client
        .describe_availability_zones()
        .filters(
            aws_sdk_ec2::types::Filter::builder()
                .name("zone-type")
                .values("availability-zone")
                .build(),
        )
        .send()
        .await?;

    let count = resp.availability_zones().len();
    tracing::debug!(count);

    Ok(count)
}
