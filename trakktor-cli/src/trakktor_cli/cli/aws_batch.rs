use std::{
    io::Write,
    sync::{Arc, OnceLock},
};

use aws_config::Region;
use clap::{Args, Parser, Subcommand};
use trakktor::aws_batch::{
    cloudformation::{verify_base_stack_presence, StackId},
    delete::{do_delete, DeleteArgs},
    download::{download_job_result, DownloadArgs},
    list::list_all_jobs,
    transcribe::{run_transcribe_job, TranscribeJobArgs},
};

use super::Cli;

#[derive(Parser, Debug)]
pub struct AwsBatch {
    /// The AWS profile to use.
    #[arg(long)]
    pub profile: Option<Arc<str>>,
    /// The AWS region to use.
    #[arg(long)]
    pub region: Option<Arc<str>>,
    /// The prefix to use for the CloudFormation stack names.
    #[arg(short, long, default_value = "trakktor")]
    pub stack_prefix: Arc<str>,
    #[clap(subcommand)]
    pub command: AwsBatchCommands,
}

#[derive(Subcommand, Debug)]
pub enum AwsBatchCommands {
    /// Initialize the Trakktor stack.
    Initialize(Initialize),
    /// List all jobs.
    List,
    /// Download the result of a job.
    Download(DownloadArgs),
    /// Delete a job.
    Delete(DeleteArgs),
    /// Run a transcription job.
    Transcribe(TranscribeJobArgs),
}

#[derive(Args, Debug)]
pub struct Initialize {
    /// Silently agree to disclaimer.
    #[arg(long)]
    pub agree: bool,
}

impl Cli {
    pub async fn run_aws_batch(&self, args: &AwsBatch) -> anyhow::Result<()> {
        let mut aws_config = aws_config::from_env();
        if let Some(profile) = &args.profile {
            aws_config = aws_config.profile_name(profile.as_ref());
        }
        if let Some(region) = &args.region {
            aws_config =
                aws_config.region(Region::new(region.as_ref().to_owned()));
        }
        let aws_config = aws_config.load().await;
        let config_provider = Arc::new(GenericConfigProvider {
            aws_config,
            stack_prefix: Arc::clone(&args.stack_prefix),
            s3_bucket: OnceLock::new(),
            dev_mode: self.dev,
        });

        if !matches!(&args.command, AwsBatchCommands::Initialize(_)) {
            if !verify_base_stack_presence(&*config_provider).await? {
                anyhow::bail!(
                    "Base stack not found. Please run `initialize` first."
                );
            }
        }

        match &args.command {
            AwsBatchCommands::Initialize(init) => {
                initialize(config_provider.clone(), init).await?
            },
            AwsBatchCommands::Transcribe(transcribe) => {
                run_transcribe_job(&*config_provider, transcribe).await?
            },
            AwsBatchCommands::Download(download) => {
                download_job_result(&*config_provider, download).await?
            },
            AwsBatchCommands::List => {
                list_all_jobs(config_provider.clone()).await?
            },
            AwsBatchCommands::Delete(delete_args) => {
                do_delete(config_provider.clone(), delete_args).await?
            },
        }

        Ok(())
    }
}

#[tracing::instrument(level = "info", skip_all)]
async fn initialize(
    config_provider: Arc<GenericConfigProvider>,
    init: &Initialize,
) -> anyhow::Result<()> {
    tracing::info!("Initializing Trakktor stack.");
    if !init.agree {
        println!("\n\n{}\n", include_str!("disclaimer.txt"));

        print!("> ");
        std::io::stdout().flush()?;
        let mut input = String::new();
        std::io::stdin().read_line(&mut input)?;
        if input.trim().to_lowercase() != "yes" {
            anyhow::bail!("User did not agree to disclaimer.");
        }
    }

    trakktor::aws_batch::cloudformation::manage_cloudformation_stacks(
        &*config_provider,
        [StackId::Base].into(),
    )
    .await?;

    tracing::info!("Trakktor stack initialized.");

    Ok(())
}

struct GenericConfigProvider {
    aws_config: aws_config::SdkConfig,
    stack_prefix: Arc<str>,
    s3_bucket: OnceLock<Box<str>>,
    dev_mode: bool,
}

impl trakktor::aws_batch::config::AwsConfigProvider for GenericConfigProvider {
    fn get_aws_config(&self) -> &aws_config::SdkConfig { &self.aws_config }
}

impl trakktor::aws_batch::config::CloudFormationStackProvider
    for GenericConfigProvider
{
    fn get_stack_prefix(&self) -> &str { &self.stack_prefix }
}

impl trakktor::aws_batch::config::S3Provider for GenericConfigProvider {
    fn get_bucket_name(&self) -> &str {
        self.s3_bucket.get_or_init(|| {
            trakktor::aws_batch::cloudformation::get_s3_storage_name(
                self.stack_prefix.as_ref(),
            )
        })
    }
}

impl trakktor::app_config::AppConfigProvider for GenericConfigProvider {
    fn is_dev_mode(&self) -> bool { self.dev_mode }
}
