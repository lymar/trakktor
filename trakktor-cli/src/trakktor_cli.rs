use std::{
    io::Write,
    sync::{Arc, OnceLock},
};

mod cli;

pub use cli::Cli;
use cli::Initialize;
use trakktor::{
    aws::cloudformation::{verify_base_stack_presence, StackId},
    delete::do_delete,
    download::download_job_result,
    list::list_all_jobs,
    transcribe::run_transcribe_job,
};

pub struct TrakktorCli {
    cli: Cli,
    config_provider: Arc<GenericConfigProvider>,
}

impl TrakktorCli {
    pub async fn run(cli: Cli) -> anyhow::Result<()> {
        let aws_config = aws_config::from_env()
            .profile_name(cli.aws_profile.as_ref())
            .load()
            .await;
        let config_provider = Arc::new(GenericConfigProvider {
            aws_config,
            stack_prefix: Arc::clone(&cli.stack_prefix),
            s3_bucket: OnceLock::new(),
            dev_mode: cli.dev,
        });
        let trk = TrakktorCli {
            cli,
            config_provider,
        };
        trk.exec().await
    }

    async fn exec(&self) -> anyhow::Result<()> {
        if !matches!(&self.cli.command, cli::Commands::Initialize(_)) {
            if !verify_base_stack_presence(&*self.config_provider).await? {
                anyhow::bail!(
                    "Base stack not found. Please run `trakktor-cli \
                     initialize` first."
                );
            }
        }

        match &self.cli.command {
            cli::Commands::Initialize(init) => self.initialize(init).await?,
            cli::Commands::Transcribe(transcribe) => {
                run_transcribe_job(&*self.config_provider, transcribe).await?
            },
            cli::Commands::Download(download) => {
                download_job_result(&*self.config_provider, download).await?
            },
            cli::Commands::List => {
                list_all_jobs(self.config_provider.clone()).await?
            },
            cli::Commands::Delete(delete_args) => {
                do_delete(self.config_provider.clone(), delete_args).await?
            },
        }

        Ok(())
    }

    #[tracing::instrument(level = "info", skip_all)]
    async fn initialize(&self, init: &Initialize) -> anyhow::Result<()> {
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

        trakktor::aws::cloudformation::manage_cloudformation_stacks(
            &*self.config_provider,
            [StackId::Base].into(),
        )
        .await?;

        tracing::info!("Trakktor stack initialized.");

        Ok(())
    }
}

struct GenericConfigProvider {
    aws_config: aws_config::SdkConfig,
    stack_prefix: Arc<str>,
    s3_bucket: OnceLock<Box<str>>,
    dev_mode: bool,
}

impl trakktor::aws::config::AwsConfigProvider for GenericConfigProvider {
    fn get_aws_config(&self) -> &aws_config::SdkConfig {
        &self.aws_config
    }
}

impl trakktor::aws::config::CloudFormationStackProvider
    for GenericConfigProvider
{
    fn get_stack_prefix(&self) -> &str {
        &self.stack_prefix
    }
}

impl trakktor::aws::config::S3Provider for GenericConfigProvider {
    fn get_bucket_name(&self) -> &str {
        self.s3_bucket.get_or_init(|| {
            trakktor::aws::cloudformation::get_s3_storage_name(
                self.stack_prefix.as_ref(),
            )
        })
    }
}

impl trakktor::app_config::AppConfigProvider for GenericConfigProvider {
    fn is_dev_mode(&self) -> bool {
        self.dev_mode
    }
}
