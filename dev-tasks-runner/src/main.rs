use clap::{Parser, Subcommand};
use cmd_lib::*;
use trakktor::whisper;

#[derive(Debug)]
struct TasksRunner {
    cli: Cli,
}

#[derive(Parser, Debug)]
#[command(about, long_about = None)]
struct Cli {
    #[arg(long, default_value = "lymar")]
    ghcr_login: Box<str>,
    #[arg(long, env = "GHCR_TOKEN")]
    ghcr_token: Box<str>,
    #[arg(long)]
    release: bool,
    #[clap(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Build and push the Docker images.
    DockerBuild,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let runner = TasksRunner { cli };

    runner.run()?;

    Ok(())
}

impl TasksRunner {
    fn run(&self) -> anyhow::Result<()> {
        match self.cli.command {
            Commands::DockerBuild => self.docker_build()?,
        }

        Ok(())
    }

    fn docker_build(&self) -> anyhow::Result<()> {
        self.ghcr_login()?;

        let model = whisper::Model::Large;
        let model_name = model.get_name();
        let full_image_name =
            whisper::make_image_name(model, !self.cli.release);

        println!("Building Docker image: {}", full_image_name);
        run_cmd! {
            docker build --platform linux/amd64 --build-arg WHISPER_MODEL=${model_name} -t ${full_image_name} -f ./whisper/Dockerfile ./whisper
        }?;

        if self.cli.release {
            let inspect_res = run_fun! {
                docker manifest inspect ${full_image_name}
            };
            if inspect_res.is_ok() {
                anyhow::bail!(
                    "This version of the Docker image already exists: {}",
                    full_image_name
                );
            }
        }

        println!("Pushing Docker image: {}", full_image_name);
        run_cmd! {
            docker push ${full_image_name}
        }?;

        Ok(())
    }

    fn ghcr_login(&self) -> anyhow::Result<()> {
        println!("Logging in to GitHub Container Registry...");
        let (login, token) = (&self.cli.ghcr_login, &self.cli.ghcr_token);

        run_cmd! {
            echo "${token}" | docker login ghcr.io -u ${login} --password-stdin
        }?;

        Ok(())
    }
}
