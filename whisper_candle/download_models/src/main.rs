use std::{fs::create_dir_all, path::Path};

use hf_hub::{api::sync::Api, Repo, RepoType};
use trakktor_candle::speech_recognition::{DataFile, WhichModel};

fn main() -> anyhow::Result<()> {
    stderrlog::new()
        .module(module_path!())
        .verbosity(log::Level::Trace)
        .init()?;

    let (model, rev) = WhichModel::LargeV3.model_and_revision();
    let data_dir = Path::new("./models_data");
    let model_dir = data_dir.join(model);
    create_dir_all(&model_dir)?;

    let api = Api::new()?;
    let repo = api.repo(Repo::with_revision(
        model.to_string(),
        RepoType::Model,
        rev.to_string(),
    ));

    for file_name in enum_iterator::all::<DataFile>().map(DataFile::file_name) {
        log::info!("Start processing {}", file_name);
        let res_path = model_dir.join(file_name);
        if res_path.exists() {
            log::info!("{} already exists", file_name);
            continue;
        }
        std::fs::copy(repo.get(file_name)?.as_path(), res_path)?;
        log::info!("{} downloaded and copied", file_name);
    }

    Ok(())
}
