use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

use serde::{Deserialize, Serialize};
use tokio::sync::Semaphore;

use crate::llm::{ChatCompletionPlatform, Role};

pub struct ChatDoc {
    pub toml_doc: toml_edit::DocumentMut,
    /// The original chat data from file before handling includes.
    pub original_chat_data: ChatData,
    pub msgs: Vec<Msg>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct ChatData {
    #[serde(default)]
    pub cfg: Vec<Cfg>,
    pub msgs: Vec<Msg>,
}

#[derive(Debug, Deserialize, Clone, Default)]
pub struct Cfg {
    pub name: Option<String>,
    pub platform: Option<ChatCompletionPlatform>,
    pub model: Option<String>,
    pub response_format: Option<String>,
    #[serde(default)]
    pub beautify_json_response: bool,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(untagged)]
pub enum Msg {
    Text { role: Role, content: String },
    Include { include: String },
}

const MAX_CONCURRENT_READS: usize = 64;

impl ChatDoc {
    #[tracing::instrument(level = "debug")]
    pub async fn load(file: &Path) -> anyhow::Result<Self> {
        let contents = tokio::fs::read_to_string(file).await?;

        let toml_doc = contents.parse::<toml_edit::DocumentMut>()?;
        let chat_data: ChatData =
            toml_edit::de::from_document(toml_doc.clone())?;
        let original_chat_data = chat_data.clone();
        let msgs = handle_msgs(
            file.to_path_buf(),
            original_chat_data.msgs.clone(),
            Arc::new(Semaphore::new(MAX_CONCURRENT_READS)),
        )
        .await?;

        Ok(Self {
            toml_doc,
            original_chat_data,
            msgs,
        })
    }

    #[tracing::instrument(level = "debug", skip(self))]
    pub async fn write_doc(&self, file: &Path) -> anyhow::Result<()> {
        let toml_str = self.toml_doc.to_string();
        tokio::fs::write(file, toml_str).await?;
        Ok(())
    }
}

async fn handle_msgs(
    file: PathBuf,
    msgs: Vec<Msg>,
    permits: Arc<Semaphore>,
) -> anyhow::Result<Vec<Msg>> {
    enum Elem<T> {
        Msg(Msg),
        Include(T),
    }

    let mut tasks = Vec::with_capacity(msgs.len());

    for msg in msgs {
        match msg {
            Msg::Include { include } => {
                let permits = Arc::clone(&permits);
                let include_path = file
                    .parent()
                    .expect("file has no parent path")
                    .join(include);
                tasks.push(Elem::Include(tokio::spawn(proc_include(
                    include_path,
                    permits,
                ))));
            },
            msg => {
                tasks.push(Elem::Msg(msg));
            },
        }
    }

    let mut msgs = vec![];
    for task in tasks {
        match task {
            Elem::Msg(msg) => msgs.push(msg),
            Elem::Include(join_handle) => {
                msgs.append(&mut join_handle.await??);
            },
        }
    }

    Ok(msgs)
}

async fn read_file(
    file: &Path,
    permits: &Arc<Semaphore>,
) -> anyhow::Result<String> {
    let _permit = permits.acquire().await?;
    Ok(tokio::fs::read_to_string(file).await?)
}

#[async_recursion::async_recursion]
async fn proc_include(
    include_path: PathBuf,
    permits: Arc<Semaphore>,
) -> anyhow::Result<Vec<Msg>> {
    let contents = read_file(&include_path, &permits).await?;
    let toml_doc = contents.parse::<toml_edit::DocumentMut>()?;
    let chat_data: ChatData = toml_edit::de::from_document(toml_doc)?;
    Ok(handle_msgs(include_path, chat_data.msgs, permits).await?)
}

impl Msg {
    pub fn is_assistant(&self) -> bool {
        matches!(
            self,
            Msg::Text {
                role: Role::Assistant,
                ..
            }
        )
    }
}
