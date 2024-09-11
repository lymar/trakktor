use std::{borrow::Cow, path::Path, sync::Arc};

use anyhow::bail;
use clap::Parser;
use itertools::Itertools;
use redb::TableDefinition;
use tokio::task::spawn_blocking;

use crate::{
    hasher::get_hash_value,
    llm::{ChatCompletions, ChatCompletionsProvider, Message, Role},
};

#[derive(Parser, Debug)]
pub struct StructifyText {
    /// The text file to structify.
    #[arg(long, short)]
    pub file: std::path::PathBuf,
}

const CHUNK_WORDS_THRESHOLD: usize = 1000;

pub async fn run_structify_text(
    args: &StructifyText,
    chat_provider: &Box<dyn ChatCompletionsProvider>,
) -> anyhow::Result<()> {
    let input_text = tokio::fs::read_to_string(&args.file).await?;
    let mut all_words = Arc::new(
        input_text
            .split_whitespace()
            .map(|c| c.to_string())
            .collect::<Vec<_>>(),
    );

    if all_words.len() < 1 {
        bail!("No words found in the input text!");
    }

    let cache = Arc::new({
        let db_name = args.file.with_extension("trakktor.cache");
        spawn_blocking(move || CallCache::open(&db_name)).await??
    });

    let mut result_paragraphs: Vec<String> = Vec::new();

    loop {
        let mut llm_text = String::new();
        let mut orig_text = String::new();
        for i in 0..all_words.len() {
            push_word(&mut orig_text, &all_words[i]);
            if i < CHUNK_WORDS_THRESHOLD {
                push_word(&mut llm_text, &all_words[i]);
            }
        }

        let last_chunk = llm_text == orig_text;

        let paragraphs =
            get_paragraphs(Arc::clone(&cache), chat_provider, &llm_text)
                .await?;

        // println!("{:#?}", paragraphs);

        if last_chunk {
            // todo: выводить если указан флаг dev
            result_paragraphs.push("**************************".into());
            result_paragraphs.extend(paragraphs.iter().cloned());
            break;
        }

        if paragraphs.len() < 2 {
            bail!(
                "Failed to split text into paragraphs: too few paragraphs \
                 returned!"
            );
        }

        let accepted_paragraphs = &paragraphs[0..paragraphs.len() - 1];
        // todo: выводить если указан флаг dev
        result_paragraphs.push("**************************".into());
        result_paragraphs.extend(accepted_paragraphs.iter().cloned());

        all_words = get_next_text_words(
            Arc::clone(&cache),
            accepted_paragraphs,
            &orig_text,
        )
        .await?;
    }

    let full_text_file = args.file.with_extension("trakktor.text.md");
    tokio::fs::write(&full_text_file, &result_paragraphs.join("\n\n")).await?;

    tracing::info!("Wrote structified text to: {}", full_text_file.display());

    Ok(())
}

fn push_word(text: &mut String, word: &str) {
    if !text.is_empty() {
        text.push(' ');
    }
    text.push_str(word);
}

// LLM's are not perfect, they tend to change the text slightly. We need to
// compare the original text with the LLM's output to determine the next
// chunk of text to process.
async fn get_next_text_words(
    call_cache: Arc<CallCache>,
    accepted_paragraphs: &[String],
    orig_text: &str,
) -> anyhow::Result<Arc<Vec<String>>> {
    let call_hash = Arc::new(get_hash_value(format!(
        "get_next_text_words:\n{}\n-----\n{}",
        accepted_paragraphs.join("\n"),
        orig_text
    )));

    if let Some(words) = spawn_blocking({
        let call_cache = Arc::clone(&call_cache);
        let call_hash = Arc::clone(&call_hash);
        move || call_cache.get_vec_string(&call_hash)
    })
    .await??
    {
        return Ok(words.into());
    }

    let llm_res_words = accepted_paragraphs
        .iter()
        .map(|p| p.split_whitespace())
        .flatten()
        .filter(|w| !w.is_empty())
        .collect::<Vec<_>>();
    let llm_res_words_text = llm_res_words.iter().join(" ");

    let res_words_count = llm_res_words.len() as isize;

    const CHECK_WORDS: isize = 25;
    let orig_from_idx = res_words_count - CHECK_WORDS;
    let orig_to_idx = res_words_count + CHECK_WORDS;

    let orig_part_words = orig_text.split_whitespace().collect::<Vec<_>>();

    let mut orig_fragment = String::new();
    let mut dist = vec![];

    for i in 0..orig_part_words.len() {
        if orig_fragment.len() > 0 {
            orig_fragment.push(' ');
        }
        orig_fragment.push_str(orig_part_words[i]);

        if (i as isize) >= orig_from_idx {
            dist.push((
                i,
                edit_distance::edit_distance(
                    &llm_res_words_text,
                    &orig_fragment,
                ),
            ));
        }

        if (i as isize) >= orig_to_idx {
            break;
        }
    }

    // println!("{:?}", dist);

    let min_e = dist.iter().min_by_key(|(_, d)| *d).unwrap();

    // println!("{:?}", min_e);

    let next_words = Arc::new(
        orig_part_words
            .iter()
            .skip(min_e.0 + 1)
            .map(|s| s.to_string())
            .collect::<Vec<_>>(),
    );

    spawn_blocking({
        let words = Arc::clone(&next_words);
        move || call_cache.insert_vec_string(&call_hash, &words)
    })
    .await??;

    Ok(next_words.into())
}

async fn get_paragraphs(
    call_cache: Arc<CallCache>,
    chat_provider: &Box<dyn ChatCompletionsProvider>,
    text: &str,
) -> anyhow::Result<Arc<Vec<String>>> {
    let call_hash = Arc::new(get_hash_value(format!(
        "get_paragraphs:\n{}\n{}",
        chat_provider.config_hash(),
        text
    )));

    if let Some(paragraphs) = spawn_blocking({
        let call_cache = Arc::clone(&call_cache);
        let call_hash = Arc::clone(&call_hash);
        move || call_cache.get_vec_string(&call_hash)
    })
    .await??
    {
        tracing::debug!("Using cached paragraphs");
        return Ok(paragraphs.into());
    }

    let content = {
        let mut retry_number = 1;
        let mut retries = vec![];

        loop {
            let content = chat_provider
                .run_chat(
                    ChatCompletions::builder()
                        .messages(&[
                            Message {
                                role: Role::System,
                                content: Cow::Borrowed(
                                    &STRUCTIFY_PROMPT.trim(),
                                ),
                            },
                            Message {
                                role: Role::User,
                                content: Cow::Borrowed(text),
                            },
                        ])
                        .build(),
                )
                .await?
                .content;
            let res_text = content.split_whitespace().join(" ");
            let distance = edit_distance::edit_distance(&res_text, &text);
            const MAX_RETRIES: usize = 10;
            tracing::info!(
                "Paragraphs Levenshtein distance: {}, retry number: {} (of {})",
                distance,
                retry_number,
                MAX_RETRIES
            );
            const MAX_DISTANCE: usize = 64;
            if distance > MAX_DISTANCE {
                retries.push((distance, content));
                if retry_number > (MAX_RETRIES - 1) {
                    let best_res =
                        retries.into_iter().min_by_key(|r| r.0).unwrap();
                    tracing::warn!(
                        "Too many changes. I give up. Using the best result \
                         with distance: {}",
                        best_res.0
                    );
                    break best_res.1;
                }
                tracing::warn!(
                    "Paragraphs distance too high: {}, retrying...",
                    distance
                );
                retry_number += 1;
            } else {
                break content;
            }
        }
    };

    let paragraphs = content
        .lines()
        .map(|l| l.trim())
        .chunk_by(|l| !l.is_empty())
        .into_iter()
        .filter_map(
            |(v, mut lines)| {
                if v {
                    Some(lines.join(" "))
                } else {
                    None
                }
            },
        )
        .collect::<Vec<_>>();
    let paragraphs = Arc::new(paragraphs);

    spawn_blocking({
        let paragraphs = Arc::clone(&paragraphs);
        move || call_cache.insert_vec_string(&call_hash, &paragraphs)
    })
    .await??;

    Ok(paragraphs)
}

struct CallCache {
    db: redb::Database,
}

const VEC_STRING_TABLE: TableDefinition<&str, Vec<u8>> =
    TableDefinition::new("vec_string_table");

impl CallCache {
    fn open(file_path: &Path) -> anyhow::Result<Self> {
        let db = redb::Database::create(file_path)?;
        let write_txn = db.begin_write()?;

        let table = write_txn.open_table(VEC_STRING_TABLE)?;
        drop(table);

        write_txn.commit()?;

        Ok(Self { db })
    }

    fn get_vec_string(
        &self,
        call_hash: &str,
    ) -> anyhow::Result<Option<Vec<String>>> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(VEC_STRING_TABLE)?;

        let data = table.get(call_hash)?;
        if let Some(data) = data {
            let vec_string = rmp_serde::from_slice(&data.value())?;
            Ok(Some(vec_string))
        } else {
            Ok(None)
        }
    }

    fn insert_vec_string(
        &self,
        call_hash: &str,
        vec_string: &[String],
    ) -> anyhow::Result<()> {
        let write_txn = self.db.begin_write()?;

        {
            let mut table = write_txn.open_table(VEC_STRING_TABLE)?;

            let data = rmp_serde::to_vec(vec_string)?;
            table.insert(call_hash, data)?;
        }

        write_txn.commit()?;
        Ok(())
    }
}

const STRUCTIFY_PROMPT: &str = r#"""
You are an AI assistant tasked with splitting any text input into paragraphs. Each paragraph should be separated by exactly one blank line. When breaking the text into paragraphs, aim for logical divisions based on context, topic shifts, or natural pauses in the flow of ideas. Make sure the output follows this format: 

- One empty line between paragraphs.
- Do not alter the content of the text; just format it into paragraphs.

Example input:
```
This is the first sentence. Here is some additional text. This is another idea.
Now we are shifting to a new point. Another sentence follows this one. Conclusion here.
```

Example output:
```
This is the first sentence. Here is some additional text. This is another idea.

Now we are shifting to a new point. Another sentence follows this one.

Conclusion here.
```

Make sure the output text maintains this format regardless of the input.

"""#;
