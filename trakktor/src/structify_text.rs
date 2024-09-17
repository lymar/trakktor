use std::{
    borrow::Cow,
    collections::{BTreeMap, HashSet},
    path::Path,
    sync::Arc,
};

use anyhow::bail;
use clap::Parser;
use itertools::Itertools;
use redb::TableDefinition;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use tokio::task::spawn_blocking;

use crate::{
    hasher::get_hash_value,
    llm::{ChatCompletionAPI, ChatCompletionsArgs, Message, Role},
};

#[derive(Parser, Debug)]
pub struct StructifyText {
    /// The text file to structify.
    #[arg(long, short)]
    pub file: std::path::PathBuf,
}

const CHUNK_WORDS_THRESHOLD: usize = 1000;
const CACHE_FILE_EXT: &str = "trakktor.cache";
const RESULT_FILE_EXT: &str = "trakktor.text.md";
const PARAGRAPHS_SUMMARY_FILE_EXT: &str = "trakktor.summaries.md";

pub async fn run_structify_text(
    args: &StructifyText,
    chat_api: &Box<dyn ChatCompletionAPI>,
) -> anyhow::Result<()> {
    let input_text = tokio::fs::read_to_string(&args.file).await?;

    let cache = Arc::new({
        let db_name = args.file.with_extension(CACHE_FILE_EXT);
        spawn_blocking(move || CallCache::open(&db_name)).await??
    });

    let result_paragraphs = words_to_paragraphs(
        chat_api,
        &cache,
        input_text.split_whitespace().map(|c| c.to_string()),
    )
    .await?;

    let full_text_file = args.file.with_extension(RESULT_FILE_EXT);
    tokio::fs::write(&full_text_file, &result_paragraphs.join("\n\n")).await?;

    tracing::info!("Wrote structified text to: {}", full_text_file.display());

    create_titles(args, chat_api, &cache, &result_paragraphs).await?;

    Ok(())
}

async fn create_titles(
    args: &StructifyText,
    chat_api: &Box<dyn ChatCompletionAPI>,
    cache: &Arc<CallCache>,
    result_paragraphs: &[String],
) -> anyhow::Result<()> {
    // Short summaries of each paragraph
    let result_summaries =
        summarize_paragraphs(chat_api, &cache, &result_paragraphs).await?;

    // Write summaries to a file
    let summaries_file = args.file.with_extension(PARAGRAPHS_SUMMARY_FILE_EXT);
    tokio::fs::write(&summaries_file, &result_summaries.join("\n\n")).await?;

    // Split summaries into paragraphs

    // Pairs of (paragraph index, word)
    let summaries_words = result_summaries
        .iter()
        .enumerate()
        .map(|(i, s)| s.split_whitespace().map(move |c| (i, c.to_string())))
        .flatten()
        .collect::<Vec<_>>();
    let sections = words_to_paragraphs(
        chat_api,
        &cache,
        summaries_words.iter().map(|s| &s.1).cloned(),
    )
    .await?;

    // ************ todo: надо переименовать файл
    let sections_file = args.file.with_extension("trakktor.sections.md");
    tokio::fs::write(&sections_file, &sections.join("\n\n")).await?;

    tracing::info!("Wrote summaries to: {}", summaries_file.display());
    // ************

    // Key: section index, Value: paragraph index -> word count in section
    let mut section_par_words: Vec<BTreeMap<usize, usize>> =
        vec![Default::default(); sections.len()];

    let mut summaries_words_p = &summaries_words[..];
    for (section, par_words) in
        sections.iter().zip(section_par_words.iter_mut())
    {
        let current_text = summaries_words_p.iter().map(|s| &s.1).join(" ");
        let next =
            get_next_text_words(cache, &[section.clone()], &current_text)
                .await?;

        for (n, _) in &summaries_words_p[..=next.skipped_words] {
            par_words.entry(*n).and_modify(|n| *n += 1).or_insert(1);
        }

        summaries_words_p = &summaries_words_p[next.skipped_words + 1..];
    }

    // println!("{:?}\n", section_par_words);

    let mut par_in_section: BTreeMap<usize, (usize, usize)> = BTreeMap::new();
    for (sec_i, pars_words) in section_par_words.iter().enumerate() {
        for (res_par_i, words) in pars_words {
            par_in_section
                .entry(*res_par_i)
                .and_modify(|(sec, count)| {
                    if words > count {
                        *sec = sec_i;
                        *count = *words;
                    }
                })
                .or_insert((sec_i, *words));
        }
    }

    let pars = par_in_section
        .into_iter()
        .map(|(par, (sec, _))| (sec, par))
        .chunk_by(|(sec, _)| *sec);

    let mut final_sections = vec![];
    let mut par_usage = HashSet::new();
    for (_, ch) in &pars {
        let par_in_section = ch.map(|c| c.1).collect::<Vec<_>>();

        let mut par_text = vec![];
        for p in &par_in_section {
            par_usage.insert(*p);
            par_text.push(result_paragraphs[*p].clone());
        }

        final_sections.push(par_text);
    }

    if par_usage.len() != result_paragraphs.len() {
        bail!("Not all paragraphs were used in the sections!");
    }

    let mut text_with_sections = String::new();
    for sec in &final_sections {
        // if text_with_sections.len() > 0 {
        //     text_with_sections.push_str("\n\n");
        // }

        let title = get_section_title(chat_api, cache, sec).await?;
        text_with_sections.push_str(&format!("###### {}\n\n", title.trim()));

        for par in sec {
            text_with_sections.push_str(&format!("{}\n\n", par));
        }

        // text_with_sections.push_str(&sec.join("\n\n"));
        // text_with_sections.push_str("\n\n");
    }

    // ************ todo: надо переименовать файл
    let final_file = args.file.with_extension("trakktor.final.md");
    tokio::fs::write(&final_file, &text_with_sections).await?;

    // tracing::info!("Wrote summaries to: {}", summaries_file.display());
    // ************

    Ok(())
}

async fn get_section_title(
    chat_api: &Box<dyn ChatCompletionAPI>,
    cache: &Arc<CallCache>,
    paragraphs: &[String],
) -> anyhow::Result<String> {
    let call_hash = Arc::new(get_hash_value(format!(
        "get_section_title:\n{}\n\n{}\n\n{:?}",
        chat_api.config_hash(),
        GET_SECTION_TITLE_PROMPT,
        paragraphs,
    )));

    if let Some(summary) = cache.get_data::<String>(&call_hash).await? {
        tracing::debug!("Using cached section title");
        Ok(summary)
    } else {
        let section_text = paragraphs.join("\n\n");
        let summary = chat_api
            .run_chat(
                ChatCompletionsArgs::builder()
                    .messages(&[
                        Message {
                            role: Role::System,
                            content: Cow::Borrowed(
                                &GET_SECTION_TITLE_PROMPT.trim(),
                                // &SUMMARIZE_PARAGRAPH_PROMPT.trim(),
                            ),
                        },
                        Message {
                            role: Role::User,
                            content: Cow::Borrowed(&section_text),
                        },
                    ])
                    .build(),
            )
            .await?
            .content
            .to_string();
        let summary = Arc::new(summary);
        cache.put_data(&call_hash, &summary).await?;
        Ok(Arc::into_inner(summary).unwrap())
    }
}

async fn words_to_paragraphs(
    chat_api: &Box<dyn ChatCompletionAPI>,
    cache: &Arc<CallCache>,
    words: impl Iterator<Item = String>,
) -> anyhow::Result<Vec<String>> {
    let mut all_words = Arc::new(words.collect::<Vec<_>>());

    if all_words.len() < 1 {
        bail!("No words found in the input text!");
    }

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
            get_paragraphs(Arc::clone(&cache), chat_api, &llm_text).await?;

        if last_chunk {
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
        result_paragraphs.extend(accepted_paragraphs.iter().cloned());

        all_words = get_next_text_words(cache, accepted_paragraphs, &orig_text)
            .await?
            .words;
    }

    Ok(result_paragraphs)
}

async fn summarize_paragraphs(
    chat_api: &Box<dyn ChatCompletionAPI>,
    cache: &Arc<CallCache>,
    paragraphs: &[String],
) -> anyhow::Result<Vec<String>> {
    let mut result_summaries: Vec<String> = Vec::new();

    for src_par in paragraphs {
        let call_hash = Arc::new(get_hash_value(format!(
            "summarize_paragraphs:\n{}\n\n{}\n\n{}",
            chat_api.config_hash(),
            SUMMARIZE_PARAGRAPH_PROMPT,
            src_par,
        )));

        if let Some(summary) = cache.get_data::<String>(&call_hash).await? {
            tracing::debug!("Using cached summary");
            result_summaries.push(summary);
        } else {
            let summary = chat_api
                .run_chat(
                    ChatCompletionsArgs::builder()
                        .messages(&[
                            Message {
                                role: Role::System,
                                content: Cow::Borrowed(
                                    &SUMMARIZE_PARAGRAPH_PROMPT.trim(),
                                ),
                            },
                            Message {
                                role: Role::User,
                                content: Cow::Borrowed(src_par),
                            },
                        ])
                        .build(),
                )
                .await?
                .content
                .to_string();
            let summary = Arc::new(summary);
            cache.put_data(&call_hash, &summary).await?;
            result_summaries.push(Arc::into_inner(summary).unwrap());
        }
    }

    Ok(result_summaries)
}

fn push_word(text: &mut String, word: &str) {
    if !text.is_empty() {
        text.push(' ');
    }
    text.push_str(word);
}

#[derive(Debug, Serialize, Deserialize)]
struct NextTextWordsRes {
    words: Arc<Vec<String>>,
    skipped_words: usize,
    distance: usize,
}

// LLM's are not perfect, they tend to change the text slightly. We need to
// compare the original text with the LLM's output to determine the next
// chunk of text to process.
async fn get_next_text_words(
    call_cache: &Arc<CallCache>,
    accepted_paragraphs: &[String],
    orig_text: &str,
) -> anyhow::Result<NextTextWordsRes> {
    let call_hash = Arc::new(get_hash_value(format!(
        "get_next_text_words:\n{}\n-----\n{}",
        accepted_paragraphs.join("\n"),
        orig_text
    )));

    if let Some(cached_res) =
        call_cache.get_data::<NextTextWordsRes>(&call_hash).await?
    {
        return Ok(cached_res);
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

    let min_e = dist.iter().min_by_key(|(_, d)| *d).unwrap();

    let next_words = Arc::new(
        orig_part_words
            .iter()
            .skip(min_e.0 + 1)
            .map(|s| s.to_string())
            .collect::<Vec<_>>(),
    );

    let res = Arc::new(NextTextWordsRes {
        words: next_words.into(),
        skipped_words: min_e.0 as usize,
        distance: min_e.1,
    });

    call_cache.put_data(&call_hash, &res).await?;

    Ok(Arc::try_unwrap(res).unwrap())
}

async fn get_paragraphs(
    call_cache: Arc<CallCache>,
    chat_api: &Box<dyn ChatCompletionAPI>,
    text: &str,
) -> anyhow::Result<Arc<Vec<String>>> {
    let call_hash = Arc::new(get_hash_value(format!(
        "get_paragraphs:\n{}\n\n{}\n\n{}",
        chat_api.config_hash(),
        STRUCTIFY_PROMPT,
        text
    )));

    if let Some(paragraphs) =
        call_cache.get_data::<Vec<String>>(&call_hash).await?
    {
        tracing::debug!("Using cached paragraphs");
        return Ok(paragraphs.into());
    }

    let content = {
        let mut retry_number = 1;
        let mut retries = vec![];

        loop {
            let content = chat_api
                .run_chat(
                    ChatCompletionsArgs::builder()
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

    call_cache.put_data(&call_hash, &paragraphs).await?;

    Ok(paragraphs)
}

struct CallCache {
    db: redb::Database,
}

const KV_TABLE: TableDefinition<&str, Vec<u8>> =
    TableDefinition::new("kv_table");

impl CallCache {
    fn open(file_path: &Path) -> anyhow::Result<Self> {
        let db = redb::Database::create(file_path)?;
        let write_txn = db.begin_write()?;

        let table = write_txn.open_table(KV_TABLE)?;
        drop(table);

        write_txn.commit()?;

        Ok(Self { db })
    }

    async fn get_data<T>(
        self: &Arc<Self>,
        call_hash: &Arc<String>,
    ) -> anyhow::Result<Option<T>>
    where
        T: DeserializeOwned + Send + 'static,
    {
        let cache = Arc::clone(self);
        let call_hash = Arc::clone(call_hash);
        spawn_blocking(move || cache.get_data_sync::<T>(&call_hash)).await?
    }

    fn get_data_sync<T>(&self, call_hash: &str) -> anyhow::Result<Option<T>>
    where
        T: DeserializeOwned,
    {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(KV_TABLE)?;

        let data = table.get(call_hash)?;
        if let Some(data) = data {
            Ok(Some(rmp_serde::from_slice(&data.value())?))
        } else {
            Ok(None)
        }
    }

    async fn put_data<T>(
        self: &Arc<Self>,
        call_hash: &Arc<String>,
        data: &Arc<T>,
    ) -> anyhow::Result<()>
    where
        T: Serialize + Send + Sync + 'static,
    {
        let cache = Arc::clone(self);
        let call_hash = Arc::clone(call_hash);
        let data = Arc::clone(data);
        spawn_blocking(move || cache.put_data_sync::<T>(&call_hash, &data))
            .await?
    }

    fn put_data_sync<T>(&self, call_hash: &str, data: &T) -> anyhow::Result<()>
    where
        T: Serialize,
    {
        let write_txn = self.db.begin_write()?;

        {
            let mut table = write_txn.open_table(KV_TABLE)?;

            let data = rmp_serde::to_vec(data)?;
            table.insert(call_hash, data)?;
        }

        write_txn.commit()?;
        Ok(())
    }
}

// const STRUCTIFY_PROMPT: &str = r#"""
// You are an AI assistant tasked with splitting any text input into paragraphs.
// Each paragraph should be separated by exactly one blank line. When breaking
// the text into paragraphs, aim for logical divisions based on context, topic
// shifts, or natural pauses in the flow of ideas. Make sure the output follows
// this format:

// - One empty line between paragraphs.
// - Do not alter the content of the text; just format it into paragraphs.

// Example input:
// ```
// This is the first sentence. Here is some additional text. This is another idea.
// Now we are shifting to a new point. Another sentence follows this one. Conclusion here.
// ```

// Example output:
// ```
// This is the first sentence. Here is some additional text. This is another idea.

// Now we are shifting to a new point. Another sentence follows this one.

// Conclusion here.
// ```

// Make sure the output text maintains this format regardless of the input.

// """#;

const STRUCTIFY_PROMPT: &str = r#"""
You are an AI assistant tasked with splitting any text input into paragraphs. Your goal is to format the text by inserting paragraph breaks at logical points without altering the original content in any way. Each paragraph should be separated by exactly one blank line. Follow these guidelines when breaking the text into paragraphs:

- **Logical Divisions:** Insert paragraph breaks where there are shifts in topic, introduction of new ideas, changes in time or place, or natural pauses in the narrative.
- **Preserve Original Formatting:** Do not change any words, punctuation, capitalization, or spacing within sentences. Maintain any existing paragraph or line breaks.
- **Consistent Output Format:** Ensure the output text matches the input exactly in content, with the only changes being the insertion of paragraph breaks as specified.
- **Special Cases:** For texts that are very short or lack clear division points, use your best judgment to determine if paragraph breaks are necessary.

*Example input:*

This is the first sentence. Here is some additional text. This is another idea.
Now we are shifting to a new point. Another sentence follows this one. Conclusion here.

*Example output:*

This is the first sentence. Here is some additional text. This is another idea.

Now we are shifting to a new point. Another sentence follows this one.

Conclusion here.


Ensure that the output text maintains this format regardless of the input, and remember not to alter the content in any way—only adjust the paragraph formatting.
"""#;

const SUMMARIZE_PARAGRAPH_PROMPT: &str = r#"""
When given a text, provide a brief summary in one sentence no longer than 20
words, using the same language as the original text. """#;

const GET_SECTION_TITLE_PROMPT: &str = r#"""
Your task is to generate a headline for the provided text. The headline should capture the main idea and key points clearly and concisely, using simple language. Make sure the headline is a single sentence, do not use quotation marks around it, and use the same language as the text.
"""#;
