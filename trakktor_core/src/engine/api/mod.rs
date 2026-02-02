use boa_engine::{Context, NativeFunction, js_string};

pub(crate) mod args;
mod chat;
mod file;
mod http;
mod markdown;
mod preview;

pub(super) fn register(context: &mut Context) -> anyhow::Result<()> {
    context
        .register_global_builtin_callable(
            js_string!("httpGetText"),
            1,
            NativeFunction::from_async_fn(http::http_get_text),
        )
        .map_err(|e| anyhow::anyhow!("Failed to register httpGetText: {e}"))?;

    context
        .register_global_builtin_callable(
            js_string!("htmlToMarkdown"),
            1,
            NativeFunction::from_fn_ptr(markdown::html_to_markdown),
        )
        .map_err(|e| anyhow::anyhow!("Failed to register httpGetText: {e}"))?;

    context
        .register_global_builtin_callable(
            js_string!("chat"),
            2,
            NativeFunction::from_async_fn(chat::chat),
        )
        .map_err(|e| anyhow::anyhow!("Failed to register chat: {e}"))?;

    context
        .register_global_builtin_callable(
            js_string!("readFile"),
            1,
            NativeFunction::from_async_fn(file::read_file),
        )
        .map_err(|e| anyhow::anyhow!("Failed to register readFile: {e}"))?;
    Ok(())
}
