use std::cell::RefCell;

use async_openai::{
    Client,
    types::chat::{
        ChatCompletionRequestMessage, CreateChatCompletionRequestArgs,
        ResponseFormat, ResponseFormatJsonSchema,
    },
};
use boa_engine::{
    Context, JsArgs, JsError, JsNativeError, JsResult, JsString, JsValue,
};
use serde::Deserialize;

use crate::engine::api::args::get_json_arg;

#[derive(Debug, Deserialize)]
struct ChatOptions {
    json_schema: Option<ResponseFormatJsonSchema>,
}

pub(super) fn chat(
    _this: &JsValue,
    args: &[JsValue],
    context: &RefCell<&mut Context>,
) -> impl Future<Output = JsResult<JsValue>> {
    async move {
        let chat_msgs = get_json_arg(
            context,
            args,
            0,
            "First argument to chat must be a JSON object",
        )?;
        let chat_msgs: Vec<ChatCompletionRequestMessage> =
            serde_json::from_value(chat_msgs).map_err(|_| -> JsError {
                JsNativeError::error()
                    .with_message(
                        "Invalid chat messages: failed to deserialize \
                         ChatCompletionRequestMessage array.",
                    )
                    .into()
            })?;

        let opts = args
            .get_or_undefined(1)
            .to_json(&mut context.borrow_mut())?;

        let mut request = CreateChatCompletionRequestArgs::default()
            .model("gpt-5-mini")
            .messages(chat_msgs)
            .build()
            .map_err(|_| -> JsError {
                JsNativeError::error()
                    .with_message("Failed to build chat completion request")
                    .into()
            })?;

        let mut with_json_schema = false;

        if let Some(opts) = opts {
            let co = serde_json::from_value::<ChatOptions>(opts).map_err(
                |_| -> JsError {
                    JsNativeError::error()
                        .with_message(
                            "Invalid chat options: failed to deserialize \
                             ChatOptions.",
                        )
                        .into()
                },
            )?;

            if let Some(json_schema) = co.json_schema {
                request.response_format =
                    Some(ResponseFormat::JsonSchema { json_schema });
                with_json_schema = true;
            }
        }

        let client = Client::new();
        let response =
            client
                .chat()
                .create(request)
                .await
                .map_err(|_| -> JsError {
                    JsNativeError::error()
                        .with_message("Failed to create chat completion")
                        .into()
                })?;

        let res = response
            .choices
            .get(0)
            .map(|v| v.message.content.clone())
            .flatten();

        if let Some(mut content) = res {
            if with_json_schema {
                let v: serde_json::Value = serde_json::from_str(&content)
                    .map_err(|_| -> JsError {
                        JsNativeError::error()
                            .with_message(
                                "Failed to parse chat completion content as \
                                 JSON",
                            )
                            .into()
                    })?;
                content = serde_json::to_string_pretty(&v).map_err(
                    |_| -> JsError {
                        JsNativeError::error()
                            .with_message(
                                "Failed to serialize chat completion content \
                                 as pretty JSON",
                            )
                            .into()
                    },
                )?;
            }
            let res = JsString::from(content).into();
            Ok(res)
        } else {
            Err(JsNativeError::error()
                .with_message("No content in chat completion response")
                .into())
        }
    }
}
