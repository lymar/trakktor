use std::error::Error;

use async_openai::{
    Client,
    types::chat::{
        ChatCompletionRequestSystemMessage, ChatCompletionRequestUserMessage,
        CreateChatCompletionRequestArgs, ResponseFormat,
        ResponseFormatJsonSchema,
    },
};
use serde_json::json;

// export OPENAI_API_KEY=

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let client = Client::new();

    let schema = json!({
      "type": "object",
      "properties": {
        "steps": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "explanation": { "type": "string" },
              "output": { "type": "string" }
            },
            "required": ["explanation", "output"],
            "additionalProperties": false
          }
        },
        "final_answer": { "type": "string" }
      },
      "required": ["steps", "final_answer"],
      "additionalProperties": false
    });

    let response_format = ResponseFormat::JsonSchema {
        json_schema: ResponseFormatJsonSchema {
            description: None,
            name: "math_reasoning".into(),
            schema: Some(schema),
            strict: Some(true),
        },
    };

    let request = CreateChatCompletionRequestArgs::default()
        .model("gpt-5-mini")
        .messages([
            ChatCompletionRequestSystemMessage::from(
                "You are a helpful math tutor. Guide the user through the \
                 solution step by step.",
            )
            .into(),
            ChatCompletionRequestUserMessage::from(
                "how can I solve 8x + 7 = -23",
            )
            .into(),
        ])
        .response_format(response_format)
        .build()?;

    let response = client.chat().create(request).await?;

    for choice in response.choices {
        if let Some(content) = choice.message.content {
            print!("{content}")
        }
    }

    Ok(())
}
