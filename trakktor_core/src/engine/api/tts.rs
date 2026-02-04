use std::cell::RefCell;

use async_openai::{
    Client,
    types::audio::{
        CreateSpeechRequestArgs, SpeechModel, SpeechResponseFormat, Voice,
    },
};
use boa_engine::{
    Context, JsArgs, JsError, JsNativeError, JsObject, JsResult, JsValue,
};
use serde::Deserialize;

use crate::artifact::Artifact;

#[derive(Debug, Deserialize, Default)]
struct TTSOptions {
    model: Option<String>,
    voice: Option<String>,
    speed: Option<f32>,
}

pub(super) fn tts(
    _this: &JsValue,
    args: &[JsValue],
    context: &RefCell<&mut Context>,
) -> impl Future<Output = JsResult<JsValue>> {
    async move {
        let Some(text) = args
            .get(0)
            .map(|v| v.to_string(&mut context.borrow_mut()).ok())
            .flatten()
            .map(|s| s.to_std_string().ok())
            .flatten()
        else {
            return Err(JsNativeError::typ()
                .with_message("First argument to textToSpeech must be a string")
                .into());
        };

        let opts = args
            .get_or_undefined(1)
            .to_json(&mut context.borrow_mut())?
            .map(|v| serde_json::from_value::<TTSOptions>(v).ok())
            .flatten()
            .unwrap_or_default();

        log::debug!(text:?, opts:?; "TTS requested for text");

        let model = opts
            .model
            .unwrap_or_else(|| "gpt-4o-mini-tts-2025-12-15".into());
        let voice = opts.voice.unwrap_or_else(|| "cedar".into());
        let speed = opts.speed.unwrap_or(1.0);

        let client = Client::new();

        let request = CreateSpeechRequestArgs::default()
            .input(&text)
            .voice(Voice::Other(voice))
            .model(SpeechModel::Other(model))
            .response_format(SpeechResponseFormat::Mp3)
            .speed(speed)
            .build()
            .map_err(|_| -> JsError {
                JsNativeError::error()
                    .with_message("Failed to build text-to-speech request")
                    .into()
            })?;

        let response = client.audio().speech().create(request).await.map_err(
            |_| -> JsError {
                JsNativeError::error()
                    .with_message("Failed to create text-to-speech audio")
                    .into()
            },
        )?;

        let afc = Artifact {
            data: response.bytes.to_vec().into(),
            mime: Some("audio/mpeg".parse().unwrap()),
            original_name: None,
        };

        log::debug!("TTS audio generated");

        // tokio::fs::write("tts.mp3", &afc.data).await.unwrap();

        Ok(JsObject::from_proto_and_data(None, afc).into())
    }
}
