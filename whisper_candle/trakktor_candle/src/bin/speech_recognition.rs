use candle_core::Device;
use trakktor_candle::speech_recognition::{
    output_provider::{
        TextOutputProvider, TimestampFormat, TimestampedTextOutputProvider,
    },
    run_speech_recognizer, SpeechRecognizerTask, WhichModel,
};

fn main() -> anyhow::Result<()> {
    stderrlog::new()
        .show_module_names(true)
        .module("trakktor_candle::speech_recognition")
        .verbosity(log::Level::Trace)
        .init()?;

    // let output = TimestampedTextOutputProvider::new(
    //     "tmp_data/output.txt",
    //     TimestampFormat::Start,
    // )?;
    let output = TextOutputProvider::new("tmp_data/output.txt")?;

    let input = "/tmp/out.wav";

    run_speech_recognizer(
        SpeechRecognizerTask {
            models_data_dir: "./models_data".into(),
            model: WhichModel::LargeV3,
            device: Device::new_metal(0)?,
            input: input.into(),
            language: Some("ru".into()),
            seed: None,
        },
        Box::new(output),
    )?;

    Ok(())
}
