// Based on https://github.com/huggingface/candle/tree/main/candle-examples/examples/whisper

use anyhow::{Error as E, Result};
use candle_core::{self as candle, Device, IndexOp, Tensor};
use candle_nn::{ops::softmax, VarBuilder};
use candle_transformers::models::whisper::{self as m, audio, Config};
use enum_iterator::Sequence;
use rand::{distributions::Distribution, SeedableRng};
use tokenizers::Tokenizer;

use crate::speech_recognition::output_provider::{
    DecodingResult, Segment, SpeechRecognitionOutputProvider,
};

mod multilingual;
pub mod output_provider;
mod pcm_decode;

#[derive(Sequence, Clone, Copy, Debug)]
pub enum DataFile {
    Config,
    Tokenizer,
    Model,
}

impl DataFile {
    pub fn file_name(self) -> &'static str {
        match self {
            Self::Config => "config.json",
            Self::Tokenizer => "tokenizer.json",
            Self::Model => "model.safetensors",
        }
    }
}

pub enum Model {
    Normal(m::model::Whisper),
    Quantized(m::quantized_model::Whisper),
}

// Maybe we should use some traits rather than doing the dispatch for all these.
impl Model {
    pub fn config(&self) -> &Config {
        match self {
            Self::Normal(m) => &m.config,
            Self::Quantized(m) => &m.config,
        }
    }

    pub fn encoder_forward(
        &mut self,
        x: &Tensor,
        flush: bool,
    ) -> candle::Result<Tensor> {
        match self {
            Self::Normal(m) => m.encoder.forward(x, flush),
            Self::Quantized(m) => m.encoder.forward(x, flush),
        }
    }

    pub fn decoder_forward(
        &mut self,
        x: &Tensor,
        xa: &Tensor,
        flush: bool,
    ) -> candle::Result<Tensor> {
        match self {
            Self::Normal(m) => m.decoder.forward(x, xa, flush),
            Self::Quantized(m) => m.decoder.forward(x, xa, flush),
        }
    }

    pub fn decoder_final_linear(&self, x: &Tensor) -> candle::Result<Tensor> {
        match self {
            Self::Normal(m) => m.decoder.final_linear(x),
            Self::Quantized(m) => m.decoder.final_linear(x),
        }
    }
}

struct Decoder {
    model: Model,
    rng: rand::rngs::StdRng,
    task: Option<Task>,
    timestamps: bool, // TODO: удалить, не поддерживаю
    verbose: bool,    // TODO: тоже удалить
    tokenizer: Tokenizer,
    suppress_tokens: Tensor,
    sot_token: u32,
    transcribe_token: u32,
    translate_token: u32,
    eot_token: u32,
    no_speech_token: u32,
    no_timestamps_token: u32,
    language_token: Option<u32>,
}

impl Decoder {
    #[allow(clippy::too_many_arguments)]
    fn new(
        model: Model,
        tokenizer: Tokenizer,
        seed: u64,
        device: &Device,
        language_token: Option<u32>,
        task: Option<Task>,
        timestamps: bool,
        verbose: bool,
    ) -> Result<Self> {
        let no_timestamps_token = token_id(&tokenizer, m::NO_TIMESTAMPS_TOKEN)?;
        // Suppress the notimestamps token when in timestamps mode.
        // https://github.com/openai/whisper/blob/e8622f9afc4eba139bf796c210f5c01081000472/whisper/decoding.py#L452
        let suppress_tokens: Vec<f32> = (0..model.config().vocab_size as u32)
            .map(|i| {
                if model.config().suppress_tokens.contains(&i) ||
                    timestamps && i == no_timestamps_token
                {
                    f32::NEG_INFINITY
                } else {
                    0f32
                }
            })
            .collect();
        let suppress_tokens = Tensor::new(suppress_tokens.as_slice(), device)?;
        let sot_token = token_id(&tokenizer, m::SOT_TOKEN)?;
        let transcribe_token = token_id(&tokenizer, m::TRANSCRIBE_TOKEN)?;
        let translate_token = token_id(&tokenizer, m::TRANSLATE_TOKEN)?;
        let eot_token = token_id(&tokenizer, m::EOT_TOKEN)?;
        let no_speech_token = m::NO_SPEECH_TOKENS
            .iter()
            .find_map(|token| token_id(&tokenizer, token).ok());
        let no_speech_token = match no_speech_token {
            None => anyhow::bail!("unable to find any non-speech token"),
            Some(n) => n,
        };
        Ok(Self {
            model,
            rng: rand::rngs::StdRng::seed_from_u64(seed),
            tokenizer,
            task,
            timestamps,
            verbose,
            suppress_tokens,
            sot_token,
            transcribe_token,
            translate_token,
            eot_token,
            no_speech_token,
            language_token,
            no_timestamps_token,
        })
    }

    fn decode(&mut self, mel: &Tensor, t: f64) -> Result<DecodingResult> {
        let model = &mut self.model;
        let audio_features = model.encoder_forward(mel, true)?;
        log::info!("audio features: {:?}", audio_features.dims());
        let sample_len = model.config().max_target_positions / 2;
        let mut sum_logprob = 0f64;
        let mut no_speech_prob = f64::NAN;
        let mut tokens = vec![self.sot_token];
        if let Some(language_token) = self.language_token {
            tokens.push(language_token);
        }
        match self.task {
            None | Some(Task::Transcribe) => tokens.push(self.transcribe_token),
            Some(Task::Translate) => tokens.push(self.translate_token),
        }
        // TODO: обратить внимание, тут отрицание!
        if !self.timestamps {
            tokens.push(self.no_timestamps_token);
        }
        for i in 0..sample_len {
            let tokens_t = Tensor::new(tokens.as_slice(), mel.device())?;

            // The model expects a batch dim but this inference loop does not
            // handle it so we add it at this point.
            let tokens_t = tokens_t.unsqueeze(0)?;
            let ys =
                model.decoder_forward(&tokens_t, &audio_features, i == 0)?;

            // Extract the no speech probability on the first iteration by
            // looking at the first token logits and the probability
            // for the according token.
            if i == 0 {
                let logits =
                    model.decoder_final_linear(&ys.i(..1)?)?.i(0)?.i(0)?;
                no_speech_prob = softmax(&logits, 0)?
                    .i(self.no_speech_token as usize)?
                    .to_scalar::<f32>()?
                    as f64;
            }

            let (_, seq_len, _) = ys.dims3()?;
            let logits = model
                .decoder_final_linear(&ys.i((..1, seq_len - 1..))?)?
                .i(0)?
                .i(0)?;
            // TODO: Besides suppress tokens, we should apply the heuristics
            // from ApplyTimestampRules, i.e.:
            // - Timestamps come in pairs, except before EOT.
            // - Timestamps should be non-decreasing.
            // - If the sum of the probabilities of timestamps is higher than
            //   any other tokens, only consider timestamps when sampling.
            // https://github.com/openai/whisper/blob/e8622f9afc4eba139bf796c210f5c01081000472/whisper/decoding.py#L439
            let logits = logits.broadcast_add(&self.suppress_tokens)?;
            let next_token = if t > 0f64 {
                let prs = softmax(&(&logits / t)?, 0)?;
                let logits_v: Vec<f32> = prs.to_vec1()?;
                let distr = rand::distributions::WeightedIndex::new(&logits_v)?;
                distr.sample(&mut self.rng) as u32
            } else {
                let logits_v: Vec<f32> = logits.to_vec1()?;
                logits_v
                    .iter()
                    .enumerate()
                    .max_by(|(_, u), (_, v)| u.total_cmp(v))
                    .map(|(i, _)| i as u32)
                    .unwrap()
            };
            tokens.push(next_token);
            let prob = softmax(&logits, candle::D::Minus1)?
                .i(next_token as usize)?
                .to_scalar::<f32>()? as f64;
            if next_token == self.eot_token ||
                tokens.len() > model.config().max_target_positions
            {
                break;
            }
            sum_logprob += prob.ln();
        }
        let text = self.tokenizer.decode(&tokens, true).map_err(E::msg)?;
        let avg_logprob = sum_logprob / tokens.len() as f64;

        Ok(DecodingResult {
            tokens,
            text,
            avg_logprob,
            no_speech_prob,
            temperature: t,
            compression_ratio: f64::NAN,
        })
    }

    fn decode_with_fallback(
        &mut self,
        segment: &Tensor,
    ) -> Result<DecodingResult> {
        for (i, &t) in m::TEMPERATURES.iter().enumerate() {
            let dr: Result<DecodingResult> = self.decode(segment, t);
            if i == m::TEMPERATURES.len() - 1 {
                return dr;
            }
            // On errors, we try again with a different temperature.
            match dr {
                Ok(dr) => {
                    let needs_fallback = dr.compression_ratio >
                        m::COMPRESSION_RATIO_THRESHOLD ||
                        dr.avg_logprob < m::LOGPROB_THRESHOLD;
                    if !needs_fallback ||
                        dr.no_speech_prob > m::NO_SPEECH_THRESHOLD
                    {
                        return Ok(dr);
                    }
                },
                Err(err) => {
                    log::error!("Error running at {t}: {err}")
                },
            }
        }
        unreachable!()
    }

    fn run(
        &mut self,
        mel: &Tensor,
        mut output_provider: Box<dyn SpeechRecognitionOutputProvider>,
    ) -> Result<()> {
        let (_, _, content_frames) = mel.dims3()?;
        let mut seek = 0;
        output_provider.start()?;
        while seek < content_frames {
            let start = std::time::Instant::now();
            let time_offset =
                (seek * m::HOP_LENGTH) as f64 / m::SAMPLE_RATE as f64;
            let segment_size = usize::min(content_frames - seek, m::N_FRAMES);
            let mel_segment = mel.narrow(2, seek, segment_size)?;
            let segment_duration =
                (segment_size * m::HOP_LENGTH) as f64 / m::SAMPLE_RATE as f64;
            let dr = self.decode_with_fallback(&mel_segment)?;
            seek += segment_size;
            if dr.no_speech_prob > m::NO_SPEECH_THRESHOLD &&
                dr.avg_logprob < m::LOGPROB_THRESHOLD
            {
                log::info!("no speech detected, skipping {seek} {dr:?}");
                continue;
            }
            let segment = Segment {
                start: time_offset,
                duration: segment_duration,
                dr,
            };
            log::info!(
                "{:.1}s -- {:.1}s: {}",
                segment.start,
                segment.start + segment.duration,
                segment.dr.text,
            );
            log::debug!("{seek}: {segment:?}, in {:?}", start.elapsed());
            output_provider.add_segment(segment.clone())?;
        }
        output_provider.finish()?;
        Ok(())
    }
}

pub fn token_id(tokenizer: &Tokenizer, token: &str) -> candle::Result<u32> {
    match tokenizer.token_to_id(token) {
        None => candle::bail!("no token-id for {token}"),
        Some(id) => Ok(id),
    }
}

#[derive(Clone, Copy, Debug)]
enum Task {
    Transcribe,
    Translate,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WhichModel {
    Tiny,
    TinyEn,
    Base,
    BaseEn,
    Small,
    SmallEn,
    Medium,
    MediumEn,
    Large,
    LargeV2,
    LargeV3,
    DistilMediumEn,
    DistilLargeV2,
}

impl WhichModel {
    fn is_multilingual(&self) -> bool {
        match self {
            Self::Tiny |
            Self::Base |
            Self::Small |
            Self::Medium |
            Self::Large |
            Self::LargeV2 |
            Self::LargeV3 |
            Self::DistilLargeV2 => true,
            Self::TinyEn |
            Self::BaseEn |
            Self::SmallEn |
            Self::MediumEn |
            Self::DistilMediumEn => false,
        }
    }

    pub fn model_and_revision(&self) -> (&'static str, &'static str) {
        match self {
            Self::Tiny => ("openai/whisper-tiny", "main"),
            Self::TinyEn => ("openai/whisper-tiny.en", "refs/pr/15"),
            Self::Base => ("openai/whisper-base", "refs/pr/22"),
            Self::BaseEn => ("openai/whisper-base.en", "refs/pr/13"),
            Self::Small => ("openai/whisper-small", "main"),
            Self::SmallEn => ("openai/whisper-small.en", "refs/pr/10"),
            Self::Medium => ("openai/whisper-medium", "main"),
            Self::MediumEn => ("openai/whisper-medium.en", "main"),
            Self::Large => ("openai/whisper-large", "refs/pr/36"),
            Self::LargeV2 => ("openai/whisper-large-v2", "refs/pr/57"),
            Self::LargeV3 => ("openai/whisper-large-v3", "main"),
            Self::DistilMediumEn => ("distil-whisper/distil-medium.en", "main"),
            Self::DistilLargeV2 => ("distil-whisper/distil-large-v2", "main"),
        }
    }
}

#[derive(Debug)]
pub struct SpeechRecognizerTask {
    pub models_data_dir: std::path::PathBuf,
    pub model: WhichModel,
    pub device: Device,
    pub input: std::path::PathBuf,
    pub language: Option<String>,
    pub seed: Option<u64>,
}

pub fn run_speech_recognizer(
    task: SpeechRecognizerTask,
    output_provider: Box<dyn SpeechRecognitionOutputProvider>,
) -> Result<()> {
    let model_dir =
        task.models_data_dir.join(task.model.model_and_revision().0);
    let config: Config = serde_json::from_str(&std::fs::read_to_string(
        model_dir.join(DataFile::Config.file_name()),
    )?)?;
    let tokenizer =
        Tokenizer::from_file(model_dir.join(DataFile::Tokenizer.file_name()))
            .map_err(E::msg)?;

    let mel_bytes = match config.num_mel_bins {
        80 => include_bytes!("melfilters.bytes").as_slice(),
        128 => include_bytes!("melfilters128.bytes").as_slice(),
        nmel => anyhow::bail!("unexpected num_mel_bins {nmel}"),
    };
    let mut mel_filters = vec![0f32; mel_bytes.len() / 4];
    <byteorder::LittleEndian as byteorder::ByteOrder>::read_f32_into(
        mel_bytes,
        &mut mel_filters,
    );

    let (pcm_data, sample_rate) = pcm_decode::pcm_decode(&task.input)?;
    if sample_rate != m::SAMPLE_RATE as u32 {
        anyhow::bail!("input file must have a {} sampling rate", m::SAMPLE_RATE)
    }
    log::info!(
        "pcm data loaded from {}, len {}",
        task.input.display(),
        pcm_data.len()
    );

    let mel = audio::pcm_to_mel(&config, &pcm_data, &mel_filters);
    let mel_len = mel.len();
    let mel = Tensor::from_vec(
        mel,
        (1, config.num_mel_bins, mel_len / config.num_mel_bins),
        &task.device,
    )?;
    log::info!("loaded mel: {:?}", mel.dims());

    let mut model = {
        let vb = VarBuilder::from_buffered_safetensors(
            std::fs::read(model_dir.join(DataFile::Model.file_name()))?,
            m::DTYPE,
            &task.device,
        )?;

        Model::Normal(m::model::Whisper::load(&vb, config)?)
    };

    let language_token = match (task.model.is_multilingual(), task.language) {
        (true, None) => {
            Some(multilingual::detect_language(&mut model, &tokenizer, &mel)?)
        },
        (false, None) => None,
        (true, Some(language)) => {
            match token_id(&tokenizer, &format!("<|{language}|>")) {
                Ok(token_id) => Some(token_id),
                Err(_) => anyhow::bail!("language {language} is not supported"),
            }
        },
        (false, Some(_)) => {
            anyhow::bail!(
                "a language cannot be set for non-multilingual models"
            )
        },
    };

    let mut dc = Decoder::new(
        model,
        tokenizer,
        task.seed.unwrap_or(299792458),
        &task.device,
        language_token,
        None,
        false,
        false,
    )?;
    dc.run(&mel, output_provider)?;

    Ok(())
}
