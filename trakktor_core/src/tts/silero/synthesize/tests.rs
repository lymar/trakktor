//! Tests for the parts of a run that do not need a model.

use std::collections::HashMap;

use super::{SAMPLE_RATES, Synthesizer, piece_budget, plan};
use crate::tts::silero::{
    SynthesisOptions,
    config::Config,
    error::SileroError,
    istft::{Pqmf, Window},
    model::{Spectrum, SpeechModel, Utterance},
    tables::Tables,
};

/// A model-shaped stand-in: two frames per symbol, a silent spectrum. Enough
/// to drive the run itself — planning, skipping, joining — without tensors.
struct StubModel {
    config: Config,
    window: Window,
}

impl SpeechModel for StubModel {
    fn config(&self) -> &Config { &self.config }

    fn window(&self) -> &Window { &self.window }

    fn pqmf(&self, _bands: usize) -> Option<&Pqmf> { None }

    fn synthesize(
        &self,
        utterance: &Utterance,
    ) -> Result<Spectrum, SileroError> {
        let frames = utterance.ids.len() * 2;
        let bins = self.config.n_fft / 2 + 1;
        Ok(Spectrum {
            magnitude: vec![0.0; frames * bins],
            phase: vec![0.0; frames * bins],
            frames,
            durations: vec![2; utterance.ids.len()],
        })
    }
}

fn stub_config(symbols: usize) -> Config {
    Config {
        symbols,
        speaker_slots: 1,
        dim: 8,
        ff_inner: 16,
        encoder_layers: 1,
        predictor_dim: 8,
        predictor_ff_inner: 8,
        predictor_layers: 1,
        utterance_types: 0,
        mel_channels: 4,
        positions: 5000,
        vocoder_dim: 8,
        vocoder_ff_inner: 16,
        vocoder_layers: 1,
        n_fft: 8,
    }
}

fn stub_tables() -> Tables {
    let alphabet = "|!'+,-.:;? абвгдеёжзийклмнопрстуфхцчшщъыьэюя—…";
    Tables {
        symbols: format!("_~{alphabet}")
            .chars()
            .map(|c| c.to_string())
            .collect(),
        alphabet: alphabet.to_owned(),
        letters: "абвгдеёжзийклмнопрстуфхцчшщъыьэюя".to_owned(),
        sos: "|".to_owned(),
        eos: "~".to_owned(),
        speakers: vec!["ru_one".to_owned()],
        translit: HashMap::new(),
    }
}

fn stub() -> Synthesizer {
    let tables = stub_tables();
    let config = stub_config(tables.symbols.len());
    Synthesizer {
        model: Box::new(StubModel {
            config,
            window: Window {
                samples: vec![1.0; 8],
                hop: 2,
            },
        }),
        tables,
    }
}

fn options() -> SynthesisOptions {
    SynthesisOptions {
        voice: "ru_one".to_owned(),
        sample_rate: 48_000,
        rate: 1.0,
        pitch: 1.0,
        pause: 0.0,
        match_levels: false,
    }
}

/// Speaks `paragraphs` through the stub with `options`.
fn speak(
    synthesizer: &Synthesizer,
    paragraphs: &[&str],
    options: &SynthesisOptions,
) -> Result<super::Synthesis, SileroError> {
    let paragraphs: Vec<String> =
        paragraphs.iter().map(|p| (*p).to_owned()).collect();
    synthesizer.speak_paragraphs(
        &paragraphs,
        options,
        &mut |_, _| Ok(Vec::new()),
        &mut |_| {},
    )
}

#[test]
fn the_models_own_rates_are_accepted_and_nothing_else() {
    for rate in SAMPLE_RATES {
        Synthesizer::check_sample_rate(rate).expect("should accept");
    }
    let error =
        Synthesizer::check_sample_rate(44_100).expect_err("should refuse");
    assert!(matches!(error, SileroError::InvalidOptions(_)), "{error}");
    // The message names what the model can do rather than only what it cannot.
    assert!(error.to_string().contains("48000"), "{error}");
}

#[test]
fn a_paragraph_within_the_budget_is_one_piece() {
    let paragraphs = vec!["Раз.".to_owned(), "Два.".to_owned()];
    let pieces = plan(&paragraphs, 100, &mut |_, _| {
        panic!("should not need splitting")
    })
    .expect("should plan");
    assert_eq!(pieces.len(), 2);
    assert!(pieces.iter().all(|piece| piece.opens_paragraph));
}

#[test]
fn an_over_budget_paragraph_is_handed_to_the_splitter() {
    let long = "слово ".repeat(40);
    let mut asked = 0;
    let pieces = plan(&[long.clone()], 60, &mut |text, budget| {
        asked += 1;
        assert_eq!(budget, 60);
        Ok(text.split_inclusive('о').map(str::to_owned).collect())
    })
    .expect("should plan");
    assert_eq!(asked, 1);
    assert!(pieces.len() > 1);
    // Only the first piece opens the paragraph; the rest continue it, and get
    // the shorter pause.
    assert!(pieces[0].opens_paragraph);
    assert!(pieces[1..].iter().all(|piece| !piece.opens_paragraph));
}

#[test]
fn a_splitter_that_returns_nothing_does_not_swallow_the_text() {
    let long = "слово ".repeat(40);
    let pieces =
        plan(&[long], 60, &mut |_, _| Ok(Vec::new())).expect("should plan");
    assert!(!pieces.is_empty());
    // The fallback cut still respects the budget.
    assert!(pieces.iter().all(|piece| piece.text.len() <= 60));
}

#[test]
fn blank_paragraphs_are_dropped() {
    let paragraphs = vec![String::new(), "  ".to_owned(), "Раз.".to_owned()];
    let pieces = plan(&paragraphs, 100, &mut |_, _| Ok(Vec::new()))
        .expect("should plan");
    assert_eq!(pieces.len(), 1);
}

#[test]
fn an_unreadable_piece_is_skipped_and_counted() {
    let synthesizer = stub();
    let synthesis = speak(
        &synthesizer,
        &["привет мир", "hello world 42", "ещё текст"],
        &options(),
    )
    .expect("should read around the hole");
    assert_eq!(synthesis.chunks, 2);
    assert_eq!(synthesis.skipped, 1);
    // Every non-whitespace character of the skipped piece is accounted for.
    assert_eq!(synthesis.dropped, "helloworld42".chars().count());
}

#[test]
fn a_text_with_no_readable_piece_is_still_an_error() {
    let synthesizer = stub();
    let error =
        speak(&synthesizer, &["hello"], &options()).expect_err("nothing left");
    assert!(matches!(error, SileroError::TextEmpty), "{error}");
}

#[test]
fn a_slow_rate_narrows_the_budget_with_it() {
    assert_eq!(piece_budget(1.0), 700);
    // Faster speech does not widen it: the budget is also the length the
    // model is known to read well.
    assert_eq!(piece_budget(2.0), 700);
    assert_eq!(piece_budget(0.5), 350);

    // 55 words of 12 bytes: one piece at the full budget, two at half rate —
    // half-speed speech doubles the frames, and the ceiling is in frames.
    let synthesizer = stub();
    let long = "сл+ово ".repeat(55);
    let fast =
        speak(&synthesizer, &[&long], &options()).expect("one whole piece");
    let mut slow_options = options();
    slow_options.rate = 0.5;
    let slow =
        speak(&synthesizer, &[&long], &slow_options).expect("split pieces");
    assert_eq!(fast.chunks, 1);
    assert!(slow.chunks >= 2, "{}", slow.chunks);
}

#[test]
fn an_impossible_reading_is_rejected() {
    for (rate, pitch) in [
        (0.0, 1.0),
        (-1.0, 1.0),
        (f32::NAN, 1.0),
        (1.0, -0.5),
        (1.0, f32::NAN),
    ] {
        let error =
            Synthesizer::check_reading(rate, pitch).expect_err("invalid");
        assert!(matches!(error, SileroError::InvalidOptions(_)), "{error}");
    }
    // Zero pitch is the reference's own `robot` preset, not a mistake.
    Synthesizer::check_reading(1.0, 0.0).expect("a robot voice is a setting");
}

#[test]
fn tables_naming_more_speakers_than_the_weights_hold_are_refused() {
    let dir = tempfile::tempdir().expect("a temporary directory");
    let mut tables = stub_tables();
    tables.speakers.push("ru_two".to_owned());
    tables.store(dir.path()).expect("should store");
    let config = stub_config(tables.symbols.len());
    let loaded = Synthesizer::load(
        dir.path(),
        Box::new(StubModel {
            config,
            window: Window {
                samples: vec![1.0; 8],
                hop: 2,
            },
        }),
    );
    // `Synthesizer` holds a boxed model and has no `Debug`, so the error is
    // taken out by hand.
    let Err(error) = loaded else {
        panic!("one row cannot serve two names");
    };
    assert!(matches!(error, SileroError::Checkpoint(_)), "{error}");
}
