//! Assembling the prompt the talker is primed with.
//!
//! The talker reads two tracks at once and sums them channel-wise. Laying the
//! prompt out is therefore a matter of deciding, position by position, which
//! text token and which codec token meet there — which is what this module
//! produces, without touching a tensor.
//!
//! The shape of it, for a preset voice:
//!
//! ```text
//! position  0        1          2     3..      …        …          last
//! text      im_start assistant  \n    tts_pad… tts_bos  text… eos  tts_pad
//! codec     –        –          –     think…   speaker  pad…       bos
//! ```
//!
//! The opening role carries no codec token; from there on the codec track
//! states how to speak (whether to reason about the language, which language,
//! which voice) while the text track spells out what to say. Everything the
//! model is to utter sits in the prompt, so generation itself only has to keep
//! feeding padding on the text track.

use super::config::ModelConfig;

/// One position of the prompt: the token each track contributes, if any.
///
/// A track that contributes nothing adds nothing to the summed embedding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Position {
    /// The text-vocabulary token, if the text track speaks here.
    pub text: Option<u32>,
    /// The codec-vocabulary token, if the codec track speaks here.
    pub codec: Option<u32>,
}

impl Position {
    /// A position where only the text track contributes.
    fn text_only(text: u32) -> Self {
        Self {
            text: Some(text),
            codec: None,
        }
    }

    /// A position where both tracks contribute.
    fn both(text: u32, codec: u32) -> Self {
        Self {
            text: Some(text),
            codec: Some(codec),
        }
    }
}

/// What the prompt should say and in whose voice.
#[derive(Debug, Clone, Copy)]
pub struct PromptSpec<'a> {
    /// The tokenized text to utter, without any markup.
    pub text_ids: &'a [u32],
    /// The newline that closes the opening role, as the text vocabulary
    /// spells it.
    pub newline_id: u32,
    /// The preset voice's codec token, when a voice was chosen.
    pub speaker_id: Option<u32>,
    /// The target language's codec token; `None` lets the model decide.
    pub language_id: Option<u32>,
}

/// Lays out the prompt for one synthesis run.
///
/// The result is consumed position by position: each entry's tokens are
/// embedded on their own track and summed.
#[must_use]
pub fn build(cfg: &ModelConfig, spec: &PromptSpec<'_>) -> Vec<Position> {
    let talker = &cfg.talker;

    // The codec track opens by stating whether the language is given. With a
    // language, the model is told to reason about it and handed the id;
    // without one, it is told not to.
    let mut control = match spec.language_id {
        Some(language) => vec![
            talker.codec_think_id,
            talker.codec_think_bos_id,
            language,
            talker.codec_think_eos_id,
        ],
        None => vec![
            talker.codec_nothink_id,
            talker.codec_think_bos_id,
            talker.codec_think_eos_id,
        ],
    };
    if let Some(speaker) = spec.speaker_id {
        control.push(speaker);
    }
    control.push(talker.codec_pad_id);
    control.push(talker.codec_bos_id);

    // The opening role is text only.
    let mut positions = vec![
        Position::text_only(cfg.im_start_token_id),
        Position::text_only(cfg.assistant_token_id),
        Position::text_only(spec.newline_id),
    ];

    // The control block, with the text track padding until it announces that
    // the spoken text is about to start. Its final codec token — the one that
    // starts the speech — is held back for the last position.
    let announced = control.len() - 2;
    for (index, &codec) in control[..control.len() - 1].iter().enumerate() {
        let text = if index == announced {
            cfg.tts_bos_token_id
        } else {
            cfg.tts_pad_token_id
        };
        positions.push(Position::both(text, codec));
    }

    // The text to utter, closed by the end-of-text marker. The codec track
    // pads throughout: no speech has been emitted yet.
    for &id in spec.text_ids {
        positions.push(Position::both(id, talker.codec_pad_id));
    }
    positions.push(Position::both(cfg.tts_eos_token_id, talker.codec_pad_id));

    // The last position starts the speech; generation continues from here.
    positions.push(Position::both(
        cfg.tts_pad_token_id,
        *control
            .last()
            .expect("the control block ends with the start token"),
    ));

    positions
}

#[cfg(test)]
mod tests;
