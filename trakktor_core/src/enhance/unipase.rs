//! The UniPASE engine: a native port of a four-network generative pipeline for
//! universal speech enhancement.
//!
//! "Universal" is the claim in the paper's title and it is the reason this is
//! one model rather than four: noise, reverberation, a narrow band and a
//! dropped packet are all handled by the same forward pass, with nothing to
//! select and no per-degradation switch. What makes that possible is where the
//! model chooses to work.
//!
//! # The idea
//!
//! Almost every classical enhancer works on the spectrum: estimate a mask,
//! multiply, invert. That can only ever remove — it cannot put back a band the
//! telephone never carried, or the 60 ms a lost packet took with it. UniPASE
//! works one level up, on the **representation a self-supervised speech
//! encoder** builds, and only comes back down to a waveform at the very end:
//!
//! 1. **the encoder** — a WavLM-Large fine-tuned for the job — reads the
//!    recording and is tapped at two depths: its first layer, which still
//!    carries what the room and the microphone did, and its twenty-fourth,
//!    which carries what is being said;
//! 2. **the adapter** takes both and produces what the acoustic layer *should*
//!    have been if the recording had been clean — the deep tap says what the
//!    speech is, the shallow one says what it sounded like, and the answer has
//!    to be true to both;
//! 3. **the vocoder** turns that back into a waveform.
//!
//! Packet loss needs no fourth network. The encoder was pre-trained with spans
//! of its input masked out, so a hole in the recording is handed to it as
//! exactly that — the frames are replaced by the model's own learned mask
//! embedding and the transformer fills them from the context on either side.
//! Detecting the holes is the whole of concealment; see [`plc`].
//!
//! # What this port runs, and what it does not
//!
//! Upstream publishes five checkpoints. Three of them are the pipeline above.
//! The fourth, `Vocoder_WavLM-L24.pt`, exists to evaluate the encoder and takes
//! no part in enhancement. The fifth is a **bandwidth extender** that widens
//! the 16 kHz result to 48 kHz, and this version deliberately does not run it:
//! measured against the reference, it costs about three and a half times what
//! the whole rest of the pipeline costs, for a band above 8 kHz that no
//! recogniser reads and that a recording made over a telephone never had. A
//! `--sample-rate` above 16 kHz therefore resamples rather than invents.
//!
//! # Credits
//!
//! Ported from **UniPASE** by Xiaobin Rong et al. (MIT), which builds on
//! **WavLM** by Microsoft (MIT), on the **Vocos** backbone by way of
//! **WavTokenizer** (MIT), and on **PASE** by Cisco Systems (Apache-2.0).

pub mod config;
#[cfg(feature = "enhance-runtime")]
pub mod download;
#[cfg(feature = "enhance-runtime")]
pub mod enhance;
#[cfg(feature = "enhance-runtime")]
pub mod istft;
pub mod plc;
#[cfg(feature = "enhance-runtime")]
pub mod runtime;
#[cfg(feature = "enhance-burn")]
pub mod runtime_burn;

#[cfg(feature = "enhance-runtime")]
pub use download::{
    DEFAULT_MODEL, KNOWN_MODELS, ResolvedModel, download_size, resolve_model,
};
#[cfg(feature = "enhance-runtime")]
pub use enhance::{enhance_file, enhance_samples};
#[cfg(feature = "enhance-runtime")]
pub use runtime::CandleModel;
#[cfg(feature = "enhance-burn")]
pub use runtime_burn::BurnModel;
