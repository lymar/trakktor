//! The published model catalog.
//!
//! Every model of the line is the same four-file bundle — the fp32 ONNX export
//! triple plus the token table — hosted on HuggingFace. The catalog records,
//! per model: the repository, the remote path of each file, its size, and its
//! BLAKE3 hash for download verification (the hashes were pinned from files
//! whose SHA-256 matched the hosting's LFS metadata).
//!
//! `--model` also accepts a local directory holding the same four files (any
//! compatible icefall Zipformer2 transducer export); the catalog then plays no
//! part.

/// Whether a model's encoder is full-context or causal (chunked with cached
/// state).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelKind {
    /// Full-context encoder; long audio is segmented along detected speech.
    Offline,
    /// Causal encoder consumed chunk by chunk with carried state.
    Streaming,
}

/// One downloadable file of a model bundle.
#[derive(Debug, Clone, Copy)]
pub struct RemoteFile {
    /// File name in the local model directory.
    pub local: &'static str,
    /// Path inside the HuggingFace repository.
    pub remote: &'static str,
    /// Expected size in bytes.
    pub size: u64,
    /// Expected BLAKE3 hash (lowercase hex).
    pub blake3: &'static str,
}

/// A published model.
#[derive(Debug, Clone, Copy)]
pub struct ModelSpec {
    /// The `--model` name.
    pub name: &'static str,
    /// HuggingFace repository id.
    pub repo: &'static str,
    /// Model language (ISO 639-1 where one exists).
    pub language: &'static str,
    /// Upstream version at the time the catalog was pinned.
    pub version: &'static str,
    pub kind: ModelKind,
    pub files: [RemoteFile; 4],
}

/// Local file names of a model bundle, in the catalog's file order.
pub const ENCODER_FILE: &str = "encoder.onnx";
pub const DECODER_FILE: &str = "decoder.onnx";
pub const JOINER_FILE: &str = "joiner.onnx";
pub const TOKENS_FILE: &str = "tokens.txt";

const fn file(
    local: &'static str,
    remote: &'static str,
    size: u64,
    blake3: &'static str,
) -> RemoteFile {
    RemoteFile {
        local,
        remote,
        size,
        blake3,
    }
}

/// The published models this build knows.
pub const KNOWN_MODELS: &[ModelSpec] = &[
    ModelSpec {
        name: "ru",
        repo: "alphacep/vosk-model-ru",
        language: "ru",
        version: "0.54",
        kind: ModelKind::Offline,
        files: [
            file(
                ENCODER_FILE,
                "am-onnx/encoder.onnx",
                261_058_126,
                "5af23eb95106438849343c1a280fb34324b9d7456c658fd66d05425e0fb8ea0a",
            ),
            file(
                DECODER_FILE,
                "am-onnx/decoder.onnx",
                2_093_080,
                "c9a9a1d167be5844ba920a1d810dd1389a6e4c0f72e9d9c02e9c304973f473d0",
            ),
            file(
                JOINER_FILE,
                "am-onnx/joiner.onnx",
                1_026_462,
                "34d2f88e408f946859f25311fa0e508e3ac7226c847ec20fd00d49affa8c4691",
            ),
            file(
                TOKENS_FILE,
                "lang/tokens.txt",
                6_388,
                "dd19167cce1771140893ff03cf157164aa95116a5e66a355a1bc860db73c17e6",
            ),
        ],
    },
    ModelSpec {
        name: "small-ru",
        repo: "alphacep/vosk-model-small-ru",
        language: "ru",
        version: "0.52",
        kind: ModelKind::Offline,
        files: [
            file(
                ENCODER_FILE,
                "am/encoder.onnx",
                90_250_660,
                "d508c2d2197337fea722c6d9b46c60b5c46bcf752a6b28a5389918ca9cd4814a",
            ),
            file(
                DECODER_FILE,
                "am/decoder.onnx",
                2_093_080,
                "8db2a5a502ebc053364f4bbd21ce39d0e15d6a4dc7374101de5ccd1a99cae6b4",
            ),
            file(
                JOINER_FILE,
                "am/joiner.onnx",
                1_026_462,
                "2d159c9bea5e528396f7a574094ae1ad89543077426602b2546aac18bb37fd01",
            ),
            file(
                TOKENS_FILE,
                "lang/tokens.txt",
                6_388,
                "dd19167cce1771140893ff03cf157164aa95116a5e66a355a1bc860db73c17e6",
            ),
        ],
    },
    ModelSpec {
        name: "streaming-ru",
        repo: "alphacep/vosk-model-streaming-ru",
        language: "ru",
        version: "0.56",
        kind: ModelKind::Streaming,
        files: [
            file(
                ENCODER_FILE,
                "am-onnx/encoder.onnx",
                260_638_202,
                "607e80de89140cc46ae878422af62848c689c0c899e06212d09ddeda3ccc7494",
            ),
            file(
                DECODER_FILE,
                "am-onnx/decoder.onnx",
                2_093_080,
                "584fffacc0751153131eb405bb394233dc60c8d2acd1f14be359b490f589109f",
            ),
            file(
                JOINER_FILE,
                "am-onnx/joiner.onnx",
                1_026_462,
                "b20524bb74c1abb37fc4c0d0d3b7f542bf425a9d46f074114f408b71bbf51f56",
            ),
            file(
                TOKENS_FILE,
                "lang/tokens.txt",
                6_388,
                "dd19167cce1771140893ff03cf157164aa95116a5e66a355a1bc860db73c17e6",
            ),
        ],
    },
    ModelSpec {
        name: "small-streaming-ru",
        repo: "alphacep/vosk-model-small-streaming-ru",
        language: "ru",
        version: "0.54",
        kind: ModelKind::Streaming,
        files: [
            file(
                ENCODER_FILE,
                "am-onnx/encoder.onnx",
                90_994_145,
                "d7ab0972399608b536500958922ea9351938655483f545bdfed48976ca677e4e",
            ),
            file(
                DECODER_FILE,
                "am-onnx/decoder.onnx",
                2_093_080,
                "0c864b9041cdc0b8386f855430185df442f4dba8e04e14131982c12f74b603b6",
            ),
            file(
                JOINER_FILE,
                "am-onnx/joiner.onnx",
                1_026_462,
                "13c5d26029e1390c2032bd35befabbb5bbcd97e1ee9a25b7da3ba312df8537df",
            ),
            file(
                TOKENS_FILE,
                "lang/tokens.txt",
                6_388,
                "dd19167cce1771140893ff03cf157164aa95116a5e66a355a1bc860db73c17e6",
            ),
        ],
    },
    ModelSpec {
        name: "small-streaming-bn",
        repo: "alphacep/vosk-model-small-streaming-bn",
        language: "bn",
        version: "0.60",
        kind: ModelKind::Streaming,
        files: [
            file(
                ENCODER_FILE,
                "am-onnx/encoder.onnx",
                90_994_145,
                "e05bccc544e8a7f1f8ac0a0eff5d6b4118dde292060990a6da609d4ad6d40233",
            ),
            file(
                DECODER_FILE,
                "am-onnx/decoder.onnx",
                2_093_080,
                "856466d4bfdcd6cb3df09c79cc97176ec76b51675d24c8dd6195bfd551d1694d",
            ),
            file(
                JOINER_FILE,
                "am-onnx/joiner.onnx",
                1_026_462,
                "f70a155fd6b171f568adf7a10bd7652889b4fb6c98ef714cf503badd4c175f9e",
            ),
            file(TOKENS_FILE, "lang/tokens.txt", 6_252, "5a6a14bfafecaa3578bc18a3fd329d334e3f524e0dae6661a15931a034e0e646"),
        ],
    },
    ModelSpec {
        name: "tg",
        repo: "alphacep/vosk-model-tg",
        language: "tg",
        version: "0.60",
        kind: ModelKind::Offline,
        files: [
            file(
                ENCODER_FILE,
                "am-onnx/encoder.onnx",
                261_058_046,
                "365047b37b8766c65e04d67a9ee773e8618214462109d2a2faea3c6dce52fef5",
            ),
            file(
                DECODER_FILE,
                "am-onnx/decoder.onnx",
                2_093_080,
                "ff5588bd6a96a4b2328990fb06aa4f43888a3130c89187ba260e28eeb8e65df9",
            ),
            file(
                JOINER_FILE,
                "am-onnx/joiner.onnx",
                1_026_462,
                "704d1ef21842246d09b8338e6d796f43f5cfceb81fd55313b3e56f2dcab21ffd",
            ),
            file(TOKENS_FILE, "lang/tokens.txt", 5_062, "bf8e93d8ad6cf5c39a94124e47f6ec9c6671144020b3df470baacf7edb83cf37"),
        ],
    },
];

/// The spec of a published model name.
pub fn spec_for(name: &str) -> Option<&'static ModelSpec> {
    KNOWN_MODELS.iter().find(|m| m.name == name)
}

/// The published names, comma-joined for error messages and help.
pub fn known_names() -> String {
    KNOWN_MODELS
        .iter()
        .map(|m| m.name)
        .collect::<Vec<_>>()
        .join(", ")
}
