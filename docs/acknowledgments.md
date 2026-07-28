# Acknowledgments

> Part of [trakktor](../README.md); each command named below is documented
> under [`docs/features/`](features/).

trakktor ports and builds on several open-source projects, all MIT- or
Apache-licensed:

- [**`asr whisper`**](features/asr/whisper.md) — [OpenAI Whisper](https://github.com/openai/whisper)
  (MIT), run on [candle](https://github.com/huggingface/candle) (Apache-2.0 OR
  MIT), with an optional alternative runtime on
  [burn](https://github.com/tracel-ai/burn) (Apache-2.0 OR MIT).
- [**`asr gigaam`**](features/asr/gigaam.md) — [GigaAM](https://github.com/salute-developers/GigaAM)
  (MIT): Conformer acoustic models (CTC and RNN-T) by the GigaChat team,
  pipeline ported to the same candle runtime, with the same optional burn
  runtime.
- [**`asr vosk`**](features/asr/vosk.md) — [Vosk](https://alphacephei.com/vosk/) models by Alpha
  Cephei (Apache-2.0): Zipformer2 RNN-T transducers trained with
  [k2-fsa/icefall](https://github.com/k2-fsa/icefall) (Apache-2.0); the
  inference pipeline is a native port of
  [k2-fsa/sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) (Apache-2.0)
  with Kaldi-compatible fbank features from
  [kaldi-native-fbank](https://github.com/csukuangfj/kaldi-native-fbank)
  (Apache-2.0), on the same candle runtime with the same optional burn
  runtime.
- [**`tts qwen3-tts`**](features/tts/qwen3-tts.md) — [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)
  (Apache-2.0) by the Alibaba Qwen team: the 12 Hz talker, residual code
  predictor, and codec decoder, ported to the same candle runtime, with the
  same optional burn runtime. Model weights and the bundled speech tokenizer
  are downloaded at runtime. Technical
  report: [arXiv:2601.15621](https://arxiv.org/abs/2601.15621).
- [**`tts espeech`**](features/tts/espeech.md) — [F5-TTS](https://github.com/SWivid/F5-TTS) (MIT) by
  Yushen Chen and co-authors: the diffusion-transformer backbone, its
  flow-matching sampler, and the inference pipeline, ported to the same candle
  runtime with the same optional burn runtime; the Russian
  [ESpeech-TTS-1](https://huggingface.co/ESpeech) checkpoints (Apache-2.0); and
  the [Vocos](https://github.com/gemelo-ai/vocos) vocoder (MIT) by Charactr /
  gemelo.ai, weights [`charactr/vocos-mel-24khz`](https://huggingface.co/charactr/vocos-mel-24khz).
  Model weights are downloaded at runtime. Papers:
  [arXiv:2410.06885](https://arxiv.org/abs/2410.06885) (F5-TTS),
  [arXiv:2306.00814](https://arxiv.org/abs/2306.00814) (Vocos).
- [**`vad`**](features/vad/README.md), and
  [**`asr --vad`**](features/asr/whisper.md#voice-activity-detection-vad) —
  [Silero-VAD](https://github.com/snakers4/silero-vad)
  (MIT): the ported speech detector behind both the audio editing commands and
  the transcription preprocessing stage.
- [**`text structify`**](features/text/README.md#text-structify--split-text-into-paragraphs) —
  [SaT / wtpsplit](https://github.com/segment-any-text/wtpsplit)
  (MIT; Frohmann et al., *Segment Any Text*, EMNLP 2024), with the
  [XLM-RoBERTa](https://huggingface.co/FacebookAI/xlm-roberta-base) (MIT)
  tokenizer, on the same candle runtime with the same optional burn runtime.
  Please cite the SaT paper if you use these models.
- [**`text punctuate`**](features/text/README.md#text-punctuate--restore-punctuation-and-casing) — the
  [1-800-BAD-CODE multilingual punctuation/true-casing model](https://huggingface.co/1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase)
  (Apache-2.0), an [XLM-RoBERTa](https://huggingface.co/FacebookAI/xlm-roberta-base)
  (MIT) encoder with cascaded heads, its post-processing following the author's
  [punctuators](https://github.com/1-800-BAD-CODE/punctuators) package (MIT), on
  the same candle runtime with the same optional burn runtime.

See [`NOTICE`](../NOTICE) for the full third-party attributions and license
notices.
