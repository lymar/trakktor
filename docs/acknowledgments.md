# Acknowledgments

> Part of [trakktor](../README.md); each command named below is documented
> under [`docs/features/`](features/).

trakktor ports and builds on several open-source projects, all MIT- or
Apache-licensed:

- [**`asr whisper`**](features/asr/whisper.md) — [OpenAI Whisper](https://github.com/openai/whisper)
  (MIT), run on [candle](https://github.com/huggingface/candle) (Apache-2.0 OR
  MIT), with an optional alternative runtime on
  [burn](https://github.com/tracel-ai/burn) (Apache-2.0 OR MIT). The
  `podlodka` and `podlodka-turbo` models are the
  [Whisper-Podlodka](https://huggingface.co/bond005/whisper-large-v3-ru-podlodka)
  Russian fine-tunes (Apache-2.0) by
  [Ivan Bondarenko](https://huggingface.co/bond005), downloaded at runtime;
  please cite them per their model cards if you use them.
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
- [**`tts silero`**](features/tts/silero.md) — [Silero TTS](https://github.com/snakers4/silero-models)
  by the Silero Team: the v5 acoustic model — sixty voices across twenty languages of the
  region — (a FastPitch-family encoder with
  duration and pitch heads and an hourglass decoder), its character frontend,
  and the analysis filterbanks its lower sample rates come from, ported to the
  same candle runtime with the same optional burn runtime; the vocoder is
  [Vocos](https://github.com/gemelo-ai/vocos) (MIT) again. Models are
  downloaded at runtime. **Their licence is per model:** `v5_cis_base` and
  `v5_cis_base_nostress` are MIT, and everything else in that repository is
  CC BY-NC-SA 4.0 and is reachable only behind
  `--allow-non-commercial-models`.
- [**`ocr paddle`**](features/ocr/README.md) —
  [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) (Apache-2.0) by the
  PaddlePaddle authors: the classic pipeline in both live generations — three
  differentiable-binarization text detectors, three CTC text recognizers and a
  text-line orientation classifier — ported to the same candle runtime, reading
  PaddleOCR's own published inference artifacts directly. The detection algorithm comes from
  [DB](https://github.com/MhLiao/DB) (Apache-2.0; Liao et al.,
  *Real-time Scene Text Detection with Differentiable Binarization*, AAAI 2020)
  by way of [DBNet.pytorch](https://github.com/WenmuZhou/DBNet.pytorch), which
  PaddleOCR names as its source; the large detector's intra-class block is
  adapted by PaddleOCR from
  [I3CL](https://github.com/ViTAE-Transformer/I3CL), and the newest detector's
  large-kernel reparameterized block from
  [UniRepLKNet](https://github.com/AILab-CVC/UniRepLKNet) — both named by its
  own source. Please cite the PaddleOCR 3.0 technical report
  (arXiv:2507.05595) and the DB paper if you use these models.
- [**`ocr vl`**](features/ocr/README.md) —
  [PaddleOCR-VL 1.6](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6)
  (Apache-2.0) by the PaddlePaddle authors: a 0.9-billion-parameter document
  model — a variable-resolution vision tower, a patch-merging projector and an
  [ERNIE-4.5](https://huggingface.co/baidu/ERNIE-4.5-0.3B-Paddle) (Apache-2.0)
  decoder — ported to the same candle and burn runtimes. Where the
  checkpoint's own code
  and the native implementation in
  [Transformers](https://github.com/huggingface/transformers) (Apache-2.0)
  disagree on interpolating the vision tower's position grid, the port follows
  Transformers. [oar-ocr](https://github.com/GreatV/oar-ocr) (Apache-2.0) was
  consulted as prior art for the same checkpoint on candle. Weights are
  downloaded at runtime. Please cite the PaddleOCR-VL technical report
  (arXiv:2510.14528) if you use this model.
- [**`ocr layout`**](features/ocr/README.md), and the layout stage both OCR
  engines run by default —
  [PP-DocLayout_plus-L](https://huggingface.co/PaddlePaddle/PP-DocLayout_plus-L)
  (Apache-2.0) by the PaddlePaddle authors: a twenty-class document-layout
  detector built on [RT-DETR](https://github.com/lyuwenyu/RT-DETR)
  (Apache-2.0; Zhao et al., *DETRs Beat YOLOs on Real-time Object Detection*,
  CVPR 2024), ported to the same candle and burn runtimes. The box
  post-processing around it, and the per-class settings the layout stage runs
  with, follow the PP-StructureV3 pipeline in
  [PaddleX](https://github.com/PaddlePaddle/PaddleX) (Apache-2.0). Weights are
  downloaded at runtime. Please cite the PaddleOCR 3.0 technical report
  (arXiv:2507.05595) and the RT-DETR paper if you use this model.
- **`--doc-orientation`**, the page-orientation step both OCR engines can run —
  [PP-LCNet_x1_0_doc_ori](https://huggingface.co/PaddlePaddle/PP-LCNet_x1_0_doc_ori)
  (Apache-2.0) by the PaddlePaddle authors: a four-class classifier saying
  which right angle a photographed page is at, ported to the same candle
  runtime and sharing its backbone with the text-line orientation classifier
  above. Weights are downloaded at runtime. Please cite the PaddleOCR 3.0
  technical report (arXiv:2507.05595) if you use this model.
- **`--unwarp`**, the page-straightening step both OCR engines can run —
  [UVDoc](https://github.com/tanguymagne/UVDoc) (MIT; Verhoeven, Magne and
  Sorkine-Hornung, *UVDoc: Neural Grid-based Document Unwarping*, SIGGRAPH
  Asia 2023), in the form
  [PaddleX](https://huggingface.co/PaddlePaddle/UVDoc) publishes it: a network
  predicting a dense backward map that turns a photograph of a page into the
  page a scanner would have seen. Ported to the same candle runtime; the map
  is applied, and inverted for reporting boxes, by trakktor itself. Weights
  are downloaded at runtime. Please cite the UVDoc paper if you use this
  model.
- [**`enhance gtcrn`**](features/enhance/README.md) —
  [GTCRN](https://github.com/Xiaobin-Rong/gtcrn) (MIT) by Xiaobin Rong, Jing Lu
  and co-authors: the ultra-lightweight masking enhancer — 48 K parameters —
  ported to the same candle runtime. Its checkpoint (MIT, carried in the
  project's own repository) is downloaded at runtime. Paper:
  [ICASSP 2024](https://ieeexplore.ieee.org/document/10448310).

  If you use this engine, please cite:

  > X. Rong, T. Sun, X. Zhang, Y. Hu, C. Zhu and J. Lu, "GTCRN: A Speech
  > Enhancement Model Requiring Ultralow Computational Resources," *ICASSP
  > 2024*, pp. 971-975.

- [**`enhance mpsenet`**](features/enhance/README.md) —
  [MP-SENet](https://github.com/yxlu-0102/MP-SENet) (MIT) by Ye-Xin Lu, Yang Ai
  and Zhen-Hua Ling: the magnitude-and-phase enhancer — the only engine here
  that estimates phase as its own output — ported to the same candle runtime
  with the same optional burn runtime. Both published checkpoints (MIT, carried
  in the project's own repository) are downloaded at runtime. Papers:
  [arXiv:2305.13686](https://arxiv.org/abs/2305.13686) (Interspeech 2023),
  [arXiv:2308.08926](https://arxiv.org/abs/2308.08926) (the extended version,
  *Neural Networks* 2025).

  If you use this engine, please cite:

  > Y.-X. Lu, Y. Ai and Z.-H. Ling, "MP-SENet: A Speech Enhancement Model with
  > Parallel Denoising of Magnitude and Phase Spectra," *Proc. Interspeech
  > 2023*, pp. 3834-3838.

- [**`enhance unipase`**](features/enhance/README.md) —
  [UniPASE](https://github.com/Xiaobin-Rong/unipase) (MIT) by Xiaobin Rong,
  Zheng Wang, Yushi Wang, Jun Gao and Jing Lu: the generative
  speech-enhancement pipeline — the two-tap read of its fine-tuned
  [WavLM](https://github.com/microsoft/unilm/tree/master/wavlm) encoder (MIT,
  Microsoft), the Vocos adapter and vocoder around it in the variant published
  by [WavTokenizer](https://github.com/jishengpeng/WavTokenizer) (MIT), and the
  packet-loss concealment that rides on the encoder's masking — ported to the
  same candle runtime with the same optional burn runtime. The pipeline
  assembly and its long-form inference are adapted from
  [PASE](https://github.com/cisco-open/pase) (Apache-2.0, Cisco Systems).
  Weights ([`Xiaobin-Rong/unipase`](https://huggingface.co/Xiaobin-Rong/unipase),
  Apache-2.0) are downloaded at runtime. Papers:
  [arXiv:2604.14606](https://arxiv.org/abs/2604.14606) (UniPASE, *IEEE TASLP*
  2026), [arXiv:2110.13900](https://arxiv.org/abs/2110.13900) (WavLM).

  If you use this engine, please cite:

  > X. Rong, Z. Wang, Y. Wang, J. Gao and J. Lu, "UniPASE: A Generative Model
  > for Universal Speech Enhancement With High Fidelity and Low
  > Hallucinations," *IEEE Transactions on Audio, Speech and Language
  > Processing*, vol. 34, pp. 3901-3915, 2026.

- [**`enhance resemble-denoise`** and
  **`enhance resemble-enhance`**](features/enhance/README.md) —
  [resemble-enhance](https://github.com/resemble-ai/resemble-enhance) (MIT) by
  Resemble AI: both networks the project publishes, and the only two engines
  here that work at 44.1 kHz. The denoiser is a UNet over the spectrum that
  predicts a gain and a rotation of the phase; the enhancer is a generative
  pipeline — a latent autoencoder over the mel, a conditional-flow-matching
  velocity field, and a UnivNet vocoder that builds the waveform out of noise.
  The vocoder's anti-aliased activation comes from
  [BigVGAN](https://github.com/NVIDIA/BigVGAN) (MIT, NVIDIA) with the half-band
  filters of [alias-free-torch](https://github.com/junjun3518/alias-free-torch)
  (Apache-2.0, Junhyeok Lee), and its location-variable convolutions follow
  [LVCNet](https://github.com/zceng/LVCNet). Both are ported to the same candle
  runtime with the same optional burn runtime. The published checkpoint
  ([`ResembleAI/resemble-enhance`](https://huggingface.co/ResembleAI/resemble-enhance),
  MIT), which carries both networks, is downloaded at runtime.

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
- [**`text stress`**](features/text/README.md#text-stress--mark-the-stress-in-russian-text) —
  [Silero Stress](https://github.com/snakers4/silero-stress) (MIT) by the Silero
  Team: the n-gram accentor with its stress and `ё` heads and the
  `rubert-tiny`-class homograph solver, ported to the same candle runtime with
  the same optional burn runtime. The model is downloaded at runtime. Also used
  by [`tts espeech`](features/tts/espeech.md) and
  [`tts silero`](features/tts/silero.md), which mark their text with it before
  speaking.
- [**`text punctuate`**](features/text/README.md#text-punctuate--restore-punctuation-and-casing) — the
  [1-800-BAD-CODE multilingual punctuation/true-casing model](https://huggingface.co/1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase)
  (Apache-2.0), an [XLM-RoBERTa](https://huggingface.co/FacebookAI/xlm-roberta-base)
  (MIT) encoder with cascaded heads, its post-processing following the author's
  [punctuators](https://github.com/1-800-BAD-CODE/punctuators) package (MIT), on
  the same candle runtime with the same optional burn runtime.

- [**`convert pdf`**](features/convert/README.md) —
  [pdf-inspector](https://github.com/firecrawl/pdf-inspector) (MIT) by
  Firecrawl: PDF classification and Markdown extraction, pure Rust with no C
  dependency. Unlike everything above it, this one is **not a port** — the crate
  is a dependency and its code runs as published. What trakktor adds around it
  is the repair of an encoding case the crate does not handle and two checks on
  the quality of what comes out. It bundles the
  [Adobe Glyph List](https://github.com/adobe-type-tools/agl-aglfn) and the
  Adobe CMap resources (BSD-3-Clause), and parses PDFs with
  [lopdf](https://github.com/J-F-Liu/lopdf) (MIT), which trakktor also uses
  directly for the repair.

- [**`pdf cut`**](features/pdf/README.md) —
  [lopdf](https://github.com/J-F-Liu/lopdf) (MIT) by Junfeng Liu and
  contributors: parsing, editing and writing PDF documents in pure Rust. A
  dependency, not a port: page extraction runs over its document model —
  loading (with decryption), the page tree, reachability-based garbage
  collection, and serialization are all the crate's. What trakktor adds on top
  is the catalog hygiene around a cut and the page-tree surgery that keeps it
  fast on large documents.

See [`NOTICE`](../NOTICE) for the full third-party attributions and license
notices.
