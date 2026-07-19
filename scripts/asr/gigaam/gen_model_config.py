#!/usr/bin/env python3
"""Emit a compact per-model config for the Rust port to embed.

Reads geometry and vocabulary from the reference model and writes a small JSON
(everything the runtime needs to build the encoder, decode, and download the
checkpoint) to trakktor_core/src/asr/gigaam/assets/<model>.json.

Run inside the gigaam venv:
  .venv-gigaam/bin/python scripts/asr/gigaam/gen_model_config.py v3_ctc multilingual_ctc
"""
import json
import sys
from pathlib import Path

import gigaam
from gigaam import _MODEL_HASHES, _URL_DIR

REPO = Path(__file__).resolve().parents[3]
ASSETS = REPO / "trakktor_core/src/asr/gigaam/assets"

# gigaam.encoder maps subsampling_factor -> conv stages via log2; we store the
# raw config values the Rust EncoderConfig expects.


def tokenizer_config(dec):
    """Charwise vocab or SentencePiece pieces, embedded in the config."""
    tok = dec.tokenizer
    if getattr(tok, "charwise", False):
        return {"kind": "charwise", "vocab": list(tok.vocab)}
    # SentencePiece: embed the piece list and the unknown id so the Rust
    # tokenizer needs no separate .model file.
    sp = tok.model
    n = len(sp)
    pieces = [sp.id_to_piece(i) for i in range(n)]
    unk_id = next((i for i in range(n) if sp.IsUnknown(i)), 0)
    return {"kind": "sentencepiece", "pieces": pieces, "unk_id": int(unk_id)}


def emit(model_name: str):
    model = gigaam.load_model(model_name, fp16_encoder=False, device="cpu").eval()
    cfg = model.cfg
    pre = cfg.preprocessor
    enc = cfg.encoder
    dec = model.decoding
    model_class = "ctc" if "ctc" in model_name else ("rnnt" if "rnnt" in model_name else cfg.get("model_class"))
    assert model_class in ("ctc", "rnnt"), f"{model_name}: unsupported class {model_class}"

    out = {
        "model_name": model_name,
        "model_class": model_class,
        "mel": {
            "n_fft": int(pre.n_fft),
            "hop_length": int(pre.hop_length),
            "n_mels": int(pre.features),
            "center": bool(pre.center),
        },
        "encoder": {
            "d_model": int(enc.d_model),
            "n_layers": int(enc.n_layers),
            "n_heads": int(enc.n_heads),
            "subsampling": str(enc.subsampling),
            "subs_kernel_size": int(enc.subs_kernel_size),
            "subsampling_factor": int(enc.subsampling_factor),
            "conv_kernel_size": int(enc.conv_kernel_size),
            "conv_norm": str(enc.conv_norm_type),
            "attention": str(enc.self_attention_model),
        },
        "tokenizer": tokenizer_config(dec),
        "blank_id": int(dec.blank_id),
        "num_classes": int(dec.blank_id) + 1,
        "download": {
            "ckpt": f"{model_name}.ckpt",
            "url": f"{_URL_DIR}/{model_name}.ckpt",
            "md5": _MODEL_HASHES[model_name],
        },
    }
    if model_class == "rnnt":
        # RNN-T head geometry, placed right after the encoder block.
        head = cfg.head
        rnnt = {
            "pred_hidden": int(head.decoder.pred_hidden),
            "pred_rnn_layers": int(head.decoder.pred_rnn_layers),
            "joint_hidden": int(head.joint.joint_hidden),
        }
        items = list(out.items())
        at = [k for k, _ in items].index("encoder") + 1
        out = dict(items[:at] + [("rnnt", rnnt)] + items[at:])
    ASSETS.mkdir(parents=True, exist_ok=True)
    path = ASSETS / f"{model_name}.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    tok = out["tokenizer"]
    size = len(tok["vocab"]) if tok["kind"] == "charwise" else len(tok["pieces"])
    print(f"wrote {path} (d_model={out['encoder']['d_model']} "
          f"n_layers={out['encoder']['n_layers']} tokenizer={tok['kind']} size={size})")


if __name__ == "__main__":
    for name in sys.argv[1:]:
        emit(name)
