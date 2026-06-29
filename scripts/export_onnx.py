"""
export_onnx.py — One-time offline script. Run BEFORE rank.py.

Loads mxbai-rerank-xsmall-v1 from HuggingFace cache (already downloaded),
exports to FP32 ONNX via torch.onnx, quantizes to INT8 via onnxruntime.quantization.

No optimum dependency whatsoever.

Usage:
    uv run python export_onnx.py

Output:
    artifacts/reranker_optimum/model_int8.onnx   <- loaded by reranker.py at rank time
    artifacts/reranker_optimum/tokenizer files   <- copied from mxbai cache
"""

import shutil
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from onnxruntime.quantization import quantize_dynamic, QuantType


# ── Paths ─────────────────────────────────────────────────────────────────────
ARTIFACTS   = Path("artifacts")
OUTPUT_DIR  = ARTIFACTS / "reranker_optimum"
ONNX_FP32   = OUTPUT_DIR / "model_fp32.onnx"
ONNX_INT8   = OUTPUT_DIR / "model_int8.onnx"
MODEL_ID    = "mixedbread-ai/mxbai-rerank-xsmall-v1"


def export_to_onnx():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Resolve local cache path for mxbai ────────────────────────────────
    # snapshot_download returns the local path without re-downloading
    # if the model is already in ~/.cache/huggingface/hub
    print(f"[export] Locating {MODEL_ID} in HuggingFace cache...")
    local_dir = snapshot_download(repo_id=MODEL_ID, local_files_only=True)
    print(f"[export] Found at: {local_dir}")

    # ── 2. Load model + tokenizer ─────────────────────────────────────────────
    print("[export] Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(local_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        local_dir, torch_dtype=torch.float32
    )
    model.eval()

    # ── 3. Verify model config ────────────────────────────────────────────────
    cfg = model.config
    print(f"[export] Model type      : {cfg.model_type}")
    print(f"[export] Hidden size     : {cfg.hidden_size}")
    print(f"[export] Num layers      : {cfg.num_hidden_layers}")
    print(f"[export] Num labels      : {cfg.num_labels}")

    # ── 4. Build dummy input for tracing ─────────────────────────────────────
    # Use realistic lengths — short enough to trace fast, long enough to
    # ensure dynamic axes are exercised correctly.
    dummy_query = "Senior AI Engineer 6 years RAG pipelines vector search production."
    dummy_doc   = "Candidate has 5 years Python ML engineering shipped vector search."
    enc = tokenizer(
        dummy_query,
        dummy_doc,
        return_tensors="pt",
        max_length=512,
        truncation=True,
        padding="max_length",
    )

    # mxbai uses token_type_ids (standard BERT); confirm it exists
    has_tti = "token_type_ids" in enc
    print(f"[export] Has token_type_ids: {has_tti}")

    if has_tti:
        dummy_inputs = (
            enc["input_ids"],
            enc["attention_mask"],
            enc["token_type_ids"],
        )
        input_names = ["input_ids", "attention_mask", "token_type_ids"]
        dynamic_axes = {
            "input_ids":      {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "token_type_ids": {0: "batch", 1: "seq"},
            "logits":         {0: "batch"},
        }
    else:
        # Fallback: some models don't use token_type_ids
        dummy_inputs = (enc["input_ids"], enc["attention_mask"])
        input_names  = ["input_ids", "attention_mask"]
        dynamic_axes = {
            "input_ids":      {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "logits":         {0: "batch"},
        }

    # ── 5. Export FP32 ONNX ───────────────────────────────────────────────────
    print(f"[export] Exporting FP32 ONNX → {ONNX_FP32}")
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_inputs,
            str(ONNX_FP32),
            opset_version=14,
            input_names=input_names,
            output_names=["logits"],
            dynamic_axes=dynamic_axes,
        )
    print("[export] FP32 export done ✓")

    # ── 6. INT8 dynamic quantization ──────────────────────────────────────────
    # Dynamic quantization: weights are pre-quantized to INT8 at export time.
    # Activations are quantized on the fly at inference.
    # No calibration dataset needed — safe and correct for cross-encoders.
    # per_channel=False: single scale per weight tensor. 
    # More stable than per-channel for BERT-class models on onnxruntime 1.17.x
    print(f"[export] Quantizing INT8 → {ONNX_INT8}")
    quantize_dynamic(
        model_input=str(ONNX_FP32),
        model_output=str(ONNX_INT8),
        weight_type=QuantType.QInt8,
        per_channel=False,
        reduce_range=False,
    )
    print("[export] INT8 quantization done ✓")

    # ── 7. Copy tokenizer files ───────────────────────────────────────────────
    # Copy from mxbai cache (NOT from reranker_model/ which has MiniLM tokenizer)
    print("[export] Copying tokenizer files from mxbai cache...")
    copied = []
    for f in Path(local_dir).iterdir():
        if f.suffix in {".json", ".txt"} and "safetensors" not in f.name:
            shutil.copy2(f, OUTPUT_DIR / f.name)
            copied.append(f.name)
    print(f"[export] Copied: {copied}")

    # ── 8. Cleanup FP32 intermediate ─────────────────────────────────────────
    ONNX_FP32.unlink()
    print("[export] Removed intermediate FP32 file ✓")

    # ── 9. Final report ───────────────────────────────────────────────────────
    size_mb = ONNX_INT8.stat().st_size / 1e6
    print(f"\n[export] ✅ Done.")
    print(f"[export] INT8 model : {ONNX_INT8}  ({size_mb:.1f} MB)")
    print(f"[export] Tokenizer  : {OUTPUT_DIR}")
    print(f"\n[export] Now run: uv run python rank.py")


if __name__ == "__main__":
    export_to_onnx()