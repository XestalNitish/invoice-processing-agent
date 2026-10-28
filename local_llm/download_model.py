"""
local_llm/download_model.py
===========================
Model Manager & Downloader for High-Efficiency Open-Weight GGUF Models.
Optimized for 16GB RAM & Multi-core CPU inference with llama.cpp.

Supported Models:
1. 'spark-4b'      : XHToken/Spark-X2.5-4B-GGUF (Fast Agentic, Default)
2. 'qwen2.5-3b'    : bartowski/Qwen2.5-3B-Instruct-GGUF (Best-in-class JSON/Schema reasoning)
3. 'qwen2.5-1.5b'  : unsloth/Qwen2.5-1.5B-Instruct-GGUF (Ultra-fast, runs >45 tok/s on CPU)
4. 'qwen2.5-coder' : bartowski/Qwen2.5-Coder-3B-Instruct-GGUF (High precision math & table parsing)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

_ROOT = Path(__file__).resolve().parent
MODELS_DIR = _ROOT / "models"

# Registry of models verified to run smoothly on 16GB RAM + 6-Core CPU
MODEL_REGISTRY: Dict[str, Dict[str, str]] = {
    "spark-4b": {
        "repo_id": "XHToken/Spark-X2.5-4B-GGUF",
        "filename": "Spark-X2.5-4B-Q4_K_M.gguf",
        "size_gb": "2.6",
        "description": "Spark-X2.5 4B (Q4_K_M) - High speed, agentic structured extraction",
        "default_dir": "Spark-X2.5-4B-GGUF",
    },
    "qwen2.5-3b": {
        "repo_id": "bartowski/Qwen2.5-3B-Instruct-GGUF",
        "filename": "Qwen2.5-3B-Instruct-Q4_K_M.gguf",
        "size_gb": "2.1",
        "description": "Qwen2.5 3B Instruct (Q4_K_M) - SOTA JSON schema adherence and reasoning",
        "default_dir": "Qwen2.5-3B-Instruct-GGUF",
    },
    "qwen2.5-1.5b": {
        "repo_id": "unsloth/Qwen2.5-1.5B-Instruct-GGUF",
        "filename": "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf",
        "size_gb": "1.1",
        "description": "Qwen2.5 1.5B (Q4_K_M) - Ultra-lightweight & lightning fast",
        "default_dir": "Qwen2.5-1.5B-Instruct-GGUF",
    },
    "qwen2.5-coder": {
        "repo_id": "bartowski/Qwen2.5-Coder-3B-Instruct-GGUF",
        "filename": "Qwen2.5-Coder-3B-Instruct-Q4_K_M.gguf",
        "size_gb": "2.1",
        "description": "Qwen2.5 Coder 3B (Q4_K_M) - Maximum precision for arithmetic & structured JSON",
        "default_dir": "Qwen2.5-Coder-3B-Instruct-GGUF",
    },
}

DEFAULT_MODEL_KEY = "spark-4b"


def get_model_path(model_key: str = DEFAULT_MODEL_KEY, custom_dir: Optional[str] = None) -> Path:
    """Return the absolute path to the designated GGUF file."""
    meta = MODEL_REGISTRY.get(model_key.lower(), MODEL_REGISTRY[DEFAULT_MODEL_KEY])
    dest_dir = Path(custom_dir) if custom_dir else MODELS_DIR / meta["default_dir"]
    return dest_dir / meta["filename"]


def is_model_downloaded(model_key: str = DEFAULT_MODEL_KEY, custom_dir: Optional[str] = None) -> bool:
    """Check if the designated model exists and has valid file size."""
    p = get_model_path(model_key, custom_dir)
    if p.exists() and p.stat().st_size > 100_000_000:
        return True
    
    # Check alternate search paths across disks
    filename = MODEL_REGISTRY.get(model_key.lower(), MODEL_REGISTRY[DEFAULT_MODEL_KEY])["filename"]
    search_dirs = [MODELS_DIR]
    if sys.platform == "win32":
        search_dirs.extend([
            Path("D:/project3/models"),
            Path("D:/agentic ai/models"),
            Path("D:/local llm/models"),
            Path("D:/project2/models"),
        ])
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        candidate = search_dir / filename
        if candidate.exists() and candidate.stat().st_size > 100_000_000:
            return True
        for sub in search_dir.glob(f"**/{filename}"):
            if sub.stat().st_size > 100_000_000:
                return True
    return False


def resolve_model_file(model_key: str = DEFAULT_MODEL_KEY) -> Optional[Path]:
    """Find the existing model file anywhere on disk, or return target path."""
    meta = MODEL_REGISTRY.get(model_key.lower(), MODEL_REGISTRY[DEFAULT_MODEL_KEY])
    filename = meta["filename"]
    
    # 1. Local project models dir
    p = MODELS_DIR / meta["default_dir"] / filename
    if p.exists() and p.stat().st_size > 100_000_000:
        return p

    # 2. Search other local drive locations
    search_dirs = []
    if sys.platform == "win32":
        search_dirs.extend([
            Path("D:/project3/models"),
            Path("D:/agentic ai/models"),
            Path("D:/local llm/models"),
            Path("D:/project2/models"),
        ])
    for search_dir in search_dirs:
        if search_dir.exists():
            for sub in search_dir.glob(f"**/{filename}"):
                if sub.stat().st_size > 100_000_000:
                    return sub

    return p


def download_model(
    model_key: str = DEFAULT_MODEL_KEY,
    save_dir: Optional[str] = None,
    force: bool = False
) -> Path:
    """
    Download the specified open-weight model from Hugging Face with resume capability.
    """
    meta = MODEL_REGISTRY.get(model_key.lower())
    if not meta:
        raise ValueError(f"Unknown model '{model_key}'. Choose from: {list(MODEL_REGISTRY.keys())}")

    existing = resolve_model_file(model_key)
    if existing and existing.exists() and not force:
        size_gb = existing.stat().st_size / (1024 ** 3)
        print(f"✅ Model '{model_key}' already present locally: {existing} ({size_gb:.2f} GB)")
        return existing

    from huggingface_hub import hf_hub_download

    target_dir = Path(save_dir) if save_dir else MODELS_DIR / meta["default_dir"]
    target_dir.mkdir(parents=True, exist_ok=True)
    filename = meta["filename"]

    print(f"📥 Downloading '{filename}' ({meta['size_gb']} GB) from Hugging Face ({meta['repo_id']})...")
    print(f"📁 Destination: {target_dir}")
    print("⏳ This uses multi-threaded download with automatic resume...")

    try:
        downloaded = hf_hub_download(
            repo_id=meta["repo_id"],
            filename=filename,
            local_dir=str(target_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        final_path = Path(downloaded)
        size_gb = final_path.stat().st_size / (1024 ** 3)
        print(f"\n🎉 Download Complete: {final_path} ({size_gb:.2f} GB)")
        return final_path
    except Exception as exc:
        print(f"\n❌ Download failed: {exc}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download & manage open-weight GGUF models")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_KEY,
        choices=list(MODEL_REGISTRY.keys()),
        help="Model choice",
    )
    parser.add_argument("--list", action="store_true", help="List available models")
    parser.add_argument("--check", action="store_true", help="Check if downloaded")
    parser.add_argument("--dir", default=None, help="Custom destination dir")
    parser.add_argument("--force", action="store_true", help="Force re-download")
    args = parser.parse_args()

    if args.list:
        print("\n--- Available Open-Weight Models (Optimized for 16GB RAM + CPU) ---")
        for k, v in MODEL_REGISTRY.items():
            downloaded = "✅ READY" if is_model_downloaded(k) else "⏳ NOT DOWNLOADED"
            print(f"- {k:<14} : {v['description']} (~{v['size_gb']} GB) [{downloaded}]")
        print()
    elif args.check:
        status = "READY" if is_model_downloaded(args.model, args.dir) else "NOT FOUND"
        print(f"Status for '{args.model}': {status} -> {resolve_model_file(args.model)}")
    else:
        download_model(model_key=args.model, save_dir=args.dir, force=args.force)
