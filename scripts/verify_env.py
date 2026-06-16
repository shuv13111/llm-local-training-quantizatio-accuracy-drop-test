#!/usr/bin/env python3
"""
Step 1: Environment Verification for DGX Spark
================================================
Run this FIRST after installing requirements.txt.
It checks every dependency the project needs and reports what's working,
what's broken, and what needs manual intervention.

Usage:
    python scripts/verify_env.py
"""

import sys
import subprocess
import shutil
from pathlib import Path


def header(msg):
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}")


def check(name, passed, detail=""):
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"  {status}  {name}")
    if detail:
        print(f"         {detail}")
    return passed


def main():
    results = {"pass": 0, "fail": 0, "warn": 0}

    # ── System Info ──────────────────────────────────────────
    header("System Information")
    import platform
    arch = platform.machine()
    print(f"  Platform:      {platform.platform()}")
    print(f"  Architecture:  {arch}")
    print(f"  Python:        {sys.version}")

    if arch != "aarch64":
        print(f"\n  ⚠️  WARNING: Expected aarch64 (DGX Spark Grace CPU), got {arch}")
        print(f"     Some package compatibility notes may not apply.")
        results["warn"] += 1

    # ── CUDA / GPU ───────────────────────────────────────────
    header("CUDA & GPU")

    # nvidia-smi
    nvidia_smi = shutil.which("nvidia-smi")
    if check("nvidia-smi found", nvidia_smi is not None):
        results["pass"] += 1
        try:
            out = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,driver_version,compute_cap",
                                           "--format=csv,noheader"], text=True, timeout=10)
            for line in out.strip().split("\n"):
                print(f"         GPU: {line.strip()}")
        except Exception as e:
            print(f"         Could not query GPU details: {e}")
    else:
        results["fail"] += 1

    # PyTorch + CUDA
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if check("PyTorch CUDA", cuda_available,
                  f"torch {torch.__version__}, CUDA {torch.version.cuda if cuda_available else 'N/A'}"):
            results["pass"] += 1
            # Check unified memory (DGX Spark specific)
            if cuda_available:
                total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                print(f"         GPU Memory: {total_mem:.1f} GB")
                if total_mem >= 120:
                    print(f"         ✅ Looks like 128GB unified memory — full BF16 fine-tuning of 8B is feasible")
                elif total_mem >= 60:
                    print(f"         ⚠️  Less than expected for DGX Spark — may need QLoRA for 8B")
        else:
            results["fail"] += 1
    except ImportError:
        check("PyTorch", False, "not installed")
        results["fail"] += 1

    # ── Core Training Stack ──────────────────────────────────
    header("Core Training Stack")

    core_packages = [
        ("transformers", "4.46.0"),
        ("peft", "0.13.0"),
        ("trl", "0.12.0"),
        ("datasets", "3.0.0"),
        ("accelerate", "1.0.0"),
    ]

    for pkg_name, min_ver in core_packages:
        try:
            mod = __import__(pkg_name)
            ver = getattr(mod, "__version__", "unknown")
            if check(f"{pkg_name}", True, f"v{ver}"):
                results["pass"] += 1
        except ImportError:
            check(f"{pkg_name}", False, "not installed")
            results["fail"] += 1

    # ── bitsandbytes (the tricky one on aarch64) ─────────────
    header("bitsandbytes (aarch64 compatibility)")

    try:
        import bitsandbytes as bnb
        ver = getattr(bnb, "__version__", "unknown")
        check("bitsandbytes import", True, f"v{ver}")
        results["pass"] += 1

        # Actually test CUDA functionality
        try:
            import torch
            if torch.cuda.is_available():
                # Try creating an 8-bit optimizer — this is what LoRA training actually uses
                param = torch.nn.Parameter(torch.randn(64, 64, device="cuda"))
                opt = bnb.optim.AdamW8bit([param], lr=1e-4)
                opt.zero_grad()
                loss = param.sum()
                loss.backward()
                opt.step()
                check("bitsandbytes 8-bit optimizer (CUDA)", True, "AdamW8bit works on GPU")
                results["pass"] += 1
        except Exception as e:
            check("bitsandbytes 8-bit optimizer (CUDA)", False, str(e))
            print("         ⚠️  bitsandbytes installed but CUDA kernels don't work on this arch.")
            print("         ➡️  Fallback: use standard AdamW optimizer (more VRAM but will work)")
            results["fail"] += 1

    except ImportError:
        check("bitsandbytes", False, "not installed")
        print("         ➡️  Try: pip install bitsandbytes --break-system-packages")
        print("         ➡️  If it fails on aarch64, use standard AdamW instead")
        results["fail"] += 1

    # ── Quantization Tools ───────────────────────────────────
    header("Quantization Tools")

    # GPTQModel
    try:
        import gptqmodel
        ver = getattr(gptqmodel, "__version__", "unknown")
        check("gptqmodel", True, f"v{ver}")
        results["pass"] += 1
    except ImportError:
        check("gptqmodel", False, "not installed (optional — install later)")
        results["warn"] += 1

    # AutoAWQ
    try:
        import awq
        ver = getattr(awq, "__version__", "unknown")
        check("autoawq", True, f"v{ver}")
        results["pass"] += 1
    except ImportError:
        check("autoawq", False, "not installed (optional — install later)")
        results["warn"] += 1

    # llama-cpp-python
    try:
        from llama_cpp import Llama
        check("llama-cpp-python", True)
        results["pass"] += 1
    except ImportError:
        check("llama-cpp-python", False, "not installed (optional — install later)")
        print("         ➡️  Build from source: CMAKE_ARGS=\"-DGGML_CUDA=on\" pip install llama-cpp-python")
        results["warn"] += 1

    # Docker (for TRT-LLM container)
    docker = shutil.which("docker")
    if check("docker", docker is not None, "needed for TRT-LLM / NVFP4 quantization"):
        results["pass"] += 1
    else:
        results["fail"] += 1

    # ── Data & Logging ───────────────────────────────────────
    header("Data & Logging")

    for pkg_name in ["pandas", "numpy", "matplotlib", "seaborn", "wandb", "jsonlines"]:
        try:
            __import__(pkg_name)
            check(pkg_name, True)
            results["pass"] += 1
        except ImportError:
            check(pkg_name, False, "not installed")
            results["fail"] += 1

    # ── HuggingFace Auth ─────────────────────────────────────
    header("HuggingFace Authentication")

    try:
        from huggingface_hub import HfApi
        api = HfApi()
        user = api.whoami()
        check("HuggingFace token", True, f"logged in as: {user.get('name', 'unknown')}")
        results["pass"] += 1
    except Exception:
        check("HuggingFace token", False, "not logged in")
        print("         ➡️  Run: huggingface-cli login")
        print("         ➡️  Needed for: downloading gated models, pushing results")
        results["fail"] += 1

    # ── Project Structure ────────────────────────────────────
    header("Project Structure")

    expected_dirs = [
        "scripts", "configs", "data/raw", "data/processed",
        "models/base", "models/finetuned", "models/quantized",
        "eval/results", "eval/logs", "outputs"
    ]
    project_root = Path(__file__).parent.parent
    for d in expected_dirs:
        p = project_root / d
        if check(f"dir: {d}", p.exists()):
            results["pass"] += 1
        else:
            results["fail"] += 1
            print(f"         ➡️  mkdir -p {d}")

    # ── Summary ──────────────────────────────────────────────
    header("Summary")
    total = results["pass"] + results["fail"]
    print(f"  Passed: {results['pass']}/{total}")
    print(f"  Failed: {results['fail']}/{total}")
    print(f"  Warnings: {results['warn']}")

    if results["fail"] == 0:
        print("\n  🚀 All checks passed! Ready to proceed to Step 2 (data download).")
    else:
        print(f"\n  ⚠️  {results['fail']} check(s) failed. Fix the issues above before proceeding.")
        print("  Most critical: CUDA, PyTorch, transformers, peft, trl, datasets, HF login.")

    return 0 if results["fail"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())