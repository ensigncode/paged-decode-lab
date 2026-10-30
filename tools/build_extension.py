#!/usr/bin/env python3
"""Compile and link the CUDA extension, including on a GPU-less build host.

This is a build check only. Successful compilation does not establish GPU
correctness, sanitizer acceptance, or speed. Architecture targets are explicit
so PyTorch never has to infer them from an attached device.
"""
import argparse
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architectures", default="7.5;8.0",
                        help="PyTorch TORCH_CUDA_ARCH_LIST, for example '7.5;8.0'")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # Keep the compiler's peak host-memory use predictable on small machines.
    os.environ.setdefault("MAX_JOBS", "1")
    os.environ["TORCH_CUDA_ARCH_LIST"] = args.architectures
    import torch
    from torch.utils.cpp_extension import CUDA_HOME
    from paged_decode.runtime import load_extension
    if torch.version.cuda is None:
        raise SystemExit("CUDA-enabled PyTorch required for a CUDA build")
    nvcc = str(Path(CUDA_HOME) / "bin/nvcc") if CUDA_HOME else shutil.which("nvcc")
    if not nvcc or not Path(nvcc).exists():
        raise SystemExit("CUDA toolkit not found; set CUDA_HOME to its installation")
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite {args.output}")
    compiler = subprocess.check_output([nvcc, "--version"], text=True)
    module = load_extension()
    report = {
        "schema_version": 1,
        "kind": "cuda_extension_build",
        "status": "passed",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "nvcc": compiler,
        "architectures": args.architectures,
        "extension_file": Path(module.__file__).name,
        "gpu_available": torch.cuda.is_available(),
        "scope": "CUDA/C++ compilation, device code generation, linking and module import only",
        "gpu_execution_validated": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
