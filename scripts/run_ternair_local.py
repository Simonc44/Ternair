"""run_ternair_local.py -- Run a HuggingFace ternary Mistral / LLaMA model on
your local box, with the Ternair native C++ engine (libternair_native) auto-
built and auto-wired into the kernel dispatcher.

What this script does, end-to-end
--------------------------------
0.  Detect the host OS (Windows / Linux / macOS).
1.  If ``g++`` is on PATH, compile ``libternair_native.{dll,so,dylib}`` from
    ``src/ternair/native/src/*.cpp``, auto-detecting AVX-512 / AVX-2 /
    scalar.  If ``g++`` is missing, fall back to the pure-Python
    backends (numpy / torch) and emit a one-line tip explaining how
    to enable the native backend later.
2.  Make the resulting library findable by Python ``ctypes``
    (PATH on Windows, ``LD_LIBRARY_PATH`` on Linux,
    ``DYLD_LIBRARY_PATH`` on macOS) BEFORE importing ternair.
3.  ``from ternair import load_ternair_model`` -- this auto-detects
    the LLaMA / Mistral ternary bundle (``model_ternair_2bit.safetensors``)
    and replaces every ``nn.Linear`` with a ``TernaryLinearFast`` whose
    backend is resolved at first forward: native > numpy > torch on
    CPU, triton > torch on CUDA.
4.  Run the prompt through ``model.generate(...)`` and print the result.
5.  Show a tiny summary table at the end (resolved backend per layer).

Usage (Windows, PowerShell)
----------------------------
    py scripts\\run_ternair_local.py ^
        --model-dir "C:\\Users\\admin\\Documents\\Mistral 7B\\mistral-7b-v0.3-ternair" ^
        --prompt "The future of AI is" ^
        --max-new-tokens 80 ^
        --temperature 0.7 --top-p 0.9 --repetition-penalty 1.1

Usage (Linux / macOS)
---------------------
    python3 scripts/run_ternair_local.py \\
        --model-dir /path/to/mistral-7b-v0.3-ternair \\
        --prompt "The future of AI is" \\
        --max-new-tokens 80

Useful flags
------------
    --backend {auto,native,numpy,torch,triton}
        Force a specific kernel backend.  Default = ``auto``.
    --skip-build
        Skip the g++ build step (use it when you've already built
        ``libternair_native`` once and don't want to wait again).
    --threads N
        Number of worker threads for the native threadpool
        (0 = auto, default).
    --show-resolved
        Print the resolved inference backend for every ternary
        layer in the model.

Requirements
------------
    * Python >= 3.10
    * torch >= 2.4
    * numpy
    * transformers + safetensors (for the HF loader)
    * g++ (only if you want the native backend; otherwise the script
      silently falls back to numpy / torch)

On Windows, g++ is not installed by default.  Either install MSYS2
(https://www.msys2.org/) and run ``pacman -S mingw-w64-x86_64-gcc``,
or run everything under WSL.
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


# ===========================================================================
# 1. Platform + paths
# ===========================================================================

REPO_ROOT = Path(__file__).resolve().parent.parent
NATIVE_DIR = REPO_ROOT / "src" / "ternair" / "native"
NATIVE_SRC = NATIVE_DIR / "src"
NATIVE_BUILD = NATIVE_DIR / "build"

IS_WINDOWS = platform.system() == "Windows"
IS_LINUX = platform.system() == "Linux"
IS_MAC = platform.system() == "Darwin"

if IS_WINDOWS:
    LIB_EXT = ".dll"
    LIB_BASENAME = "ternair_native.dll"
    LIB_ENV_VAR = "PATH"
elif IS_MAC:
    LIB_EXT = ".dylib"
    LIB_BASENAME = "ternair_native.dylib"
    LIB_ENV_VAR = "DYLD_LIBRARY_PATH"
else:
    LIB_EXT = ".so"
    LIB_BASENAME = "libternair_native.so"
    LIB_ENV_VAR = "LD_LIBRARY_PATH"


# ===========================================================================
# 2. Native engine build + library path
# ===========================================================================

def _has_gxx() -> str | None:
    """Return the absolute path to ``g++`` if available, else ``None``."""
    found = shutil.which("g++")
    if found:
        return found
    # On Windows, MSYS2 ships ``x86_64-w64-mingw32-g++`` and ``g++`` on PATH
    # once you run ``pacman -S mingw-w64-x86_64-gcc``.
    for cand in ("x86_64-w64-mingw32-g++", "g++.exe", "gcc.exe"):
        p = shutil.which(cand)
        if p:
            return p
    return None


def _cpu_features(gxx: str) -> str:
    """Return the g++ ISA flags appropriate for this CPU."""
    if not IS_LINUX and not IS_WINDOWS:
        return ""  # macOS Clang behaves a bit differently; keep it simple.
    if shutil.which("grep") is None:
        return ""
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as fh:
            text = fh.read()
    except OSError:
        return ""
    if "avx512f" in text and "avx512dq" in text:
        return "-mavx512f -mavx512bw -mavx512dq -mavx2 -mf16c -mfma"
    if "avx2" in text:
        return "-mavx2 -mf16c -mfma"
    return ""


def build_native_engine(gxx: str, *, verbose: bool = True) -> Path:
    """Compile libternair_native.{dll,so,dylib} with auto-detected SIMD ISA.

    Returns the absolute path to the freshly built library.
    """
    NATIVE_BUILD.mkdir(parents=True, exist_ok=True)
    lib_path = NATIVE_BUILD / LIB_BASENAME

    sources = [
        NATIVE_SRC / "matmul_scalar.cpp",
        NATIVE_SRC / "matmul_dispatch.cpp",
        NATIVE_SRC / "threadpool.cpp",
        NATIVE_SRC / "loader.cpp",
        NATIVE_SRC / "norm.cpp",
        NATIVE_SRC / "runtime.cpp",
    ]
    cpu_flags = _cpu_features(gxx)
    if "-mavx512f" in cpu_flags:
        sources.append(NATIVE_SRC / "matmul_avx512.cpp")
    if "-mavx2" in cpu_flags:
        sources.append(NATIVE_SRC / "matmul_avx2.cpp")

    common_flags = [
        "-std=c++17", "-O3", "-fPIC",
        "-fvisibility=default",  # CRITICAL for ctypes to resolve symbols
        "-Wall", "-Wextra", "-DNDEBUG",
    ]
    if cpu_flags:
        common_flags += cpu_flags.split()

    cmd = [
        gxx, *common_flags, "-shared",
        "-I", str(NATIVE_DIR / "include"),
        "-o", str(lib_path),
        *[str(s) for s in sources],
        "-lpthread",
    ]
    if verbose:
        print(f"[build] g++ -> {lib_path.name}")
        print(f"[build] sources: {len(sources)} .cpp files, ISA flags: "
              f"{cpu_flags or '(scalar only)'}")
    subprocess.run(cmd, check=True)
    return lib_path


def ensure_library_on_path(lib_path: Path) -> str:
    """Add ``lib_path`` to the OS library search path.  Returns the absolute path."""
    lib_abs = str(lib_path.resolve())
    current = os.environ.get(LIB_ENV_VAR, "")
    if lib_abs not in current.split(os.pathsep):
        os.environ[LIB_ENV_VAR] = (
            lib_abs + (os.pathsep + current if current else "")
        )
    # Windows additionally needs the DLL directory on PATH for ctypes.
    if IS_WINDOWS:
        lib_dir = str(lib_path.resolve().parent)
        if lib_dir not in os.environ.get("PATH", "").split(os.pathsep):
            os.environ["PATH"] = (
                lib_dir + os.pathsep + os.environ.get("PATH", "")
            )
    return lib_abs


# ===========================================================================
# 3. CLI parsing
# ===========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run a Ternair-exported Mistral / LLaMA model on your box.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model-dir",
        required=True,
        help="Directory holding config.json, tokenizer files, and "
             "model_ternair_2bit.safetensors.",
    )
    p.add_argument(
        "--safetensors-name",
        default="model_ternair_2bit.safetensors",
        help="Filename of the ternary bundle inside --model-dir.",
    )
    p.add_argument(
        "--prompt",
        default="The future of artificial intelligence is",
        help="Prompt to feed the model.",
    )
    p.add_argument(
        "--max-new-tokens", type=int, default=80,
        help="Number of new tokens to generate.",
    )
    p.add_argument(
        "--temperature", type=float, default=0.7,
        help="Sampling temperature (0 = greedy).",
    )
    p.add_argument(
        "--top-p", type=float, default=0.9,
        help="Top-P nucleus sampling (1.0 = disabled).",
    )
    p.add_argument(
        "--top-k", type=int, default=40,
        help="Top-K filtering (0 = disabled).",
    )
    p.add_argument(
        "--repetition-penalty", type=float, default=1.1,
        help="Repetition penalty (1.0 = disabled).",
    )
    p.add_argument(
        "--device",
        default=None,
        help="Target device (cuda / cpu).  Auto-detected if None.",
    )
    p.add_argument(
        "--backend",
        default="auto",
        choices=("auto", "native", "numpy", "torch", "triton"),
        help="Force a specific kernel backend.  Default: auto "
             "(native > numpy > torch on CPU, triton > torch on CUDA).",
    )
    p.add_argument(
        "--threads", type=int, default=0,
        help="Worker threads for the native threadpool (0 = auto).",
    )
    p.add_argument(
        "--skip-build", action="store_true",
        help="Skip the g++ build step (use it when libternair_native is "
             "already built).",
    )
    p.add_argument(
        "--no-native", action="store_true",
        help="Disable the native backend even if g++ is available "
             "(useful when something looks wrong in the C kernels).",
    )
    p.add_argument(
        "--show-resolved", action="store_true",
        help="Print the resolved inference backend for every ternary layer.",
    )
    return p.parse_args()


# ===========================================================================
# 4. Pretty banner
# ===========================================================================

def banner(msg: str, char: str = "=") -> None:
    bar = char * max(60, len(msg) + 4)
    print(f"\n{bar}\n  {msg}\n{bar}")


# ===========================================================================
# 5. Main
# ===========================================================================

def main() -> int:
    args = parse_args()

    banner("Ternair -- local runner")
    print(f"  system      : {platform.system()} {platform.release()}")
    print(f"  python      : {sys.version.split()[0]}")
    print(f"  cpu         : {platform.processor() or 'unknown'}")
    print(f"  model-dir   : {args.model_dir}")
    print(f"  safetensors : {args.safetensors_name}")
    print(f"  prompt      : {args.prompt[:80]!r}{'...' if len(args.prompt) > 80 else ''}")
    print(
        f"  sampling    : T={args.temperature} top_k={args.top_k} "
        f"top_p={args.top_p} rep_pen={args.repetition_penalty}"
    )
    print(f"  backend     : {args.backend}")

    # ------------------------------------------------------------------
    # Step 1: build the native C++ engine if possible
    # ------------------------------------------------------------------
    banner("Step 1: native C++ engine")
    native_lib_path: Path | None = None

    if args.no_native:
        print("  --no-native: skipping native backend.")
    elif args.skip_build:
        candidate = NATIVE_BUILD / LIB_BASENAME
        if candidate.exists():
            native_lib_path = candidate
            print(f"  --skip-build: re-using {candidate}")
        else:
            print(f"  --skip-build but {candidate} missing -- will try to build.")
            args.skip_build = False  # fall through to the build

    if native_lib_path is None and not args.no_native:
        gxx = _has_gxx()
        if gxx is None:
            print("  g++ NOT FOUND on PATH.  Falling back to numpy/torch.")
            print("  To enable the native backend:")
            print("    - Linux / macOS: install build-essential / Xcode CLT")
            print("    - Windows : install MSYS2 and run "
                  "`pacman -S mingw-w64-x86_64-gcc`")
        else:
            try:
                native_lib_path = build_native_engine(gxx)
            except subprocess.CalledProcessError as exc:
                print(f"  g++ build FAILED (rc={exc.returncode}).  "
                      f"Falling back to numpy/torch.")
            except Exception as exc:  # pragma: no cover
                print(f"  unexpected build error: {exc!r}")
                print(f"  Falling back to numpy/torch.")

    if native_lib_path is not None:
        ensure_library_on_path(native_lib_path)
        print(f"  {LIB_BASENAME} -> {native_lib_path}")
        print(f"  {LIB_ENV_VAR} updated.")
    else:
        print(f"  -- no native backend ({args.backend} will fall back accordingly) --")

    # ------------------------------------------------------------------
    # Step 2: native module self-test (round trip)
    # ------------------------------------------------------------------
    banner("Step 2: native module self-test")
    try:
        from ternair import native as _native
    except Exception as exc:
        print(f"  could not import ternair.native: {exc!r}")
        _native = None
    if _native is not None:
        print(f"  ternair.native available()      -> {_native.available()}")
        print(f"  ternair.native backend_name()  -> (lazy)")
    else:
        print("  native module not importable (will rely on numpy/torch).")

    # ------------------------------------------------------------------
    # Step 3: load the model
    # ------------------------------------------------------------------
    banner("Step 3: load model")
    import torch
    from ternair import load_ternair_model

    model_dir = Path(args.model_dir).expanduser().resolve()
    if not model_dir.exists():
        print(f"  ERROR: --model-dir {model_dir} does not exist.")
        return 2
    safetensors_full = model_dir / args.safetensors_name
    if not safetensors_full.exists():
        print(f"  ERROR: {safetensors_full} missing.")
        return 2

    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"  device  : {device}")
    print(f"  loading : {model_dir}")

    try:
        model, tokenizer, report = load_ternair_model(
            str(model_dir),
            device=device,
            safetensors_name=args.safetensors_name,
            backend=args.backend,
        )
    except Exception as exc:
        print(f"  load_ternair_model raised: {type(exc).__name__}: {exc}")
        return 3

    print(f"  load complete:")
    print(f"    schema              : {report.schema}")
    print(f"    ternary layers      : {report.n_ternary_layers}")
    print(f"    fp16 tensors        : {report.n_fp16_tensors}")
    print(f"    meta materialised   : {report.n_meta_materialised}")
    print(f"    ignored             : {report.n_ignored}")
    print(f"    skipped duplicates  : {report.n_skipped_duplicates}")

    # ------------------------------------------------------------------
    # Step 4: generate
    # ------------------------------------------------------------------
    banner("Step 4: generate")

    if tokenizer is None:
        print("  tokenizer NOT LOADED -- falling back to integer-token prompt.")
        try:
            ids = torch.tensor(
                [list(map(int, args.prompt.split(",")))], dtype=torch.long
            ).to(device)
        except ValueError:
            print(f"  ERROR: --prompt is non-numeric and no tokenizer was "
                  f"found in {model_dir}.")
            return 4
    else:
        inputs = tokenizer(args.prompt, return_tensors="pt").to(device)
        ids = inputs["input_ids"]
        print(f"  prompt tokens (first 20): {ids[0, :20].tolist()}")

    with torch.no_grad():
        try:
            outputs = model.generate(
                input_ids=ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.temperature > 0,
                temperature=max(args.temperature, 1e-5),
                top_p=args.top_p,
                top_k=args.top_k,
                repetition_penalty=args.repetition_penalty,
                pad_token_id=(
                    tokenizer.eos_token_id if tokenizer is not None
                    else ids[0, -1].item()
                ),
            )
        except Exception as exc:
            print(f"  generate raised: {type(exc).__name__}: {exc}")
            return 5

    print()
    print("  " + "=" * 60)
    if tokenizer is not None:
        print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    else:
        print(" ".join(str(int(t)) for t in outputs[0].tolist()))
    print("  " + "=" * 60)

    # ------------------------------------------------------------------
    # Step 5: resolved-backend summary
    # ------------------------------------------------------------------
    if args.show_resolved:
        banner("Step 5: resolved inference backends")
        try:
            from ternair.model.loader import TernaryLinearFast
        except Exception as exc:
            print(f"  cannot import TernaryLinearFast: {exc!r}")
        else:
            n = {"torch": 0, "numpy": 0, "native": 0, "triton": 0, "auto": 0}
            for mod in model.modules():
                if isinstance(mod, TernaryLinearFast):
                    b = mod._resolved_backend or mod.backend
                    n[str(b)] = n.get(str(b), 0) + 1
            total = sum(n.values())
            for k, v in n.items():
                if v:
                    pct = 100.0 * v / max(total, 1)
                    print(f"    {k:<8s}: {v:4d}  ({pct:5.1f} %)")
            print(f"    total   : {total} ternary layers")

    banner("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
