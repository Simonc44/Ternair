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
    --mini-smoke
        Skip --model-dir entirely and run the **bundled tiny** model
        (``ternair.model.size_profiles.tiny_profile`` = hidden 256,
        8 layers, ~2.6 M params).  Uses the in-package ``CharTokenizer``
        so no transformer/safetensors download is needed.  Finishes in
        **~5 seconds on CPU** -- the canonical way to verify your
        Ternair install before attempting a 7B model.
    --abort-no-progress SEC
        Abort generation cleanly if no token comes out within ``SEC``
        seconds.  Use this to avoid waiting hours on a hung CPU/numpy
        inference on a 7B model.  ``0`` = disabled (default).  Suggested
        values: ``600`` for 7B CPU/numpy, ``60`` for tiny, ``30`` for
        CUDA + triton.

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

Quick start (recommended first step)
------------------------------------
1.  Verify the install in 5 seconds with the bundled tiny model
    (runs end-to-end without a downloaded 7B checkpoint)::

        py scripts\\run_ternair_local.py --mini-smoke --max-new-tokens 16

2.  Install g++ (Windows + MSYS2) so the native AVX-2 backend kicks
    in automatically and 7B inference becomes ~10x faster::

        # https://www.msys2.org/ then:
        pacman -S mingw-w64-x86_64-gcc

3.  Finally, run the real 7B model -- and add --abort-no-progress
    so a hang never costs you more than a few minutes::

        py scripts\\run_ternair_local.py ^
            --model-dir "C:\\Users\\admin\\Documents\\Mistral 7B\\mistral-7b-v0.3-ternair" ^
            --prompt "Hello" ^
            --max-new-tokens 4 ^
            --abort-no-progress 300
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
    """Return the g++ ISA flags appropriate for this CPU.

    On Windows / macOS, ``/proc/cpuinfo`` is unavailable so we cannot do a
    precise feature probe.  Defaulting to ``-mavx2 -mf16c -mfma`` is safe
    because every x86_64 CPU since Haswell (2013) supports the AVX2
    family.  The C++ runtime in
    ``src/ternair/native/src/matmul_dispatch.cpp`` does its own runtime
    CPUID detection and silently falls back to scalar when called on a
    machine without AVX2 -- so even if our compile-time assumption is
    wrong on some exotic target, no crash, just slower code.

    On Linux we read ``/proc/cpuinfo`` for AVX-512 detection.
    """
    if IS_WINDOWS or IS_MAC:
        return "-mavx2 -mf16c -mfma"
    if not IS_LINUX:
        return ""
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
        "-fvisibility=default",  # ELF: makes all symbols candidate-exportable
        "-Wall", "-Wextra", "-DNDEBUG",
    ]
    if cpu_flags:
        common_flags += cpu_flags.split()
    # CRITICAL on Windows mingw / MSYS2: -fvisibility=default is an ELF
    # concept only.  ``g++ -shared`` on MinGW does NOT auto-populate the
    # .dll export table; ctypes.LoadLibrary() would succeed and then
    # raise ``AttributeError: function 'ternair_create' not found`` at
    # first attribute lookup.  -Wl,--export-all-symbols forces every
    # extern "C" symbol into the export table.
    if IS_WINDOWS:
        common_flags.append("-Wl,--export-all-symbols")

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
        "--mini-smoke", action="store_true",
        help="Skip --model-dir; run the bundled tiny model (~2.6 M params) "
             "with the in-package CharTokenizer.  No downloads, ~5 s on CPU.",
    )
    p.add_argument(
        "--model-dir",
        required=False,  # --mini-smoke makes it optional
        help="Directory holding config.json, tokenizer files, and "
             "model_ternair_2bit.safetensors.  Not required when --mini-smoke is set.",
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
    p.add_argument(
        "--abort-no-progress", type=int, default=0,
        help="If no token is produced within this many seconds, abort "
             "generation cleanly (returns exit code 8).  0 = disabled.",
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
    # Step 3: load the model  (or fall back to --mini-smoke)
    # ------------------------------------------------------------------
    import torch

    if args.mini_smoke and args.model_dir:
        print("  ERROR: --mini-smoke cannot be combined with --model-dir.")
        return 2

    # Resolve target device early so the mini-smoke branch (which skips
    # HF loader entirely) can still use it.
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # If --mini-smoke, build the bundled tiny model and skip everything
    # below.  This is the canonical "is my Ternair install healthy?"
    # check -- it completes in ~5 s on CPU/numpy and exercises every
    # code path that the 7B runner does (just on a 2.6 M-parameter
    # model that fits in cache).
    model = None
    tokenizer = None
    report = None
    if args.mini_smoke:
        from ternair.model.modeling import TernairForCausalLM
        from ternair.model.size_profiles import tiny_profile
        from ternair.model.generation import generate as _generate_tokens
        from ternair.training.data import CharTokenizer, DEFAULT_CORPUS

        banner("Step 3: load model  [MINI-SMOKE]")
        print("  using bundled tiny profile (hidden 256, 8 layers).")
        cfg = tiny_profile()
        print(f"  building model on device={device} ...", flush=True)
        model = TernairForCausalLM(cfg)
        print(f"  ternary params : {model.count_parameters():,}", flush=True)
        print(f"  freeze_storage() ...", flush=True)
        model.freeze_storage()
        model.eval()
        model.to(device)
        print(f"  loaded on {device.upper()}.")

        tok = CharTokenizer(DEFAULT_CORPUS)
        ids = torch.tensor(
            [tok.bos_id] + tok.encode(args.prompt)
        , dtype=torch.long).unsqueeze(0).to(device)
        attention_mask = torch.ones_like(ids)
        report = None  # mini-smoke has no LoadReport, but downstream needs it
        print(f"  prompt tokens (first 20): {ids[0, :20].tolist()}")
        # Jump straight to generation (skip Step 4 streaming wrapper).
        banner("Step 4: generate  [MINI-SMOKE]")
        print(f"\n  --- generation ({args.max_new_tokens} tokens max) ---")
        import time as _t
        t0 = _t.perf_counter()
        try:
            out = _generate_tokens(
                model, ids,
                max_new_tokens=args.max_new_tokens,
                temperature=max(args.temperature, 1e-5),
                top_k=args.top_k,
                top_p=args.top_p,
                eos_token_id=tok.eos_id,
            )
        except Exception as exc:
            print(f"  generate raised: {type(exc).__name__}: {exc}")
            return 5
        decoded = tok.decode(out[0].tolist())
        print(decoded)
        elapsed = _t.perf_counter() - t0
        print(f"\n  --- done in {elapsed:6.2f}s ---")
        print(f"  codebase self-test PASSED in {elapsed:.2f}s.")
        banner("Done.")
        return 0

    banner("Step 3: load model")
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

    # Warn early when the user is about to do autoregressive generation on
    # CPU without the native SIMD engine. The numpy / torch fallback on a
    # 7B-class ternary model is realistically O(minutes) per token, so
    # --max-new-tokens 80 will take hours. Suggest --max-new-tokens 4 for
    # a quick smoke test, or install MSYS2 g++ for ~10x speedup.
    try:
        from ternair import native as _native_avail
    except Exception:
        _native_avail = None
    native_ok = bool(_native_avail and _native_avail.available())
    if (not native_ok) and (report.n_ternary_layers > 100) and (args.max_new_tokens > 8):
        print(
            f"  WARNING: native backend unavailable but model has "
            f"{report.n_ternary_layers} ternary layers and you asked for "
            f"{args.max_new_tokens} tokens. CPU/numpy generation will be "
            f"VERY SLOW (often 1+ minute per token on a 7B model).\n"
            f"  Tip 1: re-run with --max-new-tokens 4 for an end-to-end smoke test.\n"
            f"  Tip 2: install MSYS2 then `pacman -S mingw-w64-x86_64-gcc` for "
            f"AVX-2 acceleration (~10x).\n"
            f"  Tip 3: GPU + --backend triton is even faster.\n"
            f"  Continuing in 3 seconds (Ctrl-C to abort)...\n"
        )
        try:
            import time as _t
            _t.sleep(3)
        except KeyboardInterrupt:
            print("  aborted.")
            return 6

    if tokenizer is None:
        print("  tokenizer NOT LOADED -- falling back to integer-token prompt.")
        try:
            ids = torch.tensor(
                [list(map(int, args.prompt.split(",")))], dtype=torch.long
            ).to(device)
            attention_mask = torch.ones_like(ids)
        except ValueError:
            print(f"  ERROR: --prompt is non-numeric and no tokenizer was "
                  f"found in {model_dir}.")
            return 4
    else:
        enc = tokenizer(args.prompt, return_tensors="pt", padding=False)
        ids = enc["input_ids"].to(device)
        # Explicit attention_mask (kills the transformers warning and avoids
        # the model treating pad=eos as "all-eos").
        attention_mask = enc.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(ids)
        else:
            attention_mask = attention_mask.to(device)
        print(f"  prompt tokens (first 20): {ids[0, :20].tolist()}")

    # ------------------------------------------------------------------
    # Streaming via TextIteratorStreamer -- lets the user see the
    # generation progress token-by-token instead of waiting for the whole
    # max_new_tokens pass to finish. Returns output to the precise same
    # final string as a blocking model.generate() call would.
    # ------------------------------------------------------------------
    import time
    from threading import Thread

    streamer = None
    try:
        if tokenizer is not None:
            from transformers import TextIteratorStreamer  # type: ignore
            streamer = TextIteratorStreamer(
                tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
            )
    except Exception as exc:
        streamer = None
        print(f"  (TextIteratorStreamer unavailable: {exc!r}; using blocking generate)")

    pad_token_id = (
        tokenizer.eos_token_id if tokenizer is not None else int(ids[0, -1].item())
    )

    gen_kwargs = dict(
        input_ids=ids,
        attention_mask=attention_mask,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.temperature > 0,
        temperature=max(args.temperature, 1e-5),
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        pad_token_id=pad_token_id,
    )
    if streamer is not None:
        gen_kwargs["streamer"] = streamer

    print(f"\n  --- generation ({args.max_new_tokens} tokens max) ---")
    t_start = time.perf_counter()
    n_tokens_emitted = 0

    if streamer is not None:
        # ------------------------------------------------------------------
        # --abort-no-progress watchdog: a one-shot threading.Timer that
        # fires os._exit(8) if no token comes out within N seconds.
        # Every chunk resets the timer.  Cancelled on completion.
        # This is the exact fix for the user-reported "stuck for 2h"
        # symptom on CPU/numpy with a 7B model.
        # ------------------------------------------------------------------
        import threading as _threading

        class _GenerationWatchdog:
            def __init__(self, max_sec: int) -> None:
                if max_sec <= 0:
                    self._timer = None
                    return
                self.max_sec = max_sec
                self._timer = _threading.Timer(max_sec, self._fire)
                self._timer.daemon = True
                self._timer.start()

            def ping(self) -> None:
                if self._timer is None:
                    return
                self._timer.cancel()
                self._timer = _threading.Timer(self.max_sec, self._fire)
                self._timer.daemon = True
                self._timer.start()

            def cancel(self) -> None:
                if self._timer is not None:
                    self._timer.cancel()

            def _fire(self) -> None:
                print(
                    f"\n\n  === ABORT ===\n"
                    f"  no token emitted within {self.max_sec} s.\n"
                    f"  For a 7B model on CPU without the native C++ engine,\n"
                    f"  each token can take 5-30 minutes.  If you expected fast\n"
                    f"  generation, install MSYS2 + `pacman -S mingw-w64-x86_64-gcc`\n"
                    f"  so the AVX-2 backend kicks in (Linux/macOS: build-essential).\n"
                    f"  Exiting with code 8.\n",
                    flush=True,
                )
                os._exit(8)

        watchdog = _GenerationWatchdog(args.abort_no_progress)
        if args.abort_no_progress > 0:
            print(f"  watchdog: abort if no token in {args.abort_no_progress}s "
                  f"(Ctrl-C to cancel earlier).", flush=True)

        # Background thread runs model.generate(...); main thread drains
        # the streamer and prints chunks as they arrive.
        gen_thread = Thread(target=model.generate, kwargs=gen_kwargs, daemon=True)
        gen_thread.start()
        try:
            for chunk in streamer:
                watchdog.ping()
                print(chunk, end="", flush=True)
                n_tokens_emitted += len(chunk.split()) if chunk else 0
        except KeyboardInterrupt:
            watchdog.cancel()
            print("\n  interrupted (Ctrl-C).  Cancelling generation ...", flush=True)
            gen_thread.join(timeout=2.0)
            return 7
        watchdog.cancel()
        gen_thread.join()
    else:
        # Fallback: blocking generate. Time it so the user has a signal.
        watchdog = _GenerationWatchdog(args.abort_no_progress)
        if args.abort_no_progress > 0:
            print(f"  watchdog: abort if generate() does not return "
                  f"within {args.abort_no_progress}s.", flush=True)
        with torch.no_grad():
            try:
                outputs = model.generate(**gen_kwargs)
                watchdog.ping()  # generation returned -- safe to cancel
            except Exception as exc:
                watchdog.cancel()
                print(f"  generate raised: {type(exc).__name__}: {exc}")
                return 5
        if tokenizer is not None:
            print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        else:
            print(" ".join(str(int(t)) for t in outputs[0].tolist()))
        watchdog.cancel()

    elapsed = time.perf_counter() - t_start
    print(f"\n  --- done in {elapsed:6.1f}s ---")

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
