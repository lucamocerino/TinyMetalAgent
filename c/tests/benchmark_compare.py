from __future__ import annotations

import argparse
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from tinyengine import Model, RuntimeOptions, detect_arch


def clean_generated_text(text: str) -> str:
    return "".join(ch for ch in text if ch in "\n\t" or ord(ch) >= 32).strip()


def qwen_chat_prompt(prompt: str) -> str:
    return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"


def parse_timing(stderr: str, label: str) -> tuple[float, int | None]:
    for line in stderr.splitlines():
        if "common_perf_print:" in line:
            line = line.split("common_perf_print:", 1)[1]
        if not line.strip().startswith(label):
            continue
        value_match = re.search(r"=\s*([0-9]+(?:[,.][0-9]+)?)\s*ms", line)
        if value_match is None:
            raise RuntimeError(f"could not parse timing value from: {line}")
        count_match = re.search(r"/\s*(\d+)\s+", line)
        value = float(value_match.group(1).replace(",", "."))
        count = int(count_match.group(1)) if count_match is not None else None
        return value, count
    raise RuntimeError(f"missing llama.cpp timing line: {label}")


def tokens_per_second(tokens: int, elapsed_ms: float) -> float:
    if tokens <= 0 or elapsed_ms <= 0.0:
        return 0.0
    return float(tokens) * 1000.0 / elapsed_ms


def safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0.0:
        return 0.0
    return numerator / denominator


def median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def run_tiny(args: argparse.Namespace, run_index: int) -> tuple[str, dict[str, Any], dict[str, Any]]:
    options = RuntimeOptions(context_tokens=args.ctx_size)
    model: Model | None = None
    started = time.perf_counter()
    try:
        model = Model(args.gguf, options)
        loaded = time.perf_counter()
        model_info = model.info()
        tokenizer_info = model.tokenizer_info()
        prompt_ids = model.tokenize(qwen_chat_prompt(args.prompt), parse_special=True)
        text = model.generate(args.prompt, args.c_max_tokens)
        generated = time.perf_counter()
    finally:
        if model is not None:
            model.close()
    finished = time.perf_counter()

    generate_ms = (generated - loaded) * 1000.0
    run = {
        "run_index": run_index,
        "load_ms": (loaded - started) * 1000.0,
        "generate_ms": generate_ms,
        "total_ms": (finished - started) * 1000.0,
        "generated_tokens_requested": args.c_max_tokens,
        "prompt_tokens": len(prompt_ids),
        "prompt_token_ids": prompt_ids,
        "tokens_per_second": tokens_per_second(args.c_max_tokens, generate_ms),
    }
    metadata = {
        "model": asdict(model_info),
        "tokenizer": asdict(tokenizer_info),
    }
    return clean_generated_text(text), run, metadata


def run_llama(args: argparse.Namespace, run_index: int) -> tuple[str, dict[str, Any]]:
    env = os.environ.copy()
    env["LC_ALL"] = "C"
    env["LANG"] = "C"
    started = time.perf_counter()
    output = subprocess.run(
        [
            str(args.llama_bin),
            "-m",
            str(args.gguf),
            "-p",
            qwen_chat_prompt(args.prompt),
            "-n",
            str(args.llama_predict),
            "--temp",
            "0",
            "--top-k",
            "1",
            "--seed",
            str(args.seed),
            "--no-display-prompt",
            "-no-cnv",
            "--simple-io",
            "-ngl",
            str(args.gpu_layers),
            "-c",
            str(args.ctx_size),
            "--no-warmup",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    wall_ms = (time.perf_counter() - started) * 1000.0
    load_ms, _ = parse_timing(output.stderr, "load time")
    prompt_eval_ms, prompt_tokens = parse_timing(output.stderr, "prompt eval time")
    eval_ms, eval_runs = parse_timing(output.stderr, "eval time")
    total_ms, total_tokens = parse_timing(output.stderr, "total time")
    eval_count = eval_runs or 0
    return clean_generated_text(output.stdout), {
        "run_index": run_index,
        "load_ms": load_ms,
        "prompt_eval_ms": prompt_eval_ms,
        "prompt_tokens": prompt_tokens or 0,
        "eval_ms": eval_ms,
        "eval_runs": eval_count,
        "decode_tokens_per_second": tokens_per_second(eval_count, eval_ms),
        "total_ms": total_ms,
        "total_tokens": total_tokens or 0,
        "wall_ms": wall_ms,
    }


def summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
    tiny_load = median([float(run["tinyengine"]["load_ms"]) for run in runs])
    tiny_total = median([float(run["tinyengine"]["total_ms"]) for run in runs])
    tiny_generate = median([float(run["tinyengine"]["generate_ms"]) for run in runs])
    tiny_tps = median([float(run["tinyengine"]["tokens_per_second"]) for run in runs])
    tiny_prompt_tokens = median([float(run["tinyengine"]["prompt_tokens"]) for run in runs])
    llama_load = median([float(run["llama_cpp"]["load_ms"]) for run in runs])
    llama_total = median([float(run["llama_cpp"]["total_ms"]) for run in runs])
    llama_wall = median([float(run["llama_cpp"]["wall_ms"]) for run in runs])
    llama_prompt = median([float(run["llama_cpp"]["prompt_eval_ms"]) for run in runs])
    llama_prompt_tokens = median([float(run["llama_cpp"]["prompt_tokens"]) for run in runs])
    llama_eval = median([float(run["llama_cpp"]["eval_ms"]) for run in runs])
    llama_tps = median([float(run["llama_cpp"]["decode_tokens_per_second"]) for run in runs])
    return {
        "tinyengine_median_load_ms": tiny_load,
        "tinyengine_median_total_ms": tiny_total,
        "tinyengine_median_generate_ms": tiny_generate,
        "tinyengine_median_tokens_per_second": tiny_tps,
        "tinyengine_median_prompt_tokens": tiny_prompt_tokens,
        "tinyengine_generation_mode": os.environ.get("TINYENGINE_GENERATION", "reference"),
        "llama_cpp_median_load_ms": llama_load,
        "llama_cpp_median_total_ms": llama_total,
        "llama_cpp_median_wall_ms": llama_wall,
        "llama_cpp_median_prompt_eval_ms": llama_prompt,
        "llama_cpp_median_prompt_tokens": llama_prompt_tokens,
        "llama_cpp_median_eval_ms": llama_eval,
        "llama_cpp_median_decode_tokens_per_second": llama_tps,
        "speed_ratios": {
            "llama_cpp_total_over_tinyengine_total": safe_ratio(llama_total, tiny_total),
            "llama_cpp_total_over_tinyengine_generate": safe_ratio(llama_total, tiny_generate),
            "llama_cpp_load_plus_total_over_tinyengine_total": safe_ratio(llama_load + llama_total, tiny_total),
            "llama_cpp_wall_over_tinyengine_total": safe_ratio(llama_wall, tiny_total),
            "tinyengine_tps_over_llama_cpp_decode_tps": safe_ratio(tiny_tps, llama_tps),
        },
    }


def existing_file(path: str) -> Path:
    value = Path(path)
    if not value.is_file():
        raise argparse.ArgumentTypeError(f"not a file: {path}")
    return value


def output_path(path: str) -> Path:
    return Path(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark TinyEngine C against llama.cpp oracle")
    parser.add_argument("--llama-bin", type=existing_file, required=True)
    parser.add_argument("--gguf", type=existing_file, required=True)
    parser.add_argument("--prompt", default="Rispondi in italiano con tre parole: cosa sei?")
    parser.add_argument("--c-max-tokens", type=int, default=3)
    parser.add_argument("--llama-predict", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-layers", default="999")
    parser.add_argument("--ctx-size", type=int, default=512)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--out", type=output_path, required=True)
    parser.add_argument("--allow-mismatch", action="store_true")
    args = parser.parse_args()

    if args.runs <= 0:
        parser.error("--runs must be greater than zero")
    if args.c_max_tokens <= 0:
        parser.error("--c-max-tokens must be greater than zero")
    if args.llama_predict <= 0:
        parser.error("--llama-predict must be greater than zero")

    try:
        arch = asdict(detect_arch())
    except Exception as exc:  # pragma: no cover - benchmark metadata only
        arch = {"error": str(exc)}

    runs: list[dict[str, Any]] = []
    model_metadata: dict[str, Any] | None = None
    all_text_matches = True
    for index in range(args.runs):
        tiny_text, tiny_run, metadata = run_tiny(args, index)
        llama_text, llama_run = run_llama(args, index)
        model_metadata = metadata
        text_matches = tiny_text == llama_text
        all_text_matches = all_text_matches and text_matches
        runs.append(
            {
                "run_index": index,
                "text_match": text_matches,
                "tinyengine": {
                    **tiny_run,
                    "text": tiny_text,
                },
                "llama_cpp": {
                    **llama_run,
                    "text": llama_text,
                },
            }
        )

    tinyengine_generation_mode = os.environ.get("TINYENGINE_GENERATION", "reference")

    result = {
        "benchmark": "tinyengine-c-vs-llama.cpp",
        "tinyengine_generation_mode": tinyengine_generation_mode,
        "model_path": str(args.gguf),
        "llama_bin": str(args.llama_bin),
        "prompt": args.prompt,
        "qwen_chat_prompt": qwen_chat_prompt(args.prompt),
        "settings": {
            "runs": args.runs,
            "c_max_tokens": args.c_max_tokens,
            "llama_predict": args.llama_predict,
            "seed": args.seed,
            "gpu_layers": args.gpu_layers,
            "ctx_size": args.ctx_size,
            "tinyengine_workload": os.environ.get("TINYENGINE_WORKLOAD", "auto"),
        },
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "arch": arch,
        },
        "model": model_metadata,
        "all_text_matches": all_text_matches,
        "summary": summarize(runs),
        "runs": runs,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    summary = result["summary"]
    status = "benchmark-ok" if all_text_matches or args.allow_mismatch else "benchmark-text-mismatch"
    print(
        f"{status} out={args.out} "
        f"mode={tinyengine_generation_mode} "
        f"tiny_total_ms={summary['tinyengine_median_total_ms']:.2f} "
        f"tiny_generate_ms={summary['tinyengine_median_generate_ms']:.2f} "
        f"prompt_tokens={summary['tinyengine_median_prompt_tokens']:.0f} "
        f"llama_total_ms={summary['llama_cpp_median_total_ms']:.2f} "
        f"llama_wall_ms={summary['llama_cpp_median_wall_ms']:.2f} "
        f"llama_decode_tps={summary['llama_cpp_median_decode_tokens_per_second']:.2f} "
        f"total_ratio={summary['speed_ratios']['llama_cpp_total_over_tinyengine_total']:.2f} "
        f"generate_ratio={summary['speed_ratios']['llama_cpp_total_over_tinyengine_generate']:.2f}"
    )
    if not all_text_matches and not args.allow_mismatch:
        print("TinyEngine C and llama.cpp generated text differ; benchmark JSON was still written.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
