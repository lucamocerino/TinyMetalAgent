from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

from tinyengine import Model


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


def run_tiny(args: argparse.Namespace) -> str:
    output = subprocess.run(
        [str(args.tiny_bin), str(args.gguf), args.prompt, str(args.c_max_tokens)],
        check=True,
        capture_output=True,
        text=True,
    )
    return clean_generated_text(output.stdout)


def run_tiny_prompt_tokens(args: argparse.Namespace) -> int:
    with Model(args.gguf) as model:
        return len(model.tokenize(qwen_chat_prompt(args.prompt), parse_special=True))


def run_llama(args: argparse.Namespace) -> tuple[str, dict[str, float | int]]:
    env = os.environ.copy()
    env["LC_ALL"] = "C"
    env["LANG"] = "C"
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
    load_ms, _ = parse_timing(output.stderr, "load time")
    prompt_eval_ms, prompt_tokens = parse_timing(output.stderr, "prompt eval time")
    eval_ms, eval_runs = parse_timing(output.stderr, "eval time")
    total_ms, total_tokens = parse_timing(output.stderr, "total time")
    return clean_generated_text(output.stdout), {
        "load_ms": load_ms,
        "prompt_eval_ms": prompt_eval_ms,
        "prompt_tokens": prompt_tokens or 0,
        "eval_ms": eval_ms,
        "eval_runs": eval_runs or 0,
        "total_ms": total_ms,
        "total_tokens": total_tokens or 0,
    }


def existing_file(path: str) -> Path:
    value = Path(path)
    if not value.is_file():
        raise argparse.ArgumentTypeError(f"not a file: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare TinyEngine C generation with llama.cpp oracle")
    parser.add_argument("--tiny-bin", type=existing_file, required=True)
    parser.add_argument("--llama-bin", type=existing_file, required=True)
    parser.add_argument("--gguf", type=existing_file, required=True)
    parser.add_argument("--prompt", default="Rispondi in italiano con tre parole: cosa sei?")
    parser.add_argument("--c-max-tokens", type=int, default=3)
    parser.add_argument("--llama-predict", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-layers", default="999")
    parser.add_argument("--ctx-size", type=int, default=512)
    args = parser.parse_args()

    tiny_prompt_tokens = run_tiny_prompt_tokens(args)
    tiny_text = run_tiny(args)
    llama_text, timings = run_llama(args)
    if tiny_text != llama_text:
        print("TinyEngine C and llama.cpp generated text differ", file=sys.stderr)
        print(f"TinyEngine C: {tiny_text!r}", file=sys.stderr)
        print(f"llama.cpp:    {llama_text!r}", file=sys.stderr)
        return 1
    if tiny_prompt_tokens != timings["prompt_tokens"]:
        print("TinyEngine C and llama.cpp prompt token counts differ", file=sys.stderr)
        print(f"TinyEngine C: {tiny_prompt_tokens}", file=sys.stderr)
        print(f"llama.cpp:    {timings['prompt_tokens']}", file=sys.stderr)
        return 1

    print(
        "oracle-ok "
        f"text={tiny_text!r} "
        f"prompt_tokens={tiny_prompt_tokens} "
        f"eval_runs={timings['eval_runs']} "
        f"llama_prompt_ms={timings['prompt_eval_ms']:.2f} "
        f"llama_eval_ms={timings['eval_ms']:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
