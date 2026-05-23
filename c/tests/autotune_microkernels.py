from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


SHORT_PROMPT = "Rispondi in italiano con tre parole: cosa sei?"
LONG_PROMPT = (
    "Contesto: questo benchmark contiene una richiesta volutamente lunga per misurare la fase di prefill su input "
    "piu estesi. Ripeti mentalmente che devi ignorare il contesto descrittivo e rispondere solo alla domanda finale. "
    "Dettagli: agente, runtime, kernel Metal, memoria KV, prompt lungo, ottimizzazione, confronto con llama.cpp, "
    "misurazione stabile, batch prefill, decode su GPU, sincronizzazioni ridotte, cache persistente, throughput, "
    "latenza, accuratezza, oracle, output breve. Ancora contesto non rilevante: il sistema deve mantenere lo stesso "
    "comportamento deterministico anche quando il prompt cresce e contiene molte parole prima della domanda finale. "
    "Domanda finale: Rispondi in italiano con tre parole: cosa sei?"
)


TUNING_ENV_KEYS = (
    "TINYENGINE_METAL_MATVEC",
    "TINYENGINE_WORKLOAD",
    "TINYENGINE_METAL_ARGMAX",
    "TINYENGINE_METAL_QKV",
    "TINYENGINE_BATCH_PREFILL",
    "TINYENGINE_Q4_LLAMA",
    "TINYENGINE_Q4_PAIR_LLAMA",
    "TINYENGINE_Q4_BATCH_LLAMA",
    "TINYENGINE_Q4_BATCH_PAIR_LLAMA",
    "TINYENGINE_Q4_MATMUL",
    "TINYENGINE_Q4_MATMUL_PAIR",
    "TINYENGINE_Q4_MATMUL_KV",
    "TINYENGINE_Q4_MATMUL_GATEUP",
    "TINYENGINE_Q8_LLAMA",
    "TINYENGINE_FUSED_DECODE_LAYER",
    "TINYENGINE_DECODE_ALL_LAYERS",
    "TINYENGINE_FUSED_PREFILL_LAYER",
    "TINYENGINE_PREFILL_ALL_LAYERS",
    "TINYENGINE_FUSED_DECODE_POST_ATTN",
    "TINYENGINE_FUSED_POST_ATTN",
    "TINYENGINE_FLASH_ATTN",
    "TINYENGINE_FLASH_ATTN_LONG_TILE",
    "TINYENGINE_FLASH_ATTN_HALF_KV",
    "TINYENGINE_FLASH_ATTN_GQA",
    "TINYENGINE_Q4_FFN_HALF",
    "TINYENGINE_Q4_FFN_PAIR_SWIGLU",
    "TINYENGINE_Q4_FFN_DOWN_ADD",
    "TINYENGINE_Q4_FFN_GATE_HALF",
)

SCRUB_ENV_KEYS = TUNING_ENV_KEYS + (
    "TINYENGINE_GENERATION",
    "TINYENGINE_PROFILE",
    "TINYENGINE_METAL_PROFILE",
)


@dataclass(frozen=True)
class Profile:
    name: str
    description: str
    env: dict[str, str]


@dataclass(frozen=True)
class Workload:
    name: str
    prompt: str
    ctx_size: int
    c_max_tokens: int
    llama_predict: int


PROFILES: dict[str, Profile] = {
    "default": Profile(
        "default",
        "Current runtime defaults after clearing tuning overrides.",
        {},
    ),
    "matmul-off": Profile(
        "matmul-off",
        "Disable Q4 mat-mat and fall back to batched matvec kernels.",
        {"TINYENGINE_Q4_MATMUL": "0"},
    ),
    "gateup-matmul-off": Profile(
        "gateup-matmul-off",
        "Keep Q4 mat-mat, but disable the fused gate/up mat-mat path.",
        {"TINYENGINE_Q4_MATMUL_GATEUP": "0"},
    ),
    "kv-matmul-on": Profile(
        "kv-matmul-on",
        "Force Q4 mat-mat for K/V projections, overriding the batch-size auto gate.",
        {"TINYENGINE_Q4_MATMUL_KV": "1"},
    ),
    "kv-matmul-off": Profile(
        "kv-matmul-off",
        "Disable Q4 mat-mat for K/V projections, overriding the batch-size auto gate.",
        {"TINYENGINE_Q4_MATMUL_KV": "0"},
    ),
    "q8-llama-off": Profile(
        "q8-llama-off",
        "Disable the llama.cpp-style Q8 lm_head kernel.",
        {"TINYENGINE_Q8_LLAMA": "0"},
    ),
    "flash-attn-off": Profile(
        "flash-attn-off",
        "Disable the online-softmax tiled Metal attention kernel.",
        {"TINYENGINE_FLASH_ATTN": "0"},
    ),
    "flash-longtile-off": Profile(
        "flash-longtile-off",
        "Disable the larger FlashAttention tile selected automatically for very long prompts.",
        {"TINYENGINE_FLASH_ATTN_LONG_TILE": "0"},
    ),
    "ffn-half-off": Profile(
        "ffn-half-off",
        "Disable half-precision FFN intermediate storage between fused up/SwiGLU and down mat-mat.",
        {"TINYENGINE_Q4_FFN_HALF": "0"},
    ),
    "ffn-down-add-off": Profile(
        "ffn-down-add-off",
        "Disable fused half-input FFN down-projection plus residual add.",
        {"TINYENGINE_Q4_FFN_DOWN_ADD": "0"},
    ),
    "ffn-down-add-on": Profile(
        "ffn-down-add-on",
        "Force fused half-input FFN down-projection plus residual add, overriding the batch-size auto gate.",
        {"TINYENGINE_Q4_FFN_DOWN_ADD": "1"},
    ),
    "ffn-pair-swiglu-off": Profile(
        "ffn-pair-swiglu-off",
        "Disable the long-prefill paired gate/up/SwiGLU Q4 mat-mat kernel.",
        {"TINYENGINE_Q4_FFN_PAIR_SWIGLU": "0"},
    ),
    "ffn-pair-swiglu-on": Profile(
        "ffn-pair-swiglu-on",
        "Force the paired gate/up/SwiGLU Q4 mat-mat kernel.",
        {"TINYENGINE_Q4_FFN_PAIR_SWIGLU": "1"},
    ),
    "ffn-gate-half-on": Profile(
        "ffn-gate-half-on",
        "Enable half-precision gate storage before fused up/SwiGLU.",
        {"TINYENGINE_Q4_FFN_GATE_HALF": "1"},
    ),
    "argmax-cpu": Profile(
        "argmax-cpu",
        "Disable the Metal output projection/argmax path.",
        {"TINYENGINE_METAL_ARGMAX": "0"},
    ),
    "prefill-layer": Profile(
        "prefill-layer",
        "Disable all-layer prefill command-buffer fusion.",
        {"TINYENGINE_PREFILL_ALL_LAYERS": "0"},
    ),
    "decode-layer": Profile(
        "decode-layer",
        "Disable all-layer decode command-buffer fusion.",
        {"TINYENGINE_DECODE_ALL_LAYERS": "0"},
    ),
    "qkv-off": Profile(
        "qkv-off",
        "Disable fused Metal QKV projection dispatch.",
        {"TINYENGINE_METAL_QKV": "0"},
    ),
}


def existing_file(path: str) -> Path:
    value = Path(path)
    if not value.is_file():
        raise argparse.ArgumentTypeError(f"not a file: {path}")
    return value


def split_names(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def slug(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "-" for ch in value).strip("-")


def objective_ms(candidate: dict[str, Any]) -> float:
    value = candidate.get("summary", {}).get("tinyengine_median_generate_ms")
    return float(value) if isinstance(value, (int, float)) else float("inf")


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def benchmark_env(profile: Profile) -> dict[str, str]:
    env = os.environ.copy()
    for key in SCRUB_ENV_KEYS:
        env.pop(key, None)
    env.update(profile.env)
    env["TINYENGINE_AUTOTUNE_PROFILE"] = profile.name
    return env


def workload_env(profile: Profile, workload: Workload) -> dict[str, str]:
    env = benchmark_env(profile)
    env["TINYENGINE_WORKLOAD"] = "long" if workload.name == "long" else "short"
    return env


def run_candidate(
    benchmark_script: Path,
    args: argparse.Namespace,
    run_dir: Path,
    workload: Workload,
    profile: Profile,
) -> dict[str, Any]:
    out_path = run_dir / f"{slug(workload.name)}__{slug(profile.name)}.json"
    command = [
        sys.executable,
        str(benchmark_script),
        "--llama-bin",
        str(args.llama_bin),
        "--gguf",
        str(args.gguf),
        "--prompt",
        workload.prompt,
        "--c-max-tokens",
        str(workload.c_max_tokens),
        "--llama-predict",
        str(workload.llama_predict),
        "--seed",
        str(args.seed),
        "--gpu-layers",
        str(args.gpu_layers),
        "--ctx-size",
        str(workload.ctx_size),
        "--runs",
        str(args.runs),
        "--out",
        str(out_path),
    ]
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=workload_env(profile, workload),
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    data = load_json(out_path)
    summary = data.get("summary", {}) if data is not None else {}
    all_text_matches = bool(data.get("all_text_matches", False)) if data is not None else False
    valid = completed.returncode == 0 and all_text_matches
    entry: dict[str, Any] = {
        "workload": workload.name,
        "profile": profile.name,
        "description": profile.description,
        "env": profile.env,
        "benchmark_json": str(out_path),
        "returncode": completed.returncode,
        "valid": valid,
        "all_text_matches": all_text_matches,
        "wall_ms": elapsed_ms,
        "summary": summary,
        "stdout": completed.stdout.strip(),
        "stderr_tail": completed.stderr[-4000:].strip(),
    }
    if data is not None:
        entry["settings"] = data.get("settings", {})
        entry["host"] = data.get("host", {})
    else:
        entry["error"] = "benchmark JSON was not written or could not be parsed"
    return entry


def choose_winner(
    entries: list[dict[str, Any]],
    min_improvement: float,
) -> tuple[dict[str, Any] | None, str]:
    valid_entries = [entry for entry in entries if entry["valid"]]
    if not valid_entries:
        return None, "no valid profile passed correctness"

    best = min(valid_entries, key=objective_ms)
    baseline = next((entry for entry in valid_entries if entry["profile"] == "default"), None)
    if baseline is None or best["profile"] == "default":
        return best, "fastest valid profile"

    baseline_ms = objective_ms(baseline)
    best_ms = objective_ms(best)
    improvement = (baseline_ms - best_ms) / baseline_ms if baseline_ms > 0.0 else 0.0
    if improvement < min_improvement:
        return baseline, (
            f"best profile {best['profile']} improved {improvement:.2%}, "
            f"below min_improvement {min_improvement:.2%}"
        )
    return best, f"fastest valid profile improved {improvement:.2%} over default"


def write_env_file(path: Path, workload: str, winner: dict[str, Any]) -> None:
    profile = winner["profile"]
    env = dict(winner["env"])
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Generated by c/tests/autotune_microkernels.py.",
        "# Source this file to reproduce the selected TinyEngine microkernel profile.",
        f"export TINYENGINE_AUTOTUNE_WORKLOAD={shlex.quote(workload)}",
        f"export TINYENGINE_AUTOTUNE_PROFILE={shlex.quote(profile)}",
    ]
    for key in TUNING_ENV_KEYS:
        lines.append(f"unset {key}")
    for key in sorted(env):
        lines.append(f"export {key}={shlex.quote(env[key])}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_workloads(args: argparse.Namespace) -> dict[str, Workload]:
    return {
        "short": Workload(
            "short",
            args.short_prompt,
            args.short_ctx_size,
            args.short_c_max_tokens,
            args.short_llama_predict,
        ),
        "long": Workload(
            "long",
            args.long_prompt,
            args.long_ctx_size,
            args.long_c_max_tokens,
            args.long_llama_predict,
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Autotune TinyEngine Metal microkernel strategy profiles for local hardware/workloads."
    )
    parser.add_argument("--llama-bin", type=existing_file, required=True)
    parser.add_argument("--gguf", type=existing_file, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("../benchmarks/autotune"))
    parser.add_argument("--profiles", default="default,matmul-off,gateup-matmul-off,kv-matmul-on,kv-matmul-off,q8-llama-off,flash-attn-off,flash-longtile-off,ffn-half-off,ffn-down-add-off,ffn-down-add-on,ffn-gate-half-on,argmax-cpu,prefill-layer,decode-layer,qkv-off")
    parser.add_argument("--workloads", default="short,long")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-layers", default="999")
    parser.add_argument("--min-improvement", type=float, default=0.02)
    parser.add_argument("--cooldown-sec", type=float, default=0.0)
    parser.add_argument("--short-prompt", default=SHORT_PROMPT)
    parser.add_argument("--short-ctx-size", type=int, default=512)
    parser.add_argument("--short-c-max-tokens", type=int, default=4)
    parser.add_argument("--short-llama-predict", type=int, default=4)
    parser.add_argument("--long-prompt", default=LONG_PROMPT)
    parser.add_argument("--long-ctx-size", type=int, default=1024)
    parser.add_argument("--long-c-max-tokens", type=int, default=4)
    parser.add_argument("--long-llama-predict", type=int, default=4)
    parser.add_argument("--list-profiles", action="store_true")
    args = parser.parse_args()

    if args.list_profiles:
        for profile in PROFILES.values():
            print(f"{profile.name}: {profile.description}")
        return 0
    if args.runs <= 0:
        parser.error("--runs must be greater than zero")
    if args.min_improvement < 0.0:
        parser.error("--min-improvement must be non-negative")
    if args.cooldown_sec < 0.0:
        parser.error("--cooldown-sec must be non-negative")

    profile_names = split_names(args.profiles)
    workload_names = split_names(args.workloads)
    unknown_profiles = [name for name in profile_names if name not in PROFILES]
    if unknown_profiles:
        parser.error(f"unknown profiles: {', '.join(unknown_profiles)}")

    workloads = make_workloads(args)
    unknown_workloads = [name for name in workload_names if name not in workloads]
    if unknown_workloads:
        parser.error(f"unknown workloads: {', '.join(unknown_workloads)}")

    benchmark_script = Path(__file__).with_name("benchmark_compare.py")
    run_id = time.strftime("%Y%m%d-%H%M%S")
    run_dir = args.out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    profiles = [PROFILES[name] for name in profile_names]
    selected_workloads = [workloads[name] for name in workload_names]
    results: dict[str, list[dict[str, Any]]] = {workload.name: [] for workload in selected_workloads}
    selection: dict[str, Any] = {}
    host: dict[str, Any] | None = None

    for workload in selected_workloads:
        print(f"autotune-workload-start name={workload.name} profiles={len(profiles)} runs={args.runs}")
        for profile in profiles:
            print(f"autotune-profile-start workload={workload.name} profile={profile.name}")
            entry = run_candidate(benchmark_script, args, run_dir, workload, profile)
            results[workload.name].append(entry)
            if host is None and entry.get("host"):
                host = entry["host"]
            objective = objective_ms(entry)
            status = "valid" if entry["valid"] else "invalid"
            print(
                f"autotune-profile-done workload={workload.name} profile={profile.name} "
                f"status={status} tiny_generate_ms={objective:.2f} out={entry['benchmark_json']}"
            )
            if args.cooldown_sec > 0.0:
                time.sleep(args.cooldown_sec)

        winner, reason = choose_winner(results[workload.name], args.min_improvement)
        if winner is not None:
            selection[workload.name] = {
                "profile": winner["profile"],
                "env": winner["env"],
                "benchmark_json": winner["benchmark_json"],
                "tinyengine_median_generate_ms": objective_ms(winner),
                "reason": reason,
            }
            env_path = args.out_dir / f"selected-{workload.name}.env"
            write_env_file(env_path, workload.name, winner)
            selection[workload.name]["env_file"] = str(env_path)
            print(f"autotune-selected workload={workload.name} profile={winner['profile']} env_file={env_path}")
        else:
            selection[workload.name] = {"profile": None, "reason": reason}
            print(f"autotune-selected workload={workload.name} profile=none reason={reason}", file=sys.stderr)

    matrix_path = args.out_dir / f"autotune-{run_id}.json"
    summary_path = args.out_dir / "selected-summary.json"
    matrix = {
        "autotune": "tinyengine-metal-microkernels",
        "run_id": run_id,
        "settings": {
            "runs": args.runs,
            "profiles": profile_names,
            "workloads": workload_names,
            "min_improvement": args.min_improvement,
            "cooldown_sec": args.cooldown_sec,
        },
        "host": host
        or {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "profiles": [asdict(profile) for profile in profiles],
        "workloads": [asdict(workload) for workload in selected_workloads],
        "results": results,
        "selection": selection,
    }
    matrix_path.write_text(json.dumps(matrix, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary_path.write_text(
        json.dumps({"matrix": str(matrix_path), "selection": selection}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"autotune-matrix out={matrix_path}")
    print(f"autotune-summary out={summary_path}")

    missing = [workload for workload, winner in selection.items() if winner.get("profile") is None]
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
