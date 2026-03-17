#!/usr/bin/env python3
"""Run repeated proof attempts for one hard math problem and store traces."""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import json
import os
import re
import sys
import traceback
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from openrouter_credits import start_credit_tracking

OPENROUTER_API = "https://openrouter.ai/api/v1/chat/completions"


def sanitize_slug(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip()).strip("-").lower()
    return slug or "run"


def choose_run_id(model: str, samples: int) -> str:
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_{sanitize_slug(model)}_prooftrace_n{samples}"


def json_write(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def json_read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def append_log(path: Path, message: str) -> None:
    stamp = dt.datetime.now().isoformat(timespec="seconds")
    with path.open("a", encoding="utf-8") as f:
        f.write(f"[{stamp}] {message}\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run repeated proof attempts of one hard math problem.")
    p.add_argument("--problem-file", default="math_problem.txt", help="Path to text file with problem statement")
    p.add_argument("--model", default="openai/gpt-5-mini", help="OpenRouter model id")
    p.add_argument("--samples", type=int, default=50, help="How many repeated attempts")
    p.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    p.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high"],
        default="high",
        help="Reasoning effort sent to the model via OpenRouter",
    )
    p.add_argument("--workers", type=int, default=8, help="Parallel OpenRouter requests")
    p.add_argument("--runs-dir", default="math_trace_runs", help="Run artifact directory")
    p.add_argument("--run-id", default=None, help="Run identifier")
    p.add_argument("--resume", action="store_true", help="Resume existing run")
    return p.parse_args()


def build_messages(problem_text: str) -> list[dict[str, str]]:
    system = (
        "You are solving a hard proof-based math problem. "
        "Write a full proof attempt with clear steps. "
        "End with exactly one final line in this format: "
        "'Claimed-Solved: yes' or 'Claimed-Solved: no'."
    )
    user = f"Solve the following problem with a rigorous proof attempt:\n\n{problem_text}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def call_openrouter(
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    reasoning_effort: str,
) -> dict:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "reasoning": {"effort": reasoning_effort},
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        OPENROUTER_API,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://localhost",
            "X-Title": "ProxyPrior Proof Trace Runner",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read().decode("utf-8"))


def extract_text_from_response(payload: dict) -> str:
    choices = payload.get("choices", [])
    if not choices:
        raise ValueError("No choices returned by OpenRouter.")
    message = choices[0].get("message", {})
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [chunk.get("text", "") for chunk in content if isinstance(chunk, dict)]
        text = "".join(parts).strip()
        if text:
            return text
    raise ValueError("Could not parse text content from OpenRouter response.")


def extract_claimed_solved(text: str) -> str:
    m = re.search(r"(?im)^claimed-solved\s*:\s*(yes|no)\s*$", text.strip())
    if not m:
        return "unknown"
    return m.group(1).lower()


def main() -> int:
    load_dotenv()
    args = parse_args()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("[error] Missing OPENROUTER_API_KEY in environment/.env", file=sys.stderr)
        return 1

    problem_path = Path(args.problem_file)
    if not problem_path.exists():
        print(f"[error] Missing problem file: {problem_path}", file=sys.stderr)
        return 1
    problem_text = problem_path.read_text(encoding="utf-8").strip()
    if not problem_text:
        print(f"[error] Empty problem file: {problem_path}", file=sys.stderr)
        return 1

    run_id = args.run_id or choose_run_id(args.model, args.samples)
    run_dir = Path(args.runs_dir) / run_id
    rows_dir = run_dir / "rows"
    log_path = run_dir / "run.log"

    if run_dir.exists() and args.run_id and not args.resume:
        print(f"[error] Run exists at {run_dir}. Use --resume to continue.", file=sys.stderr)
        return 1

    run_dir.mkdir(parents=True, exist_ok=True)
    rows_dir.mkdir(parents=True, exist_ok=True)
    json_write(run_dir / "args.json", vars(args))
    json_write(
        run_dir / "task.json",
        {
            "problem_file": str(problem_path),
            "problem_text": problem_text,
        },
    )
    append_log(log_path, f"run_started id={run_id}")
    start_credit_tracking(
        api_key=api_key,
        log_fn=lambda msg: append_log(log_path, msg),
    )

    messages = build_messages(problem_text)
    json_write(run_dir / "prompt.json", {"messages": messages})

    pending: list[tuple[int, Path, Path]] = []
    results: list[dict] = []
    for i in range(args.samples):
        row_id = f"sample_{i:04d}"
        row_path = rows_dir / f"{row_id}.json"
        raw_path = rows_dir / f"{row_id}_raw.json"
        if args.resume and row_path.exists():
            record = json_read(row_path)
            results.append(record)
            append_log(log_path, f"skip_existing {row_id}")
            continue
        pending.append((i, row_path, raw_path))

    def run_one(task: tuple[int, Path, Path]) -> dict:
        idx, _, _ = task
        row_id = f"sample_{idx:04d}"
        payload = call_openrouter(
            api_key=api_key,
            model=args.model,
            messages=messages,
            temperature=args.temperature,
            reasoning_effort=args.reasoning_effort,
        )
        response_text = extract_text_from_response(payload)
        return {
            "row_id": row_id,
            "model": args.model,
            "temperature": args.temperature,
            "reasoning_effort": args.reasoning_effort,
            "claimed_solved": extract_claimed_solved(response_text),
            "response_text": response_text,
            "raw_payload": payload,
        }

    if pending:
        append_log(log_path, f"starting_parallel samples={len(pending)} workers={max(1, args.workers)}")
        print(
            f"[{dt.datetime.now().strftime('%H:%M:%S')}] Starting proof trace run: "
            f"{len(pending)} samples with {max(1, args.workers)} workers"
        )
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            future_to_task = {executor.submit(run_one, t): t for t in pending}
            done = 0
            total = len(future_to_task)
            for future in concurrent.futures.as_completed(future_to_task):
                idx, row_path, raw_path = future_to_task[future]
                row_id = f"sample_{idx:04d}"
                try:
                    record = future.result()
                    payload = record.pop("raw_payload")
                    json_write(raw_path, payload)
                    json_write(row_path, record)
                    results.append(record)
                    append_log(log_path, f"completed {row_id} claimed_solved={record['claimed_solved']}")
                except urllib.error.HTTPError as exc:
                    body = exc.read().decode("utf-8", errors="ignore")
                    error_record = {"row_id": row_id, "error": f"HTTP {exc.code}", "body": body}
                    json_write(row_path, error_record)
                    results.append(error_record)
                    append_log(log_path, f"http_error {row_id} code={exc.code}")
                except Exception as exc:
                    error_record = {"row_id": row_id, "error": str(exc)}
                    json_write(row_path, error_record)
                    results.append(error_record)
                    append_log(log_path, f"failed {row_id} {exc}")
                    append_log(log_path, traceback.format_exc())
                done += 1
                print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] Completed {done}/{total}")

    valid = [r for r in results if "claimed_solved" in r]
    yes_count = sum(1 for r in valid if r.get("claimed_solved") == "yes")
    no_count = sum(1 for r in valid if r.get("claimed_solved") == "no")
    unk_count = sum(1 for r in valid if r.get("claimed_solved") not in ("yes", "no"))
    summary = {
        "run_id": run_id,
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "samples_requested": args.samples,
        "samples_completed": len(valid),
        "claimed_solved_yes": yes_count,
        "claimed_solved_no": no_count,
        "claimed_solved_unknown": unk_count,
        "claimed_solved_yes_rate": (yes_count / len(valid)) if valid else None,
    }
    json_write(run_dir / "summary.json", summary)
    append_log(log_path, "run_finished")

    print(f"Run ID: {run_id}")
    print(f"Run directory: {run_dir}")
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
