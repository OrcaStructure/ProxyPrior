#!/usr/bin/env python3
"""Generate reusable fake-article event sets from a prepared real corpus."""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import os
import sys
import traceback
from pathlib import Path

from dotenv import load_dotenv

from openrouter_credits import start_credit_tracking
from news_real_fake_experiment import (
    generate_fake_set_with_progress,
    json_read,
    json_write,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate fake event sets from corpus_real_articles.json.")
    p.add_argument("--corpus-dir", default=None, help="Directory containing corpus_real_articles.json")
    p.add_argument("--corpora-dir", default="news_corpora", help="Parent directory for corpus folders (used when --corpus-dir is omitted)")
    p.add_argument("--generator-model", default="openai/gpt-5-mini", help="OpenRouter model used to generate fake articles")
    p.add_argument("--generator-temperature", type=float, default=0.8)
    p.add_argument("--generator-workers", type=int, default=12, help="Concurrent OpenRouter requests for fake generation")
    p.add_argument("--max-article-chars", type=int, default=4500, help="Max chars stored per generated article")
    p.add_argument(
        "--from-date",
        default=None,
        help="Optional override date window start used in prompts (defaults to corpus_meta.json)",
    )
    p.add_argument(
        "--to-date",
        default=None,
        help="Optional override date window end used in prompts (defaults to corpus_meta.json)",
    )
    p.add_argument("--resume", action="store_true", help="Skip event sets already present in fake_generation_rows/")
    return p.parse_args()


def append_log(path: Path, message: str) -> None:
    stamp = dt.datetime.now().isoformat(timespec="seconds")
    with path.open("a", encoding="utf-8") as f:
        f.write(f"[{stamp}] {message}\n")


def resolve_corpus_dir(corpus_dir_arg: str | None, corpora_dir: str) -> Path:
    if corpus_dir_arg:
        return Path(corpus_dir_arg)

    root = Path(corpora_dir)
    latest_pointer = root / "latest_corpus.txt"
    if latest_pointer.exists():
        corpus_id = latest_pointer.read_text(encoding="utf-8").strip()
        if corpus_id:
            candidate = root / corpus_id
            if candidate.exists():
                return candidate

    dirs = [p for p in root.iterdir() if p.is_dir()] if root.exists() else []
    if dirs:
        dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return dirs[0]

    raise FileNotFoundError(
        f"Could not resolve corpus directory. Provide --corpus-dir or create one under {root}."
    )


def main() -> int:
    load_dotenv()
    args = parse_args()

    openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
    if not openrouter_api_key:
        print("[error] Missing OPENROUTER_API_KEY in environment/.env", file=sys.stderr)
        return 1

    try:
        corpus_dir = resolve_corpus_dir(args.corpus_dir, args.corpora_dir)
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    corpus_real_path = corpus_dir / "corpus_real_articles.json"
    if not corpus_real_path.exists():
        print(f"[error] Missing {corpus_real_path}", file=sys.stderr)
        return 1
    credit_log_path = corpus_dir / "fake_generation.log"
    start_credit_tracking(
        api_key=openrouter_api_key,
        log_fn=lambda msg: append_log(credit_log_path, msg),
    )

    real_loaded = json_read(corpus_real_path)
    real_event_sets = real_loaded.get("event_sets", [])
    if not real_event_sets:
        print("[error] No real event_sets found in corpus_real_articles.json", file=sys.stderr)
        return 1

    corpus_meta_path = corpus_dir / "corpus_meta.json"
    corpus_meta = json_read(corpus_meta_path) if corpus_meta_path.exists() else {}
    from_date = args.from_date or corpus_meta.get("from_date", "2026-01-01")
    to_date = args.to_date or corpus_meta.get("to_date", "2026-03-10")

    articles_per_set = int(real_loaded.get("articles_per_set") or len(real_event_sets[0].get("articles", [])) or 10)
    rows_dir = corpus_dir / "fake_generation_rows"
    rows_dir.mkdir(parents=True, exist_ok=True)

    pending_generation: list[tuple[int, dict, str, Path, Path]] = []
    fake_event_sets: list[dict] = []

    for set_idx, real_set in enumerate(real_event_sets):
        row_id = f"fake_set_{set_idx:03d}"
        fake_path = rows_dir / f"{row_id}.json"
        fake_raw_path = rows_dir / f"{row_id}_raw.txt"
        if args.resume and fake_path.exists():
            loaded = json_read(fake_path)
            if isinstance(loaded, dict) and loaded.get("articles"):
                fake_event_sets.append(loaded)
                continue
        pending_generation.append((set_idx, real_set, row_id, fake_path, fake_raw_path))

    def generate_one_set(task: tuple[int, dict, str, Path, Path]) -> tuple[str, str, dict]:
        _, real_set, row_id, _, _ = task
        return generate_fake_set_with_progress(
            real_set=real_set,
            row_id=row_id,
            api_key=openrouter_api_key,
            model=args.generator_model,
            temperature=args.generator_temperature,
            from_date=str(from_date),
            to_date=str(to_date),
            articles_per_set=articles_per_set,
            max_article_chars=args.max_article_chars,
            reference_articles_per_set=3,
        )

    if pending_generation:
        print(
            f"Starting fake generation: {len(pending_generation)} sets, {max(1, args.generator_workers)} workers"
        )
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.generator_workers)) as executor:
            future_to_task = {executor.submit(generate_one_set, task): task for task in pending_generation}
            completed_generation = 0
            total_generation = len(future_to_task)
            for future in concurrent.futures.as_completed(future_to_task):
                _, _, row_id, fake_path, fake_raw_path = future_to_task[future]
                try:
                    _, raw, fake_set = future.result()
                    fake_raw_path.write_text(raw, encoding="utf-8")
                    json_write(fake_path, fake_set)
                    fake_event_sets.append(fake_set)
                    completed_generation += 1
                    print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] Fake sets: {completed_generation}/{total_generation}")
                except Exception as exc:
                    json_write(fake_path, {"set_id": row_id, "error": str(exc)})
                    completed_generation += 1
                    print(
                        f"[{dt.datetime.now().strftime('%H:%M:%S')}] Fake sets: {completed_generation}/{total_generation} (failed: {exc})"
                    )
                    traceback.print_exc()

    fake_event_sets.sort(key=lambda s: str(s.get("set_id", "")))
    json_write(
        corpus_dir / "corpus_fake_articles.json",
        {
            "event_queries": [s.get("event_query", "") for s in real_event_sets],
            "articles_per_set": articles_per_set,
            "event_sets": fake_event_sets,
            "articles": [a for s in fake_event_sets for a in s.get("articles", [])],
        },
    )

    updated_meta = dict(corpus_meta)
    updated_meta.update(
        {
            "from_date": str(from_date),
            "to_date": str(to_date),
            "articles_per_set": articles_per_set,
            "num_fake_sets": len(fake_event_sets),
            "num_fake": sum(len(s.get("articles", [])) for s in fake_event_sets),
            "generator_model": args.generator_model,
        }
    )
    json_write(corpus_meta_path, updated_meta)

    print(f"Corpus directory: {corpus_dir}")
    print(f"Generated {len(fake_event_sets)} fake sets x {articles_per_set} articles.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
