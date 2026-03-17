#!/usr/bin/env python3
"""Prepare reusable real-article event sets for the news real/fake experiment."""

from __future__ import annotations

import argparse
import datetime as dt
import os
import random
import re
import sys
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv

from news_real_fake_experiment import (
    fetch_guardian_articles,
    json_write,
    sanitize_slug,
)


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "with",
}

HARDCODED_WORLD_DISCOVERY_SEEDS = [
    {
        "web_url": "https://www.theguardian.com/global-development/2026/feb/04/death-of-nigerian-singer-highlights-crisis-of-preventable-snakebite-fatalities",
        "headline": "Death of Nigerian singer highlights crisis of preventable snakebite fatalities",
    },
    {
        "web_url": "https://www.theguardian.com/business/2026/feb/05/shell-increases-multibillion-pound-debt-pile-record-shareholder-payouts-oil-prices",
        "headline": "Shell increases multibillion-pound debt pile with record shareholder payouts amid oil-price strain",
    },
    {
        "web_url": "https://www.theguardian.com/world/2026/feb/06/japan-cherry-blossom-festival-cancelled-tourists",
        "headline": "Japan cherry blossom festival cancelled as authorities respond to tourist pressure",
    },
]


def choose_corpus_id(seed: int) -> str:
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_news_corpus_s{seed}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download and sample real news article sets into a reusable corpus directory.")
    p.add_argument("--query", default="world", help="Fallback Guardian search query")
    p.add_argument(
        "--event-query",
        action="append",
        default=[],
        help="Event-specific Guardian query. Repeat for multiple real event sets.",
    )
    p.add_argument("--from-date", default="2026-01-01", help="Guardian from-date (YYYY-MM-DD)")
    p.add_argument("--to-date", default="2026-03-10", help="Guardian to-date (YYYY-MM-DD)")
    p.add_argument("--articles-per-set", type=int, default=10, help="Number of articles sampled per event set")
    p.add_argument("--guardian-page-size", type=int, default=50, help="Page size per Guardian request (max 50)")
    p.add_argument("--guardian-max-pages", type=int, default=6, help="Max pages to scan per query")
    p.add_argument(
        "--num-events",
        type=int,
        default=3,
        help="How many event sets to prepare when auto-discovering from --query",
    )
    p.add_argument(
        "--auto-discover-events",
        action="store_true",
        help="Force event discovery from broad --query (default if --event-query is omitted).",
    )
    p.add_argument(
        "--discovery-max-candidates",
        type=int,
        default=30,
        help="Max candidate seed headlines to evaluate during auto-discovery",
    )
    p.add_argument(
        "--guardian-api-key",
        default=None,
        help="Guardian API key (optional). Falls back to GUARDIAN_API_KEY env or 'test'.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--corpora-dir", default="news_corpora", help="Directory where corpus folders are stored")
    p.add_argument("--corpus-id", default=None, help="Corpus identifier")
    p.add_argument("--resume", action="store_true", help="Reuse existing corpus directory")
    return p.parse_args()


def normalize_tokens(text: str) -> list[str]:
    tokens = re.findall(r"[A-Za-z0-9]+", text.lower())
    return [t for t in tokens if len(t) >= 3 and t not in STOPWORDS]


def build_event_query_from_headline(headline: str, max_terms: int = 6) -> str:
    tokens = normalize_tokens(headline)
    if not tokens:
        return ""
    return " ".join(tokens[:max_terms])


def append_discovery_log(path: Path, message: str) -> None:
    stamp = dt.datetime.now().isoformat(timespec="seconds")
    with path.open("a", encoding="utf-8") as f:
        f.write(f"[{stamp}] {message}\n")


def discover_event_queries(
    *,
    discovery_query: str,
    from_date: str,
    to_date: str,
    page_size: int,
    max_pages: int,
    api_key: str,
    articles_per_set: int,
    num_events: int,
    discovery_max_candidates: int,
    log_fn: Callable[[str], None] | None = None,
) -> tuple[list[str], list[dict]]:
    logger = log_fn or (lambda _: None)
    logger(
        "discovery_start "
        f"query='{discovery_query}' from={from_date} to={to_date} "
        f"page_size={page_size} max_pages={max_pages} min_articles_per_set={articles_per_set} "
        f"num_events={num_events} max_candidate_queries={discovery_max_candidates}"
    )
    if discovery_query.strip().lower() == "world":
        seed_articles = list(HARDCODED_WORLD_DISCOVERY_SEEDS)
        logger(
            "seed_source=hardcoded_world_urls "
            f"count={len(seed_articles)}"
        )
        for idx, article in enumerate(seed_articles, start=1):
            logger(
                f"seed_article idx={idx} "
                f"url='{str(article.get('web_url', '')).strip()}' "
                f"headline='{str(article.get('headline', '')).strip()}'"
            )
    else:
        seed_articles = fetch_guardian_articles(
            query=discovery_query,
            from_date=from_date,
            to_date=to_date,
            page_size=page_size,
            max_pages=max_pages,
            api_key=api_key,
        )
        logger("seed_source=guardian_search")
    logger(f"seed_fetch_complete count={len(seed_articles)}")
    if not seed_articles:
        raise ValueError(f"No discovery seed articles found for query '{discovery_query}'.")

    candidate_queries: list[str] = []
    seen: set[str] = set()
    for article_idx, article in enumerate(seed_articles, start=1):
        headline = str(article.get("headline", "")).strip()
        candidate = build_event_query_from_headline(headline)
        if not candidate:
            logger(f"candidate_skip idx={article_idx} reason=no_tokens headline='{headline[:120]}'")
            continue
        if candidate in seen:
            logger(f"candidate_skip idx={article_idx} reason=duplicate candidate='{candidate}'")
            continue
        seen.add(candidate)
        candidate_queries.append(candidate)
        logger(
            f"candidate_add idx={article_idx} candidate='{candidate}' "
            f"source_headline='{headline[:120]}'"
        )
        if len(candidate_queries) >= max(1, discovery_max_candidates):
            logger(f"candidate_cap_reached limit={max(1, discovery_max_candidates)}")
            break

    if not candidate_queries:
        raise ValueError("Could not derive any event query candidates from discovery headlines.")

    logger(f"candidate_generation_complete unique_candidates={len(candidate_queries)}")
    scored: list[dict] = []
    for candidate_idx, candidate in enumerate(candidate_queries, start=1):
        logger(
            f"candidate_eval_start idx={candidate_idx}/{len(candidate_queries)} candidate='{candidate}'"
        )
        candidates = fetch_guardian_articles(
            query=candidate,
            from_date=from_date,
            to_date=to_date,
            page_size=page_size,
            max_pages=max_pages,
            api_key=api_key,
        )
        num_candidates = len(candidates)
        is_viable = num_candidates >= articles_per_set
        logger(
            f"candidate_eval_done idx={candidate_idx}/{len(candidate_queries)} "
            f"candidate='{candidate}' num_candidates={num_candidates} "
            f"viable={str(is_viable).lower()} threshold={articles_per_set}"
        )
        scored.append(
            {
                "event_query": candidate,
                "num_candidates": num_candidates,
                "viable": is_viable,
            }
        )

    viable = [row for row in scored if int(row["num_candidates"]) >= articles_per_set]
    viable.sort(key=lambda x: int(x["num_candidates"]), reverse=True)
    selected = viable[: max(1, num_events)]
    logger(
        f"candidate_selection_summary viable={len(viable)} "
        f"requested={max(1, num_events)} selected={len(selected)}"
    )
    for rank, row in enumerate(selected, start=1):
        logger(
            f"selected_event rank={rank} query='{row['event_query']}' num_candidates={int(row['num_candidates'])}"
        )
    if len(selected) < max(1, num_events):
        raise ValueError(
            f"Only discovered {len(selected)} viable event queries (need {num_events}). "
            "Try increasing date range/pages or provide --event-query explicitly."
        )

    return [str(row["event_query"]) for row in selected], scored


def main() -> int:
    load_dotenv()
    args = parse_args()

    if args.articles_per_set < 2:
        print("[error] --articles-per-set must be >= 2", file=sys.stderr)
        return 1

    explicit_event_queries = [q.strip() for q in args.event_query if q and q.strip()]
    use_discovery = args.auto_discover_events or not explicit_event_queries
    event_queries = list(explicit_event_queries)
    guardian_api_key = args.guardian_api_key or os.environ.get("GUARDIAN_API_KEY") or "test"

    corpus_id = args.corpus_id or choose_corpus_id(args.seed)
    corpus_dir = Path(args.corpora_dir) / corpus_id
    if corpus_dir.exists() and args.corpus_id and not args.resume:
        print(f"[error] Corpus exists at {corpus_dir}. Use --resume to continue.", file=sys.stderr)
        return 1

    corpus_dir.mkdir(parents=True, exist_ok=True)
    discovery_log_path = corpus_dir / "discovery.log"

    rng = random.Random(args.seed)
    discovery_report: dict = {"used": False}
    if use_discovery:
        def discovery_log(message: str) -> None:
            print(message)
            append_discovery_log(discovery_log_path, message)

        print(f"Discovering {args.num_events} event queries from broad query: {args.query}")
        try:
            discovered, scored = discover_event_queries(
                discovery_query=args.query,
                from_date=args.from_date,
                to_date=args.to_date,
                page_size=args.guardian_page_size,
                max_pages=args.guardian_max_pages,
                api_key=guardian_api_key,
                articles_per_set=args.articles_per_set,
                num_events=max(1, args.num_events),
                discovery_max_candidates=max(1, args.discovery_max_candidates),
                log_fn=discovery_log,
            )
        except Exception as exc:
            print(f"[error] Failed discovering event queries: {exc}", file=sys.stderr)
            return 1
        event_queries = discovered
        discovery_log(f"discovery_complete selected_event_queries={event_queries}")
        discovery_report = {
            "used": True,
            "discovery_query": args.query,
            "requested_num_events": max(1, args.num_events),
            "selected_event_queries": event_queries,
            "candidates_scored": scored,
        }

    all_candidates_by_query: dict[str, list[dict]] = {}
    try:
        for query in event_queries:
            print(f"Fetching real articles for query: {query}")
            candidates = fetch_guardian_articles(
                query=query,
                from_date=args.from_date,
                to_date=args.to_date,
                page_size=args.guardian_page_size,
                max_pages=args.guardian_max_pages,
                api_key=guardian_api_key,
            )
            print(
                f"query_fetch_complete query='{query}' num_candidates={len(candidates)} "
                f"required={args.articles_per_set}"
            )
            if len(candidates) < args.articles_per_set:
                raise ValueError(
                    f"Only found {len(candidates)} candidate real articles for query '{query}', need {args.articles_per_set}."
                )
            all_candidates_by_query[query] = candidates
    except Exception as exc:
        print(f"[error] Failed loading real articles: {exc}", file=sys.stderr)
        return 1

    real_event_sets: list[dict] = []
    for set_idx, query in enumerate(event_queries):
        picked = rng.sample(all_candidates_by_query[query], args.articles_per_set)
        picked.sort(key=lambda x: str(x.get("published_at", "")))
        preview_headlines = [str(a.get("headline", ""))[:90] for a in picked[:3]]
        print(
            f"set_sampled set_id=real_set_{set_idx:03d} query='{query}' "
            f"picked={len(picked)} preview_headlines={preview_headlines}"
        )
        real_event_sets.append(
            {
                "set_id": f"real_set_{set_idx:03d}",
                "event_query": query,
                "articles": picked,
            }
        )

    json_write(
        corpus_dir / "real_candidates_snapshot.json",
        {
            "event_queries": event_queries,
            "from_date": args.from_date,
            "to_date": args.to_date,
            "articles_per_set": args.articles_per_set,
            "discovery": discovery_report,
            "num_candidates_by_query": {k: len(v) for k, v in all_candidates_by_query.items()},
            "candidates_by_query": all_candidates_by_query,
        },
    )

    json_write(
        corpus_dir / "corpus_real_articles.json",
        {
            "seed": args.seed,
            "event_queries": event_queries,
            "articles_per_set": args.articles_per_set,
            "event_sets": real_event_sets,
            "articles": [a for s in real_event_sets for a in s.get("articles", [])],
        },
    )

    json_write(
        corpus_dir / "corpus_meta.json",
        {
            "corpus_id": corpus_id,
            "corpus_slug": sanitize_slug(corpus_id),
            "query": args.query,
            "event_queries": event_queries,
            "discovery": discovery_report,
            "from_date": args.from_date,
            "to_date": args.to_date,
            "articles_per_set": args.articles_per_set,
            "num_real_sets": len(real_event_sets),
            "num_real": sum(len(s.get("articles", [])) for s in real_event_sets),
        },
    )
    # Update latest-corpus pointer so follow-up scripts can run with no args.
    (Path(args.corpora_dir) / "latest_corpus.txt").write_text(f"{corpus_id}\n", encoding="utf-8")

    print(f"Corpus ID: {corpus_id}")
    print(f"Corpus directory: {corpus_dir}")
    print(f"Prepared {len(real_event_sets)} real sets x {args.articles_per_set} articles.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
