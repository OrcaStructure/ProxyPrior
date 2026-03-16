# ProxyPrior

A small test harness for evaluating whether an LLM can detect a subtle sentence substitution in a Wikipedia article via OpenRouter.

## Setup

```bash
pip install python-dotenv
```

Create a `.env` file in the project root:

```bash
OPENROUTER_API_KEY=your_key_here
```

OpenRouter runners now log credit snapshots at run start and run end (including `used_this_run` delta when available) in each run log.

## What it does

1. Fetches a Wikipedia article as plain text.
2. Uses an LLM to rewrite one real sentence so it is just a bit out of place but believable.
3. Sends the corrupted article to an OpenRouter model.
4. Asks the model to identify the bad sentence.
5. Scores the model output against the inserted sentence.

## Usage

```bash
python3 wiki_substitution_test.py \
  --title "Alan Turing" \
  --model "openai/gpt-4o-mini"
```

Useful options:

- `--replacement "..."` use your own injected sentence.
- `--replacement-mode llm|template` choose LLM-generated (default) or template replacement.
- `--replacement-model "..."` model used to generate the replacement sentence (default: same as `--model`).
- `--replacement-temperature 0.7` creativity for replacement generation.
- `--seed 123` make sentence selection reproducible.
- `--max-chars 9000` control context length sent to the model.
- `--dry-run` build and display a test case without calling OpenRouter.
- `--run-id my_test_01` name a run folder.
- `--resume --run-id my_test_01` continue an interrupted run.
- `--runs-dir runs` choose where run artifacts are written (default `runs/`).

## Run logging and resume

Each run writes full artifacts into `runs/<run_id>/`:

- `args.json`
- `run.log`
- `state.json`
- `article.txt`
- `test_case.json`
- `prompt.json`
- `replacement_prompt.json`
- `openrouter_replacement_raw.txt`
- `replacement_output.json`
- `openrouter_raw.txt`
- `model_output.json`
- `evaluation.json`

If a run is interrupted, rerun with the same `--run-id` and `--resume`:

```bash
python3 wiki_substitution_test.py \
  --title "Alan Turing" \
  --model "openai/gpt-4o-mini" \
  --run-id turing_eval_01 \
  --resume
```

## Example dry run

```bash
python3 wiki_substitution_test.py --title "Alan Turing" --model "openai/gpt-4o-mini" --dry-run
```

This prints the inserted sentence and the hidden original sentence so you can verify the setup before spending API credits.

## Multi-model benchmark (same 15 substitutions)

Use `benchmark_models.py` to generate one shared 15-case substitution dataset, then evaluate multiple models against that exact same set.

Run GPT-5-mini and GPT-5 on the same 15 substitutions across 15 different articles:

```bash
python3 benchmark_models.py \
  --models openai/gpt-5-mini openai/gpt-5 \
  --num-cases 15 \
  --benchmark-id gpt5_pair_15
```

Resume the same benchmark:

```bash
python3 benchmark_models.py \
  --models openai/gpt-5-mini openai/gpt-5 \
  --num-cases 15 \
  --benchmark-id gpt5_pair_15 \
  --resume
```

Artifacts are written to `benchmarks/<benchmark_id>/`:

- `dataset_manifest.json` shared case set
- `cases/` per-case corrupted article and replacement generation outputs
- `results/<model_slug>/` per-model per-case results
- `summary.json` aggregate accuracy and mean similarity per model

## AIME25 dataset runner

The AIME runner loads the dataset with `pandas`:

```python
import pandas as pd

df = pd.read_json("hf://datasets/math-ai/aime25/test.jsonl", lines=True)
```

Run GPT-5 on it with low reasoning effort:

```bash
python3 run_aime25_gpt5.py \
  --model openai/gpt-5 \
  --reasoning-effort low \
  --run-id aime25_gpt5_low
```

Resume the same run:

```bash
python3 run_aime25_gpt5.py \
  --model openai/gpt-5 \
  --reasoning-effort low \
  --run-id aime25_gpt5_low \
  --resume
```

Quick smoke test:

```bash
python3 run_aime25_gpt5.py \
  --model openai/gpt-5 \
  --reasoning-effort low \
  --max-rows 3 \
  --run-id aime25_gpt5_low_smoke
```

## News bias + hidden-mode experiment (early 2026)

Use three scripts:
1. `prepare_news_real_corpus.py` downloads and samples reusable real event sets.
2. `generate_news_fake_corpus.py` generates reusable fake sets from those real sets.
3. `news_real_fake_experiment.py` only judges (no repeated data prep).

If you do not pass any `--event-query`, the prepare step now auto-discovers event queries from `--query` (default `world`) and then fetches related articles per discovered event.

Step 1: prepare real corpus once:

```bash
python3 prepare_news_real_corpus.py \
  --event-query "uk local council budget dispute" \
  --event-query "regional rail strike service disruption" \
  --event-query "city flood response infrastructure delays" \
  --articles-per-set 10 \
  --from-date 2026-01-01 \
  --to-date 2026-03-10 \
  --corpus-id news_bias_2026
```

Minimal auto-discovery prepare command:

```bash
python3 prepare_news_real_corpus.py
```

Step 2: generate fake corpus once:

```bash
python3 generate_news_fake_corpus.py \
  --corpus-dir news_corpora/news_bias_2026 \
  --generator-model openai/gpt-5-mini \
  --generator-workers 12
```

Step 3: run judging using prepared corpus:

```bash
python3 news_real_fake_experiment.py \
  --corpus-dir news_corpora/news_bias_2026 \
  --run-mode both \
  --prior-real 0.2 \
  --turns-per-conversation 10 \
  --judge-model openai/gpt-5 \
  --judge-workers 12 \
  --run-id news_bias_2026_judge
```

Force a specific mode:

```bash
python3 news_real_fake_experiment.py --corpus-dir news_corpora/news_bias_2026 --run-mode deploy --run-id news_bias_2026_deploy
python3 news_real_fake_experiment.py --corpus-dir news_corpora/news_bias_2026 --run-mode eval --run-id news_bias_2026_eval
```

Artifacts are written to `news_runs/<run_id>/`:

- `args.json`
- `run.log`
- `real_candidates_snapshot.json`
- `corpus_real_articles.json`
- `corpus_fake_articles.json`
- `corpus_meta.json`
- `items_to_score_*.json`
- `conversation_schedule_*.json`
- `conversations_*/`
- `rows/*_fake.json`, `rows/*_fake_raw.txt`
- `rows/*_turn_*_score.json`, `rows/*_turn_*_judge_raw.txt`
- `summary.json` and `summary_*.json`

## Bayesian diagnostics for a completed run

Compute Bayesian update diagnostics from a completed run directory:

```bash
python3 bayes_analysis.py \
  --run-dir news_runs/news_rf_early_2026_multiturn \
  --reference-prior 0.5 \
  --prior-grid 0.2,0.5,0.8 \
  --threshold 0.5 \
  --cost-fp 1.0 \
  --cost-fn 1.0
```

This writes `bayes_analysis_summary.json` into the run directory, including:

- per-turn Bayes factors (`log_bf`, `bf`)
- calibration by turn
- information gain by turn (entropy drop)
- cumulative evidence separation (real vs fake)
- position/order effects and path dependence checks
- prior sensitivity
- posterior odds accuracy
- decision utility and simple variance decomposition

## Single-problem math trace experiment

Run one hard proof-based math problem many times and store full model outputs (reasoning traces).

Create file:

- `math_problem.txt` with the full problem statement

Run Claude Sonnet 4.5 multiple times:

```bash
python3 run_math_trace_experiment.py \
  --problem-file math_problem.txt \
  --model anthropic/claude-sonnet-4.5 \
  --samples 50 \
  --reasoning-effort high \
  --temperature 0.8 \
  --workers 8 \
  --run-id claude45_mathtrace_01
```

Artifacts are written to `math_trace_runs/<run_id>/`:

- `args.json`
- `task.json`
- `prompt.json`
- `run.log`
- `rows/sample_XXXX.json` (proof trace text + `claimed_solved` flag from model)
- `rows/sample_XXXX_raw.json` (raw OpenRouter response)
- `summary.json` (completion counts + claimed-solved rate)

## Idea graph extraction across traces

Build a per-trace idea graph (ideas + prerequisite edges), then relabel equivalent ideas across traces.

If `--run-dir` is omitted, it uses the most recent folder in `math_trace_runs/`.

```bash
python3 run_idea_graph_experiment.py \
  --run-dir math_trace_runs/claude45_prooftrace_01 \
  --analysis-model openai/gpt-5-mini \
  --workers 8 \
  --resume
```

Artifacts are written to `math_trace_runs/<run_id>/idea_graphs/`:

- `idea_graph.log`
- `per_trace/sample_XXXX_graph.json`
- `per_trace/sample_XXXX_graph_raw.txt`
- `canonical_idea_graph.json` (canonical nodes + relabeled edges + mappings)

Visualize the canonical graph with hover tooltips (defaults to most recent run):

```bash
python3 visualize_idea_graph.py
```

This writes:

- `math_trace_runs/<run_id>/idea_graphs/canonical_idea_graph_viz.html`
