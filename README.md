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
