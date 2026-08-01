# Flash time and cost benchmark

Measures how long PageIndex Flash takes and what it costs, broken down by pipeline
stage, with and without `--optimize`.

Extraction runs once per document (deterministic, no LLM) and is reused by both
variants, so the difference between them is only what `--optimize` adds. Nothing
in `pageindex/` is modified: the OpenAI async client and two internal functions
are wrapped at runtime to record token usage and per-stage timing.

## Setup

`OPENAI_API_KEY` in `.env` at the repo root. The model defaults to `summary_model`
in `pageindex/config.yaml`; per-token pricing is looked up from litellm's cost map
for that model, so it stays correct as models change.

## Run

```bash
# every PDF in examples/documents/ (~1600 pages, ~20 min, roughly $2)
python tools/bench/bench_flash.py

# print the tables
python tools/bench/report.py
```

Records go to `results/bench/`, one JSON per document. Already-recorded documents
are skipped, so an interrupted sweep resumes where it stopped. `--force` redoes them.

No sample records are checked in. Run the sweep first, or `report.py` will tell you
there is nothing to report.

## Common variations

```bash
# your own documents
python tools/bench/bench_flash.py path/to/a.pdf path/to/b.pdf

# only the --optimize pipeline, halves the spend
python tools/bench/bench_flash.py --variant optimize

# a different model
python tools/bench/bench_flash.py --model gpt-5.6-terra

# a model litellm has no pricing for (USD per million tokens)
python tools/bench/bench_flash.py --model my-model --price-in 0.5 --price-out 3.0

# also write the trees, to eyeball what optimize did to them
python tools/bench/bench_flash.py --save-trees

# keep a run separate from the main record set
python tools/bench/bench_flash.py --out results/bench-experiment
python tools/bench/report.py --out results/bench-experiment
```

## Reading the output

Each table prints its own column glossary underneath. Two things worth knowing:

**Cold vs billed.** `report.py` prices every input token at the full rate by
default. The provider's automatic prompt cache leaks across runs and across
overlapping documents, so whichever variant happens to run second gets cache hits
the first one paid for, and looks cheaper than it is. `--billed` shows what was
actually charged; use it to reconcile an invoice, not to compare the two variants.

**Empty responses.** `llm_acompletion` returns `""` after exhausting retries
rather than raising, so a run can silently produce empty summaries. The
reliability table counts those separately from retried errors. A nonzero value
there invalidates the cost numbers for that document.
