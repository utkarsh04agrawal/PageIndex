#!/usr/bin/env python3
"""Measure PageIndex Flash time and cost, with and without --optimize.

Extraction runs once per document (deterministic, no LLM) and is reused by both
variants, so the numbers isolate what --optimize actually adds.

Instrumentation is done by wrapping the OpenAI async client and two internal
functions. No pageindex source is modified.

    python tools/bench/bench_flash.py                    # all example documents
    python tools/bench/bench_flash.py path/to/doc.pdf    # your own
    python tools/bench/report.py                         # print the tables

Results land in one JSON per document under --out and are skipped on re-run, so
an interrupted sweep resumes where it stopped. Pass --force to redo them.
"""
import argparse
import asyncio
import copy
import glob
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

RECORDS = []
STAGE = ["init"]
TIMERS = {}


# ------------------------------------------------------------- instrument
import openai

_real_client = openai.AsyncOpenAI


def _kind_of(messages):
    """Classify a call by prompt prefix; the three templates are distinct."""
    text = (messages[0].get("content") or "")[:80]
    if text.startswith("You are given a text chunk"):
        return "summary_leaf"
    if text.startswith("You are given a section of a document"):
        return "summary_parent"
    return "expand"


def _instrumented(*a, **k):
    client = _real_client(*a, **k)
    orig = client.chat.completions.create

    async def create(**kw):
        t0 = time.perf_counter()
        stage, kind = STAGE[0], _kind_of(kw.get("messages") or [{}])
        try:
            r = await orig(**kw)
        except Exception as exc:
            RECORDS.append({"stage": stage, "kind": kind, "error": repr(exc)[:160],
                            "latency": time.perf_counter() - t0})
            raise
        u = r.usage
        RECORDS.append({
            "stage": stage, "kind": kind,
            "prompt": u.prompt_tokens, "completion": u.completion_tokens,
            "cached": getattr(getattr(u, "prompt_tokens_details", None), "cached_tokens", 0) or 0,
            "reasoning": getattr(getattr(u, "completion_tokens_details", None),
                                 "reasoning_tokens", 0) or 0,
            "latency": time.perf_counter() - t0,
            "empty": not (r.choices[0].message.content or "").strip(),
        })
        return r

    client.chat.completions.create = create
    return client


openai.AsyncOpenAI = _instrumented

import pageindex.flash.main as flash_main
import pageindex.tree_optimize as tree_optimize
from pageindex.flash.api import _merge, _optimize
from pageindex.utils import ConfigLoader, summarize_tree


def _time_into(key, fn):
    """Wrap fn so its cumulative wall time lands in TIMERS[key]."""
    def wrapped(*a, **k):
        t0 = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            TIMERS[key] = TIMERS.get(key, 0.0) + time.perf_counter() - t0
    return wrapped


flash_main.parse_charlevel_meta_parallel = _time_into(
    "pdf_parse", flash_main.parse_charlevel_meta_parallel)
tree_optimize.merge = _time_into("merge", tree_optimize.merge)


# ------------------------------------------------------------------ helpers
def price_for(model):
    """Per-token USD from litellm's cost map, so this stays right as models change."""
    import litellm
    entry = litellm.model_cost.get(model) or litellm.model_cost.get(f"openai/{model}")
    if not entry:
        raise SystemExit(
            f"No pricing found for model {model!r}. Pass --price-in/--price-out "
            f"(USD per million tokens) explicitly.")
    return {"input": entry["input_cost_per_token"],
            "output": entry["output_cost_per_token"],
            "cached": entry.get("cache_read_input_token_cost",
                                entry["input_cost_per_token"])}


def preflight(model):
    """Spend one token verifying the key actually works.

    llm_acompletion swallows auth and config errors, retries ten times, then
    returns "". Without this check a bad key burns the whole sweep printing
    `Retrying` and lands on a table of empty summaries and zero cost.
    """
    import openai
    try:
        reply = openai.OpenAI(max_retries=0).chat.completions.create(
            model=model, messages=[{"role": "user", "content": "Reply with the word OK"}])
    except Exception as exc:
        raise SystemExit(f"Preflight call to {model!r} failed, aborting before the sweep:\n"
                         f"  {type(exc).__name__}: {exc}")
    if not (reply.choices[0].message.content or "").strip():
        raise SystemExit(f"Preflight call to {model!r} returned an empty reply, aborting.")


def tree_stats(structure):
    n = {"nodes": 0, "leaves": 0, "max_depth": 0, "key_items": 0}

    def walk(ns, d):
        for node in ns:
            n["nodes"] += 1
            n["max_depth"] = max(n["max_depth"], d)
            n["key_items"] += len(node.get("key_items") or [])
            kids = node.get("nodes") or []
            if not kids:
                n["leaves"] += 1
            walk(kids, d + 1)

    walk(structure, 1)
    return n


def usage(stage=None, kind=None):
    rs = [r for r in RECORDS
          if (stage is None or r["stage"] == stage) and (kind is None or r["kind"] == kind)]
    ok = [r for r in rs if "error" not in r]
    total = lambda f: sum(r[f] for r in ok)
    return {"calls": len(rs), "errors": len(rs) - len(ok),
            "empty": sum(1 for r in ok if r["empty"]),
            "in": total("prompt"), "cached": total("cached"),
            "out": total("completion"), "reasoning": total("reasoning"),
            "llm_wall": round(sum(r["latency"] for r in rs), 1),
            "slowest_call": round(max((r["latency"] for r in rs), default=0), 1)}


# --------------------------------------------------------------------- run
def run_one(pdf, model, variant, out_dir, save_trees):
    RECORDS.clear()
    TIMERS.clear()
    rec = {"doc": os.path.basename(pdf), "model": model, "variant": variant}

    t0 = time.perf_counter()
    result = flash_main.extract_toc(pdf)
    t_extract = time.perf_counter() - t0
    base = result.get("structure", [])
    page_texts = result.get("page_texts") or []
    t_parse = TIMERS.get("pdf_parse", 0.0)

    rec["pages"] = len(page_texts)
    rec["raw_tree"] = tree_stats(base)
    rec["time"] = {"pdf_parse": round(t_parse, 1),
                   "layout_outline": round(t_extract - t_parse, 1)}

    if not base:
        rec["empty_structure"] = True
        rec["time"]["total_optimize"] = rec["time"]["total_baseline"] = round(t_extract, 1)
        return rec

    page_list = [(text, 0) for text in page_texts]
    struct_a = struct_b = None

    if variant in ("both", "optimize"):
        struct_a = copy.deepcopy(base)
        STAGE[0] = "optimize"
        t0 = time.perf_counter()
        info = _optimize(struct_a, page_texts, True, model)
        t_opt = time.perf_counter() - t0
        t_merge = TIMERS.get("merge", 0.0)

        STAGE[0] = "summary_opt"
        t0 = time.perf_counter()
        asyncio.run(summarize_tree(struct_a, page_list, model=model))
        t_sum_a = time.perf_counter() - t0

        rec["time"].update({
            "merge": round(t_merge, 2), "expand": round(t_opt - t_merge, 1),
            "summary_optimized": round(t_sum_a, 1),
            "total_optimize": round(t_extract + t_opt + t_sum_a, 1)})
        rec["optimize"] = info

    if variant in ("both", "baseline"):
        struct_b = copy.deepcopy(base)
        t0 = time.perf_counter()
        _merge(struct_b)
        t_merge_b = time.perf_counter() - t0

        STAGE[0] = "summary_base"
        t0 = time.perf_counter()
        asyncio.run(summarize_tree(struct_b, page_list, model=model))
        t_sum_b = time.perf_counter() - t0

        rec["time"].update({
            "merge_baseline": round(t_merge_b, 2),
            "summary_baseline": round(t_sum_b, 1),
            "total_baseline": round(t_extract + t_merge_b + t_sum_b, 1)})

    rec["usage"] = {
        "expand": usage("optimize"),
        "summary_opt_leaf": usage("summary_opt", "summary_leaf"),
        "summary_opt_parent": usage("summary_opt", "summary_parent"),
        "summary_opt": usage("summary_opt"),
        "summary_baseline": usage("summary_base"),
    }
    rec["tree"] = {"optimized": tree_stats(struct_a) if struct_a else None,
                   "baseline": tree_stats(struct_b) if struct_b else None}
    rec["errors"] = [r for r in RECORDS if "error" in r]

    if save_trees:
        stem = os.path.splitext(os.path.basename(pdf))[0]
        for tag, struct in (("optimized", struct_a), ("baseline", struct_b)):
            if struct is None:
                continue
            payload = {k: v for k, v in result.items() if k != "page_texts"}
            payload["structure"] = struct
            with open(f"{out_dir}/{stem}.tree-{tag}.json", "w") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
    return rec


def main():
    ap = argparse.ArgumentParser(
        description="Benchmark PageIndex Flash time and cost, with and without --optimize.")
    ap.add_argument("pdfs", nargs="*",
                    help="PDF paths; default is every file in examples/documents/")
    ap.add_argument("--out", default=str(REPO / "results" / "bench"),
                    help="directory for the per-document JSON records")
    ap.add_argument("--model", default=None,
                    help="model for expand and summaries; default is summary_model in config.yaml")
    ap.add_argument("--variant", choices=["both", "optimize", "baseline"], default="both",
                    help="which pipelines to run; 'optimize' alone halves the spend")
    ap.add_argument("--price-in", type=float, default=None,
                    help="USD per million input tokens; default looks the model up in litellm")
    ap.add_argument("--price-out", type=float, default=None,
                    help="USD per million output tokens")
    ap.add_argument("--save-trees", action="store_true",
                    help="also write the produced trees, useful for eyeballing quality")
    ap.add_argument("--force", action="store_true", help="redo documents already recorded")
    args = ap.parse_args()

    docs = args.pdfs or sorted(glob.glob(str(REPO / "examples" / "documents" / "*.pdf")))
    if not docs:
        raise SystemExit("no PDFs to run")
    model = args.model or (lambda c: getattr(c, "summary_model", None) or c.model)(
        ConfigLoader().load())

    if args.price_in is not None and args.price_out is not None:
        price = {"input": args.price_in / 1e6, "output": args.price_out / 1e6,
                 "cached": args.price_in / 1e6}
    else:
        price = price_for(model)

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set (put it in .env at the repo root)")

    preflight(model)

    os.makedirs(args.out, exist_ok=True)
    print(f"model={model}  documents={len(docs)}  variant={args.variant}\n"
          f"price=${price['input'] * 1e6:.2f}/M in, ${price['output'] * 1e6:.2f}/M out\n"
          f"out={args.out}", flush=True)

    empties = 0
    for pdf in docs:
        stem = os.path.splitext(os.path.basename(pdf))[0]
        path = f"{args.out}/{stem}.bench.json"
        if os.path.exists(path) and not args.force:
            print(f"skip (already recorded): {stem}", flush=True)
            continue
        print(f"\n>>> {stem}", flush=True)
        t0 = time.perf_counter()
        try:
            rec = run_one(pdf, model, args.variant, args.out, args.save_trees)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            # keep the .bench.json name free so a re-run retries this document
            path = f"{args.out}/{stem}.failed.json"
            rec = {"doc": os.path.basename(pdf), "failed": repr(exc)[:300]}
        rec["wall"] = round(time.perf_counter() - t0, 1)
        rec["price"] = price
        with open(path, "w") as f:
            json.dump(rec, f, indent=2)
        empties += sum(u["empty"] for u in (rec.get("usage") or {}).values())
        print(json.dumps({k: rec.get(k) for k in ("pages", "time")}, indent=1), flush=True)

    if empties:
        print(f"\nWARNING: {empties} calls returned an empty reply. Those nodes have no "
              f"summary and their cost is understated. Check the reliability table.",
              flush=True)
    print(f"\nsweep complete. now run:  python {Path(__file__).parent / 'report.py'} "
          f"--out {args.out}", flush=True)


if __name__ == "__main__":
    main()
