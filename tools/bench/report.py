#!/usr/bin/env python3
"""Print time and cost tables from the records bench_flash.py wrote.

    python tools/bench/report.py             # cold run, every input token at full price
    python tools/bench/report.py --billed    # what the provider actually charged

Costs are recomputed here from the raw token counters rather than read back from
the records, so both variants are always priced the same way. Cold is the default
because the provider prompt cache leaks across runs and across overlapping
documents, which makes whichever variant ran second look cheaper than it is.
"""
import argparse
import glob
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def load(out_dir):
    recs = []
    for path in sorted(glob.glob(f"{out_dir}/*.bench.json")):
        rec = json.load(open(path))
        if not rec.get("failed"):
            recs.append(rec)
    recs.sort(key=lambda r: r.get("pages", 0))
    return recs


def header(groups):
    """groups: [(group_label, [(column_label, width), ...]), ...] -> aligned lines."""
    top, col = "", ""
    for label, cols in groups:
        span = sum(w for _, w in cols)
        top += label.center(span) if label else " " * span
        for text, width in cols:
            col += f"{text:>{width}}"
    return top.rstrip(), col, "-" * len(col)


def banner(title, subtitle=""):
    print(f"\n{'=' * 126}\n{title}")
    if subtitle:
        print(subtitle)
    print("=" * 126)


def main():
    ap = argparse.ArgumentParser(description="Report on a bench_flash.py sweep.")
    ap.add_argument("--out", default=str(REPO / "results" / "bench"),
                    help="directory bench_flash.py wrote its records to")
    ap.add_argument("--billed", action="store_true",
                    help="price input tokens served from the prompt cache at the cache rate")
    args = ap.parse_args()

    recs = load(args.out)
    if not recs:
        raise SystemExit(f"no records in {args.out}. Run bench_flash.py first.")
    price = recs[0].get("price") or {"input": 0.0, "output": 0.0, "cached": 0.0}
    model = recs[0].get("model", "?")

    def usd(u):
        if not u:
            return 0.0
        cached = u["cached"] if args.billed else 0
        return ((u["in"] - cached) * price["input"] + cached * price["cached"]
                + u["out"] * price["output"])

    def U(rec, key):
        return (rec.get("usage") or {}).get(key)

    def name(doc):
        stem = os.path.splitext(doc)[0]
        return stem if len(stem) <= 30 else stem[:28] + ".."

    # ----------------------------------------------------------------- time
    banner("TIME", "how long each pipeline stage took, and the two end-to-end totals")
    top, col, rule = header([
        ("", [("document", 32), ("pages", 7)]),
        ("stage duration (seconds)",
         [("pdf parse", 11), ("layout + outline", 18), ("merge", 8),
          ("expand", 9), ("node summaries", 16)]),
        ("end-to-end total (seconds)", [("with --optimize", 17), ("default", 10)]),
    ])
    print(top, col, rule, sep="\n")
    tot = dict.fromkeys(("pages", "pdf_parse", "layout_outline", "merge", "expand",
                         "summary_optimized", "total_optimize", "total_baseline"), 0.0)
    for rec in recs:
        g = rec["time"].get
        print(f"{name(rec['doc']):<32}{rec['pages']:>7}{g('pdf_parse', 0):>11.1f}"
              f"{g('layout_outline', 0):>18.1f}{g('merge', 0):>8.2f}{g('expand', 0):>9.1f}"
              f"{g('summary_optimized', 0):>16.1f}{g('total_optimize', 0):>17.1f}"
              f"{g('total_baseline', 0):>10.1f}")
        tot["pages"] += rec["pages"]
        for k in tot:
            if k != "pages":
                tot[k] += g(k, 0)
    print(rule)
    print(f"{'TOTAL':<32}{int(tot['pages']):>7}{tot['pdf_parse']:>11.1f}"
          f"{tot['layout_outline']:>18.1f}{tot['merge']:>8.2f}{tot['expand']:>9.1f}"
          f"{tot['summary_optimized']:>16.1f}{tot['total_optimize']:>17.1f}"
          f"{tot['total_baseline']:>10.1f}")
    print("""
  pdf parse         PDFium character-level parse, no LLM, shared by both variants
  layout + outline  layout classification and outline assembly, no LLM, shared by both variants
  merge             deterministic tree merge, no LLM  (runs only with --optimize)
  expand            LLM pass that adds a child level to oversized nodes  (runs only with --optimize)
  node summaries    LLM summary for every node of the OPTIMIZED tree, 64 concurrent calls
  with --optimize   pdf parse + layout + merge + expand + node summaries
  default           pdf parse + layout + merge (~0s) + node summaries of the MERGED tree""")

    # ----------------------------------------------------------------- cost
    banner(f"COST  ({'as billed, prompt-cache hits included' if args.billed else 'cold run, every input token at full price'})",
           f"{model} at ${price['input'] * 1e6:.2f} per million input tokens, "
           f"${price['output'] * 1e6:.2f} per million output tokens")
    top, col, rule = header([
        ("", [("document", 32), ("pages", 7)]),
        ("cost by stage (USD)",
         [("expand", 10), ("leaf summaries", 16), ("parent summaries", 18)]),
        ("end-to-end total (USD)",
         [("with --optimize", 17), ("default", 10), ("difference", 12)]),
    ])
    print(top, col, rule, sep="\n")
    ct = dict.fromkeys(("expand", "leaf", "parent", "opt", "base"), 0.0)
    for rec in recs:
        expand = usd(U(rec, "expand"))
        leaf, parent = usd(U(rec, "summary_opt_leaf")), usd(U(rec, "summary_opt_parent"))
        opt, base = expand + usd(U(rec, "summary_opt")), usd(U(rec, "summary_baseline"))
        print(f"{name(rec['doc']):<32}{rec['pages']:>7}{expand:>10.4f}{leaf:>16.4f}"
              f"{parent:>18.4f}{opt:>17.4f}{base:>10.4f}{opt - base:>+12.4f}")
        for k, v in (("expand", expand), ("leaf", leaf), ("parent", parent),
                     ("opt", opt), ("base", base)):
            ct[k] += v
    print(rule)
    print(f"{'TOTAL':<32}{int(tot['pages']):>7}{ct['expand']:>10.4f}{ct['leaf']:>16.4f}"
          f"{ct['parent']:>18.4f}{ct['opt']:>17.4f}{ct['base']:>10.4f}"
          f"{ct['opt'] - ct['base']:>+12.4f}")
    print("""
  expand            the only LLM cost --optimize adds; merge is free and runs in both
  leaf summaries    summaries of nodes with no children
  parent summaries  summaries composed from child summaries plus uncovered pages
  difference        positive means --optimize cost more than the default run""")
    if ct["base"] and tot["total_baseline"]:
        print(f"\n  overall: {(ct['opt'] / ct['base'] - 1) * 100:+.1f}% cost, "
              f"{(tot['total_optimize'] / tot['total_baseline'] - 1) * 100:+.1f}% time. "
              f"expand is {ct['expand'] / ct['opt'] * 100:.1f}% of the --optimize run cost "
              f"but {tot['expand'] / tot['total_optimize'] * 100:.1f}% of its wall time.")

    # ----------------------------------------------------------- what it buys
    banner("WHAT --optimize BUYS",
           "search cost = how many pages an agent must read to reach the answer")
    top, col, rule = header([
        ("", [("document", 32), ("pages", 7)]),
        ("node count", [("raw tree", 10), ("after optimize", 16), ("default tree", 16)]),
        ("operations", [("merges", 9), ("expands", 9)]),
        ("worst-case pages", [("before", 9), ("after", 8)]),
        ("average pages", [("before", 9), ("after", 8)]),
    ])
    print(top, col, rule, sep="\n")
    for rec in recs:
        if rec.get("empty_structure") or not rec.get("optimize"):
            print(f"{name(rec['doc']):<32}{rec['pages']:>7}   (empty structure, no LLM work)")
            continue
        o = rec["optimize"]
        before, after = o["before"], o["after"]
        tree = rec.get("tree") or {}
        cell = lambda t: t["nodes"] if t else 0
        print(f"{name(rec['doc']):<32}{rec['pages']:>7}{rec['raw_tree']['nodes']:>10}"
              f"{cell(tree.get('optimized')):>16}{cell(tree.get('baseline')):>16}"
              f"{o['merges']:>9}{o['expands']:>9}"
              f"{before.get('worst_case_search_complexity', 0):>9}"
              f"{after.get('worst_case_search_complexity', 0):>8}"
              f"{before.get('average_search_complexity', 0):>9.1f}"
              f"{after.get('average_search_complexity', 0):>8.1f}")
    print("""
  raw tree          nodes straight out of extraction, before any merge
  worst-case pages  pages read on the most expensive path through the tree (lower is better)
  average pages     pages read on an average path (lower is better)""")

    # ---------------------------------------------------------- reliability
    banner("LLM CALLS AND RELIABILITY")
    top, col, rule = header([
        ("", [("document", 32)]),
        ("call count", [("expand", 9), ("leaf summaries", 16), ("parent summaries", 18),
                        ("default summaries", 19)]),
        ("", [("retried errors", 16), ("empty responses", 17), ("cache hit rate", 16)]),
    ])
    print(top, col, rule, sep="\n")
    for rec in recs:
        u = rec.get("usage")
        if not u:
            continue
        # summary_opt already aggregates leaf+parent; summing every key would double-count
        disjoint = ("expand", "summary_opt", "summary_baseline")
        tokens_in = sum(u[k]["in"] for k in disjoint)
        cached = sum(u[k]["cached"] for k in disjoint)
        print(f"{name(rec['doc']):<32}{u['expand']['calls']:>9}"
              f"{u['summary_opt_leaf']['calls']:>16}{u['summary_opt_parent']['calls']:>18}"
              f"{u['summary_baseline']['calls']:>19}"
              f"{sum(u[k]['errors'] for k in disjoint):>16}"
              f"{sum(u[k]['empty'] for k in disjoint):>17}"
              f"{(cached / tokens_in * 100 if tokens_in else 0):>15.0f}%")
    print("""
  default summaries  summary calls on the merged default tree, i.e. the run without --optimize
  retried errors     transient API failures (connection reset, HTTP 500) that a retry recovered
  empty responses    calls that returned an empty string, which the pipeline swallows silently
  cache hit rate     share of input tokens served from the provider prompt cache""")


if __name__ == "__main__":
    main()
