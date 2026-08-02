#!/usr/bin/env python3
"""Turn bench records into the chart dataset, and print the full breakdown record."""
import glob
import json
import os
import sys

OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else "results/bench-doclen"


def load(out_dir):
    recs = []
    for path in sorted(glob.glob(f"{out_dir}/*.bench.json")):
        rec = json.load(open(path))
        if rec.get("failed"):
            print(f"FAILED: {rec['doc']}: {rec['failed'][:120]}", file=sys.stderr)
            continue
        recs.append(rec)
    recs.sort(key=lambda r: r.get("pages", 0))
    return recs


def main():
    recs = load(OUT_DIR)
    if not recs:
        raise SystemExit(f"no records in {OUT_DIR}")
    price = recs[0].get("price") or {}
    model = recs[0].get("model", "?")

    def cost(u, billed):
        if not u:
            return 0.0
        cached = u["cached"] if billed else 0
        return ((u["in"] - cached) * price["input"] + cached * price["cached"]
                + u["out"] * price["output"])

    rows = []
    for rec in recs:
        usage = rec.get("usage") or {}
        # the --optimize run: expand calls + summaries of the optimized tree
        stages = {k: usage.get(k) for k in ("expand", "summary_opt")}
        tin = sum((u or {}).get("in", 0) for u in stages.values())
        tout = sum((u or {}).get("out", 0) for u in stages.values())
        tcached = sum((u or {}).get("cached", 0) for u in stages.values())
        rows.append({
            "doc": os.path.splitext(rec["doc"])[0],
            "pages": rec["pages"],
            "seconds": rec["time"].get("total_optimize", 0),
            "cost_cold": round(sum(cost(u, False) for u in stages.values()), 6),
            "cost_billed": round(sum(cost(u, True) for u in stages.values()), 6),
            "tokens_in": tin,
            "tokens_out": tout,
            "tokens_cached": tcached,
            # recorded, not charted
            "breakdown": {
                "time": rec["time"],
                "tree": rec.get("tree", {}).get("optimized"),
                "raw_tree": rec.get("raw_tree"),
                "optimize": rec.get("optimize"),
                "usage": usage,
                "errors": len(rec.get("errors") or []),
                "wall": rec.get("wall"),
            },
        })

    payload = {"model": model, "price": price, "rows": rows}
    dest = os.path.join(OUT_DIR, "figure_data.json")
    json.dump(payload, open(dest, "w"), ensure_ascii=False, indent=1)

    print(f"model={model}  docs={len(rows)}  "
          f"price=${price.get('input',0)*1e6:.2f}/M in ${price.get('output',0)*1e6:.2f}/M out")
    print(f"\n{'doc':<40}{'pages':>7}{'sec':>8}{'cost':>10}{'in tok':>11}{'out tok':>10}")
    for r in rows:
        print(f"{r['doc'][:38]:<40}{r['pages']:>7}{r['seconds']:>8.1f}"
              f"{r['cost_cold']:>10.4f}{r['tokens_in']:>11,}{r['tokens_out']:>10,}")
    print(f"\n{'TOTAL':<40}{sum(r['pages'] for r in rows):>7}"
          f"{sum(r['seconds'] for r in rows):>8.1f}"
          f"{sum(r['cost_cold'] for r in rows):>10.4f}"
          f"{sum(r['tokens_in'] for r in rows):>11,}"
          f"{sum(r['tokens_out'] for r in rows):>10,}")
    print(f"\nchart data -> {dest}")


if __name__ == "__main__":
    main()
