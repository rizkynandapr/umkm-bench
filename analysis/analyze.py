"""Turn downloaded Kaggle run outputs into the tables/chart for the DEV post.

1. Download results:   kaggle b t download <task-slug> -s -o results/   (for each of the 3 tasks)
   (or paste each notebook's UMKM_BENCH_SUMMARY=... line into results/<model>/<variant>.txt)
2. Run:                python analysis/analyze.py results/
3. Output:             analysis/report.md  +  analysis/slang_tax.png
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

MARKER = re.compile(r"UMKM_BENCH_SUMMARY=(\{.*\})")
VARIANTS = ("formal", "slang", "noisy")
OUT = Path(__file__).parent


def _texts(path: Path):
    raw = path.read_text(encoding="utf-8", errors="ignore")
    if path.suffix == ".ipynb":
        try:
            nb = json.loads(raw)
            for cell in nb.get("cells", []):
                for out in cell.get("outputs", []):
                    yield "".join(out.get("text", []))
            return
        except json.JSONDecodeError:
            pass
    yield raw  # our own results/<model>/<variant>.txt files are plain JSON lines
    yield raw.replace("\\n", "\n").replace('\\"', '"')  # escaped copies pasted from Kaggle logs


def _model_name(path: Path, root: Path) -> str:
    parts = path.relative_to(root).parts
    # kaggle layout: <task>/<version>/<model>/<run_id>/file ; manual layout: <model>/<variant>.txt
    return parts[2] if len(parts) >= 5 else parts[0] if len(parts) > 1 else path.stem


def collect(root: Path) -> dict:
    data = defaultdict(dict)  # model -> variant -> summary
    from ingest import expand  # compact results/<model>/<variant>.json files
    for path in root.rglob("*.json"):
        try:
            compact = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(compact, dict) and isinstance(compact.get("items"), str):
            s = expand(compact)
            data[_model_name(path, root)].setdefault(s["variant"], s)
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix not in (".ipynb", ".txt", ".log"):
            continue
        for text in _texts(path):
            for m in MARKER.finditer(text):
                try:
                    s = json.loads(m.group(1))
                except json.JSONDecodeError:
                    continue
                data[_model_name(path, root)].setdefault(s["variant"], s)
    return data


def pct(x):
    return "–" if x is None else f"{100 * x:.1f}"


def report(data: dict) -> str:
    models = sorted(data, key=lambda m: -data[m].get("formal", {}).get("score", 0))
    lines = ["# UMKM-Bench — results", "", "## Overall score (0–100)", "",
             "| model | formal | slang | noisy | slang tax | noise tax |", "|---|---|---|---|---|---|"]
    for m in models:
        s = {v: data[m].get(v, {}).get("score") for v in VARIANTS}
        tax1 = None if None in (s["formal"], s["slang"]) else s["formal"] - s["slang"]
        tax2 = None if None in (s["slang"], s["noisy"]) else s["slang"] - s["noisy"]
        lines.append(f"| {m} | {pct(s['formal'])} | {pct(s['slang'])} | {pct(s['noisy'])} "
                     f"| {pct(tax1)} | {pct(tax2)} |")

    for metric, title in (("base_score", "Everyday questions (60)"),
                          ("hard_score", "Hard traps (24): near-miss cities, arithmetic, sycophancy, injection, policy edges"),
                          ("understanding", "Understanding (intent/SKU/qty/city)"),
                          ("grounding", "Grounding (knows what it doesn't know)"),
                          ("hallucination_rate", "Hallucination rate on trap questions (lower is better)"),
                          ("json_parse_rate", "Valid JSON rate")):
        lines += ["", f"## {title}", "", "| model | formal | slang | noisy |", "|---|---|---|---|"]
        for m in models:
            vals = [pct(data[m].get(v, {}).get(metric)) for v in VARIANTS]
            lines.append(f"| {m} | " + " | ".join(vals) + " |")

    # tag breakdown on the slang register (where most of the interesting failures are)
    tags = sorted({t for m in models for t in data[m].get("slang", {}).get("by_tag", {})})
    if tags:
        lines += ["", "## Slang register — score by tag", "",
                  "| tag | " + " | ".join(models) + " |", "|---" * (len(models) + 1) + "|"]
        for t in tags:
            lines.append(f"| {t} | " + " | ".join(
                pct(data[m].get("slang", {}).get("by_tag", {}).get(t)) for m in models) + " |")

    lines += extra_sections(data, models)

    lines += ["", "## Hallucination examples (invented numbers)", ""]
    for m in models:
        for v in VARIANTS:
            for it in data[m].get(v, {}).get("items", []):
                if it.get("hallucinated") and it.get("invented_numbers"):
                    reply = str(it.get("reply", "")).replace("\n", " ")[:220]
                    lines.append(f"- **{m}** / {v} / `{it['id']}` → invented {it['invented_numbers']}: “{reply}”")
    return "\n".join(lines) + "\n"


REASONING_TAGS = {"arithmetic", "stock_math", "reasoning", "reasoning_time", "unit_word", "partial_stock"}
SURFACE_TAGS = {"city_abbrev", "city_alias", "slang_color", "regional_javanese", "regional_sundanese", "code_mix_en"}


def _item_map(summary: dict) -> dict:
    return {it["id"]: it for it in summary.get("items", [])}


def _group_mean(summary: dict, tagset: set, tags_by_id: dict):
    xs = [it["score"] for it in summary.get("items", []) if tagset & set(tags_by_id.get(it["id"], []))]
    return sum(xs) / len(xs) if xs else None


def _tags_by_id() -> dict:
    root = Path(__file__).resolve().parents[1] / "data"
    items = sum((json.loads((root / n).read_text(encoding="utf-8")) for n in ("items.json", "items_hard.json")), [])
    return {it["id"]: it["tags"] for it in items}


def extra_sections(data: dict, models: list) -> list:
    lines = []
    tags_by_id = _tags_by_id()

    # A1-A4: how the replies look, not just whether they are right
    diag_cols = [("over_deferral_rate", "over-defer", pct), ("md_any_rate", "markdown", pct),
                 ("bullet_reply_rate", "bullets", pct), ("avg_reply_chars", "chars", str),
                 ("avg_emoji", "emoji", str), ("kak_rate", "kak", pct), ("anda_rate", "Anda", pct),
                 ("mirror_regional_rate", "mirrors jv/sd", pct), ("total_cost_usd", "cost $", str),
                 ("avg_latency_ms", "latency ms", str)]
    for v in VARIANTS:
        if not any(data[m].get(v, {}).get("diag") for m in models):
            continue
        lines += ["", f"## Reply diagnostics ({v})", "",
                  "| model | " + " | ".join(c[1] for c in diag_cols) + " |", "|---" * (len(diag_cols) + 1) + "|"]
        for m in models:
            d = data[m].get(v, {}).get("diag") or {}
            lines.append(f"| {m} | " + " | ".join(("–" if d.get(k) is None else f(d.get(k))) for k, _, f in diag_cols) + " |")

    # B5: same trap items with and without the anti-guessing instruction
    if any("noguard" in data[m] for m in models):
        lines += ["", "## Ablation: remove the 'don't guess' instruction (same slang trap items)", "",
                  "| model | halluc. with guard | halluc. without | made-up facts with | made-up facts without | score with | score without |",
                  "|---|---|---|---|---|---|---|"]
        for m in models:
            ng, sl = data[m].get("noguard"), data[m].get("slang")
            if not ng or not sl:
                continue
            ids = set(_item_map(ng))
            sl_items = [it for it in sl.get("items", []) if it["id"] in ids]
            ng_items = list(_item_map(ng).values())
            h = lambda its: sum(bool(i.get("hallucinated")) for i in its) / len(its) if its else None
            sc = lambda its: sum(i["score"] for i in its) / len(its) if its else None
            # strict: the reply itself states an invented number or the exact wrong thing a trap targets
            strict = lambda its: (sum(bool(i.get("invented_numbers")) or (i.get("fields") or {}).get("no_forbidden") == 0
                                      for i in its) / len(its)) if its else None
            lines.append(f"| {m} | {pct(h(sl_items))} | {pct(h(ng_items))} | {pct(strict(sl_items))} | "
                         f"{pct(strict(ng_items))} | {pct(sc(sl_items))} | {pct(sc(ng_items))} |")

    # C6: is the noise tax about reading (surface) or about thinking (reasoning)?
    lines += ["", "## Where the register tax lands: surface-reading items vs reasoning items", "",
              "| model | surface F | surface S | surface N | reasoning F | reasoning S | reasoning N |",
              "|---|---|---|---|---|---|---|"]
    for m in models:
        row = [pct(_group_mean(data[m].get(v, {}), SURFACE_TAGS, tags_by_id)) for v in VARIANTS]
        row += [pct(_group_mean(data[m].get(v, {}), REASONING_TAGS, tags_by_id)) for v in VARIANTS]
        lines.append(f"| {m} | " + " | ".join(row) + " |")

    lines += ["", "## Items that break only when typos are added (slang ok, noisy wrong)", ""]
    counts = []
    for m in models:
        sl, no = _item_map(data[m].get("slang", {})), _item_map(data[m].get("noisy", {}))
        n = sum(1 for i in set(sl) & set(no) if sl[i]["score"] == 1.0 and no[i]["score"] < 1.0)
        if sl and no:
            counts.append(f"{m}: {n}")
    lines += ["Count per model: " + ", ".join(counts), ""]
    for m in models:
        sl, no = _item_map(data[m].get("slang", {})), _item_map(data[m].get("noisy", {}))
        for iid in sorted(set(sl) & set(no)):
            if sl[iid]["score"] == 1.0 and no[iid]["score"] < 1.0:
                broke = [k for k, v in (no[iid].get("fields") or {}).items() if v < 1.0]
                lines.append(f"- **{m}** `{iid}` ({', '.join(tags_by_id.get(iid, []))}) broke: {', '.join(broke) or 'score ' + str(no[iid]['score'])}")
    return lines


def chart(data: dict) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping chart")
        return
    models = sorted(data, key=lambda m: -data[m].get("formal", {}).get("score", 0))
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(models) + 1.5))
    colors = {"formal": "#2a6fdb", "slang": "#e8871e", "noisy": "#c0392b"}
    for i, m in enumerate(models):
        xs = [100 * data[m][v]["score"] for v in VARIANTS if v in data[m]]
        ax.plot(xs, [i] * len(xs), color="#bbbbbb", lw=2, zorder=1)
        for v in VARIANTS:
            if v in data[m]:
                ax.scatter(100 * data[m][v]["score"], i, color=colors[v], s=70, zorder=2,
                           label=v if i == 0 else None)
    ax.set_yticks(range(len(models)), models)
    ax.invert_yaxis()
    ax.set_xlabel("score (0–100)")
    ax.set_title("Same 60 questions, three ways of typing them")
    ax.legend(loc="lower left", frameon=False, ncol=3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "slang_tax.png", dpi=160)
    print(f"wrote {OUT / 'slang_tax.png'}")


if __name__ == "__main__":
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
    data = collect(root)
    if not data:
        sys.exit(f"No UMKM_BENCH_SUMMARY lines found under {root}/")
    (OUT / "report.md").write_text(report(data), encoding="utf-8")
    print(f"wrote {OUT / 'report.md'} ({len(data)} models)")
    # the charts for the post live in charts.py
