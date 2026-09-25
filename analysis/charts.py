"""Charts for the DEV post, built from results/ via analyze.collect().

    python analysis/charts.py results/

Writes analysis/fig_register.png, fig_guard.png, fig_cost.png (light background, DEV renders on white).
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams["svg.fonttype"] = "none"  # keep text as text in SVG output

sys.path.insert(0, str(Path(__file__).parent))
from analyze import _item_map, collect  # noqa: E402

OUT = Path(__file__).parent
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"  # validated categorical slots 1-3
SHORT = {"Gemini 3.1 Flash-Lite Preview": "Gemini 3.1 Flash-Lite", "Qwen 3 Next 80B Instruct": "Qwen 3 Next 80B"}


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, length=0, labelsize=10)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def _fig(h):
    fig, ax = plt.subplots(figsize=(8.5, h), dpi=180)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    return fig, ax


def register_chart(data):
    models = [m for m in data if all(v in data[m] for v in ("formal", "slang", "noisy"))]
    models.sort(key=lambda m: data[m]["noisy"]["score"])
    fig, ax = _fig(0.5 * len(models) + 1.6)
    for i, m in enumerate(models):
        xs = [100 * data[m][v]["score"] for v in ("formal", "slang", "noisy")]
        ax.plot([min(xs), max(xs)], [i, i], color="#cfceca", lw=2, zorder=1, solid_capstyle="round")
        for x, c, mk, lab, dy in zip(xs, (S1, S2, S3), ("o", "s", "D"), ("formal", "slang", "slang with typos"),
                                     (0.14, 0, -0.14)):
            ax.scatter(x, i + dy, s=64, color=c, marker=mk, edgecolor=SURFACE, lw=2, zorder=3,
                       label=lab if i == 0 else None)
        ax.text(min(xs) - 0.6, i, f"{min(xs):.1f}", va="center", ha="right", fontsize=9, color=INK2)
    ax.set_yticks(range(len(models)), [SHORT.get(m, m) for m in models], color=INK)
    ax.set_xlim(72, 101)
    ax.set_xlabel("score (0 to 100)", color=INK2)
    ax.set_title("Same 84 questions, typed three ways", loc="left", color=INK, fontsize=13, pad=26)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, frameon=False, fontsize=10,
              labelcolor=INK2, handletextpad=0.3, columnspacing=1.2)
    fig.tight_layout()
    fig.savefig(OUT / "fig_register.png", facecolor=SURFACE)
    fig.savefig(OUT / "fig_register.svg", facecolor=SURFACE)
    return models


def guard_chart(data):
    rows = []
    for m in data:
        ng, sl = data[m].get("noguard"), data[m].get("slang")
        if not ng or not sl:
            continue
        ids = set(_item_map(ng))
        with_ = [i for i in sl.get("items", []) if i["id"] in ids]
        without = ng.get("items", [])
        h = lambda its: 100 * sum(bool(i.get("hallucinated")) for i in its) / len(its)
        rows.append((m, h(with_), h(without)))
    rows.sort(key=lambda r: r[2])
    fig, ax = _fig(0.5 * len(rows) + 1.6)
    bh = 0.36
    for i, (m, a, b) in enumerate(rows):
        ax.barh(i + bh / 2 + 0.01, a, height=bh, color=S1, label="with the line" if i == 0 else None)
        ax.barh(i - bh / 2 - 0.01, b, height=bh, color=S2, label="without it" if i == 0 else None)
        if a == 0 and b == 0:
            ax.text(0.3, i, "0% with and without", va="center", fontsize=9, color=INK2)
        else:
            ax.text(b + 0.4, i - bh / 2, f"{b:.0f}%", va="center", fontsize=9, color=INK2)
            ax.text(a + 0.4, i + bh / 2, f"{a:.0f}%", va="center", fontsize=9, color=INK2)
    ax.set_yticks(range(len(rows)), [SHORT.get(r[0], r[0]) for r in rows], color=INK)
    ax.set_xticks([0, 5, 10, 15, 20], ["0%", "5%", "10%", "15%", "20%"])
    ax.set_xlabel("% of the 27 trap messages answered as if the shop info had the answer", color=INK2)
    ax.set_title('What one line ("don\'t guess, say admin will check") is worth', loc="left",
                 color=INK, fontsize=13, pad=26)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=10, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(OUT / "fig_guard.png", facecolor=SURFACE)
    fig.savefig(OUT / "fig_guard.svg", facecolor=SURFACE)


def cost_chart(data):
    pts = []
    for m in data:
        n = data[m].get("noisy")
        if n and (n.get("diag") or {}).get("total_cost_usd"):
            pts.append((m, n["diag"]["total_cost_usd"] / n["n_items"] * 1000, 100 * n["score"]))
    fig, ax = _fig(4.6)
    ax.grid(axis="y", color=GRID, lw=0.8)
    for m, cost, score in pts:
        ax.scatter(cost, score, s=70, color=S1, edgecolor=SURFACE, lw=2, zorder=3)
        off = {"GPT-5.5": (-8, -14), "Claude Sonnet 5": (-8, 6)}.get(m, (7, -3))
        ax.annotate(f"{SHORT.get(m, m)}", (cost, score), xytext=off, textcoords="offset points",
                    fontsize=9, color=INK2, ha="right" if off[0] < 0 else "left")
    ax.set_xscale("log")
    ax.set_xticks([0.1, 0.3, 1, 3, 10], ["$0.10", "$0.30", "$1", "$3", "$10"])
    ax.minorticks_off()
    ax.set_xlabel("cost per 1,000 customer messages (USD, log scale)", color=INK2)
    ax.set_ylabel("score on slang with typos", color=INK2)
    ax.set_title("What each model costs, and how it handles messy text", loc="left", color=INK, fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "fig_cost.png", facecolor=SURFACE)
    fig.savefig(OUT / "fig_cost.svg", facecolor=SURFACE)


if __name__ == "__main__":
    d = collect(Path(sys.argv[1] if len(sys.argv) > 1 else "results"))
    register_chart(d)
    guard_chart(d)
    cost_chart(d)
    print("wrote fig_register.png, fig_guard.png, fig_cost.png")
