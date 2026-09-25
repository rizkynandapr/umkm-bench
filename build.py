"""Bundle data/ + src/ into three self-contained Kaggle task files in dist/.

Kaggle runs one .py per task, so the sources stay modular here and the upload
file is generated. Run:  python build.py
"""

import json
from pathlib import Path

from src.perturb import make_noisy

ROOT = Path(__file__).parent
REGISTERS = ("formal", "slang", "noisy")


def _is_guard_probe(it: dict) -> bool:
    g = it["gold"]
    return (not g["answerable"]) or bool(g.get("forbid")) or bool({"sycophancy", "injection"} & set(it["tags"]))


# name -> (label, message register, anti-guessing guard on?, item filter, note shown in the notebook)
VARIANTS = {
    "formal": ("Formal", "formal", True, None, "the formal version."),
    "slang": ("Slang", "slang", True, None, "the slang version."),
    "noisy": ("Noisy", "noisy", True, None, "the noisy version, so slang plus typos. The typos are the same on every run."),
    "noguard": ("NoGuard", "slang", False, _is_guard_probe,
                "the slang version, but only the 27 trap, sycophancy and injection items, and the prompt "
                "no longer tells the model not to guess. Put it next to the slang task and you can see how much "
                "of a model's honesty comes from that one line."),
}


def _strip_docstring(src: str) -> str:
    if src.startswith('"""'):
        end = src.index('"""', 3) + 3
        return src[end:].lstrip("\n")
    return src


def load_items() -> list:
    items = []
    for name in ("items.json", "items_hard.json"):
        items += json.loads((ROOT / "data" / name).read_text(encoding="utf-8"))
    for it in items:
        it["messages"] = {
            "formal": it["formal"],
            "slang": it["slang"],
            "noisy": make_noisy(it["slang"], seed=it["id"]),
        }
    return items


def main() -> None:
    stores = json.loads((ROOT / "data/stores.json").read_text(encoding="utf-8"))
    items = load_items()
    template = (ROOT / "src/task_template.py").read_text(encoding="utf-8")
    prompt_src = _strip_docstring((ROOT / "src/prompt.py").read_text(encoding="utf-8"))
    scoring_src = _strip_docstring((ROOT / "src/scoring.py").read_text(encoding="utf-8"))

    for bad in ('"""',):
        blob = json.dumps(items, ensure_ascii=False) + json.dumps(stores, ensure_ascii=False)
        assert bad not in blob, "data must not contain triple quotes"

    out_dir = ROOT / "dist"
    out_dir.mkdir(exist_ok=True)
    for variant, (label, key, guard, keep, note) in VARIANTS.items():
        v_items = [it for it in items if keep is None or keep(it)]
        code = (
            template.replace("__VARIANT_LABEL__", label)
            .replace("__VARIANT_NOTE__", note)
            .replace("__VARIANT__", variant)
            .replace("__DESC_EXTRA__", "" if guard else ", prompt without the no guessing line")
            .replace("__MESSAGE_KEY__", key)
            .replace("__GUARD__", "True" if guard else "False")
            .replace("__STORES_JSON__", json.dumps(stores, ensure_ascii=False))
            .replace("__ITEMS_JSON__", json.dumps(v_items, ensure_ascii=False))
            .replace("__PROMPT_SRC__", prompt_src)
            .replace("__SCORING_SRC__", scoring_src)
        )
        path = out_dir / f"umkm_bench_{variant}.py"
        path.write_text(code, encoding="utf-8")
        assert "__" + "VARIANT" not in code and "__GUARD__" not in code, "unfilled placeholder"
        print(f"wrote {path.relative_to(ROOT)}  ({len(code) / 1024:.1f} KB, {len(v_items)} items)")
        print(f"wrote {to_ipynb(path).relative_to(ROOT)}")

    # Human-readable preview of every message in all three registers.
    preview = ["| id | formal | slang | noisy |", "|---|---|---|---|"]
    for it in items:
        cells = [it["messages"][v].replace("\n", " ⏎ ").replace("|", "\\|") for v in REGISTERS]
        preview.append(f"| {it['id']} | " + " | ".join(cells) + " |")
    (ROOT / "data/messages_preview.md").write_text("\n".join(preview) + "\n", encoding="utf-8")
    print("wrote data/messages_preview.md")



def to_ipynb(py_path: Path) -> Path:
    """Split a '# %%' script into notebook cells (for Kaggle: File > Import Notebook)."""
    cells, cur, kind = [], [], "code"

    def flush():
        src = "\n".join(cur).strip("\n")
        if src:
            if kind == "markdown":
                src = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in src.split("\n"))
            cell = {"cell_type": kind, "metadata": {}, "source": src.splitlines(keepends=True)}
            if kind == "code":
                cell.update(execution_count=None, outputs=[])
            cells.append(cell)

    for line in py_path.read_text(encoding="utf-8").split("\n"):
        if line.startswith("# %%"):
            flush()
            cur, kind = [], ("markdown" if "[markdown]" in line else "code")
            continue
        if line.strip() == "# %choose umkm_main":
            line = "%choose umkm_main"
        cur.append(line)
    flush()
    nb = {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3",
          "language": "python"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    out = py_path.with_suffix(".ipynb")
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


if __name__ == "__main__":
    main()
