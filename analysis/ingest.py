"""Compact per-model results: results/<model>/<variant>.json.

A result file holds the run summary ("s") and one string with only the items that lost
points or carry a flag, e.g. "KL02:.5 KL07:.75h SH26:.33fo"
(h = hallucinated, f = made-up number or trap hit, o = handed an answerable question to the admin).
Every other item scored 1.0. expand() rebuilds the full summary that analyze.py expects.

    python analysis/ingest.py "<model name>" compact.json
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from build import VARIANTS, load_items  # noqa: E402

_IDS = {}


def _ids(variant: str) -> list:
    if variant not in _IDS:
        keep = VARIANTS[variant][3]
        _IDS[variant] = [it["id"] for it in load_items() if keep is None or keep(it)]
    return _IDS[variant]


def expand(compact: dict) -> dict:
    s = dict(compact["s"])
    bad = {}
    for tok in compact.get("items", "").split():
        iid, rest = tok.split(":")
        num = rest.rstrip("hfo") or "0"
        flags = rest.lstrip("0123456789.")
        bad[iid] = (float(num), flags)
    items = []
    for iid in _ids(s["variant"]):
        score, flags = bad.get(iid, (1.0, ""))
        items.append({"id": iid, "score": score, "hallucinated": "h" in flags,
                      "invented_numbers": [], "fields": {"no_forbidden": 0} if "f" in flags else {},
                      "diag": {"over_deferral": "o" in flags}})
    s["items"] = items
    return s


def ingest(model: str, compact: dict) -> Path:
    out = ROOT / "results" / model / f"{compact['s']['variant']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"s": compact["s"], "items": compact.get("items", "")}, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    return out


if __name__ == "__main__":
    print(ingest(sys.argv[1], json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))))
