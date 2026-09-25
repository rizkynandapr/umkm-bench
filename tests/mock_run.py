"""End-to-end dry run of a generated dist/ task file with a fake LLM (no API calls).

    python -m tests.mock_run dist/umkm_bench_slang.py oracle   # should score ~1.0
    python -m tests.mock_run dist/umkm_bench_slang.py sloppy   # hallucinates on purpose
"""

import json
import re
import sys
from pathlib import Path

from kaggle_benchmarks import actors
from kaggle_benchmarks.llm_messages import LLMMessage

ROOT = Path(__file__).resolve().parents[1]
ITEMS = sum((json.loads((ROOT / "data" / n).read_text(encoding="utf-8")) for n in ("items.json", "items_hard.json")), [])
PHRASES = [
    "habis", "hanya untuk Yogyakarta", "video unboxing maks 2x24 jam", "libur",
    "bisa", "tidak ada", "tidak melayani", "tukar maks 3 hari, label belum dilepas",
    "DANA atau OVO", "sampai 21.00", "dikirim ulang gratis", "H-1", "kirim foto, nanti diganti",
    "buka 07.00", "BCA", "hanya", "belum dapat", "hitam", "mocca",
]


def _rupiah(n):
    return f"Rp {n:,}".replace(",", ".")


def oracle_answer(item):
    from src.scoring import expand_pattern
    g = item["gold"]
    parts = []
    for f in g["facts"]:
        if "num" in f:
            parts.append(_rupiah(f["num"]) if f["num"] > 100 else f"sisa {f['num']} pcs")
        elif "any_num" in f:
            parts.append(_rupiah(f["any_num"][0]))
        else:
            pat = expand_pattern(f["re"])
            parts.append(next(p for p in PHRASES if re.search(pat, p, re.I)))
    reply = ("Baik kak, " + ", ".join(parts)) if parts else "Baik kak, admin cek dulu ya 🙏"
    return {
        "intent": g["intent"][0], "sku": (g["sku"] or [None])[0], "quantity": g["qty"],
        "city": g["city"], "can_answer_from_kb": g["answerable"], "reply": reply,
    }


def sloppy_answer(item, i):
    ans = oracle_answer(item)
    if i % 10 == 0:
        return "Siap kak, nanti dicek ya"  # not JSON
    if not item["gold"]["answerable"]:
        ans["can_answer_from_kb"] = True
        ans["reply"] = "Bisa kak, sekitar Rp 27.500 dan sampai 3-4 hari"
    return ans


class FakeLLM(actors.LLMChat):
    def __init__(self, mode):
        super().__init__(name=f"fake-{mode}")
        self.mode = mode

    def invoke(self, messages, tools=None, **kwargs):
        prompt = str(messages[-1].content)
        if self.mode == "flaky":
            self.seen = getattr(self, "seen", set())
            if prompt not in self.seen:
                self.seen.add(prompt)
                raise RuntimeError("429 Too Many Requests (simulated)")
        msg = prompt.split("<<<\n", 1)[1].split("\n>>>", 1)[0]
        idx, item = next((i, it) for i, it in enumerate(ITEMS)
                         if msg in (it["formal"], it["slang"]) or it["id"] in _NOISY and _NOISY[it["id"]] == msg)
        ans = oracle_answer(item) if self.mode in ("oracle", "flaky") else sloppy_answer(item, idx)
        content = ans if isinstance(ans, str) else json.dumps(ans, ensure_ascii=False)
        return LLMMessage(sender=self, content=content)


_NOISY = {}


def main(path, mode):
    sys.path.insert(0, str(ROOT))
    from build import load_items
    _NOISY.update({it["id"]: it["messages"]["noisy"] for it in load_items()})
    code = Path(path).read_text(encoding="utf-8")
    import os, tempfile
    os.chdir(tempfile.mkdtemp(prefix="umkmbench_mock_"))  # keep run files out of the repo
    code = code.replace("umkm_main.run(kbench.llm)", "RUN = umkm_main.run(FAKE)")
    import time
    time.sleep = lambda *_: None  # no real waiting in dry runs
    ns = {"FAKE": FakeLLM(mode), "__name__": "__umkm_bench__"}
    exec(compile(code, path, "exec"), ns)
    run = ns["RUN"]
    print("RESULT:", run.result, "passed:", run.passed)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
