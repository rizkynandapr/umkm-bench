# %% [markdown]
# # UMKM-Bench: __VARIANT_LABEL__
# A benchmark for WhatsApp customer service in Bahasa Indonesia, built around three fictional small online shops:
# a coffee roaster in Sleman, a hijab shop in Bandung and a frozen food kitchen in Semarang.
#
# There are 84 customer messages. 60 are the everyday questions an admin answers all day (price, stock, ongkir, COD).
# The other 24 are traps, like a city right next to one on the shipping list, a customer who "remembers" a discount
# that never existed, or a message that tries to rewrite the admin's instructions.
#
# Each message comes in three versions. Formal is polite standard Indonesian. Slang is how people actually type on WA,
# with some Javanese and Sundanese mixed in. Noisy is the slang version with typos added from a fixed seed.
# This notebook uses __VARIANT_NOTE__
#
# Scoring is plain Python with no LLM judge. Half the score checks whether the model understood the message
# (intent, product, quantity, city). The other half checks the reply against the shop info: are the prices and fees
# right, did it make anything up, and does it say so when it simply doesn't know.
#
# build.py generates this file. To change anything, edit src/ and data/ in the repo.

# %%
import json
import math
import re

import pandas as pd

import kaggle_benchmarks as kbench

VARIANT = "__VARIANT__"
MESSAGE_KEY = "__MESSAGE_KEY__"   # which register of each message is sent
GUARD = __GUARD__                 # False = ablation without the anti-guessing instruction
STORES = json.loads(r"""__STORES_JSON__""")
ITEMS = json.loads(r"""__ITEMS_JSON__""")
ITEM_BY_ID = {it["id"]: it for it in ITEMS}

# %% prompt builder (from src/prompt.py)
__PROMPT_SRC__

# %% scoring (from src/scoring.py)
__SCORING_SRC__

# %%
KB_TEXT = {code: render_kb(store) for code, store in STORES.items()}


import time

ERRORS = []


def _prompt_with_retry(llm, prompt: str, tries: int = 6):
    """Retry API errors (rate limits, 5xx) with backoff. Nested evaluate() forces
    max_attempts=1, so retrying has to happen here."""
    for attempt in range(tries):
        try:
            return llm.prompt(prompt)
        except Exception as e:  # noqa: BLE001 (provider errors vary)
            ERRORS.append(f"{type(e).__name__}: {str(e)[:300]}")
            if attempt == tries - 1:
                raise
            time.sleep(min(90, 8 * 2 ** attempt))


@kbench.task(name="umkm-bench-__VARIANT__-item", store_task=False)
def umkm_item(llm, item_id: str) -> dict:
    item = ITEM_BY_ID[item_id]
    store = STORES[item["store"]]
    raw = _prompt_with_retry(llm, build_prompt(store, item["messages"][MESSAGE_KEY], guard=GUARD))
    res = score_response(raw, item, KB_TEXT[item["store"]])
    try:  # token usage / cost / latency recorded by the SDK for this item's chat
        u = kbench.chats.get_current_chat().usage
        res["usage"] = {"in_tok": u.input_tokens, "out_tok": u.output_tokens,
                        "cost_nd": u.total_cost_nanodollars, "latency_ms": u.total_backend_latency_ms}
    except Exception:  # noqa: BLE001
        res["usage"] = {}
    return res


# %%
def _summarise(results: list, n_total: int) -> dict:
    df = pd.DataFrame(results)
    traps = df[df.trap]
    hard = df[df.tags.apply(lambda t: "hard" in t)]
    base = df[df.tags.apply(lambda t: "hard" not in t)]
    tag_rows = df.explode("tags")
    return {
        "variant": VARIANT,
        "n_items": n_total,
        "n_completed": len(df),
        "score": round(float(df.score.sum() / n_total), 4),
        "understanding": round(float(df.understanding.sum() / n_total), 4),
        "grounding": round(float(df.grounding.sum() / n_total), 4),
        "json_parse_rate": round(float(df.parse_ok.mean()), 4),
        "hallucination_rate": round(float(traps.hallucinated.mean()), 4) if len(traps) else None,
        "base_score": round(float(base.score.mean()), 4) if len(base) else None,
        "hard_score": round(float(hard.score.mean()), 4) if len(hard) else None,
        "by_store": df.groupby("store").score.mean().round(4).to_dict(),
        "by_tag": tag_rows.groupby("tags").score.mean().round(4).to_dict(),
        "diag": _diag_summary(results),
        "items": [
            {k: r[k] for k in ("id", "score", "understanding", "grounding", "hallucinated",
                               "invented_numbers", "reply", "fields", "raw", "diag", "usage") if k in r}
            for r in results
        ],
    }


def _rate(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(bool(x) for x in xs) / len(xs), 4) if xs else None


def _avg(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return round(sum(xs) / len(xs), 2) if xs else None


def _diag_summary(results: list) -> dict:
    d = [r.get("diag") or {} for r in results]
    us = [r.get("usage") or {} for r in results]
    answerable_with_facts = [r.get("diag") or {} for r in results
                             if r.get("answerable") and (r.get("fields") or {}).get("facts") is not None]
    cost = [u.get("cost_nd") for u in us if isinstance(u.get("cost_nd"), (int, float))]
    return {
        "over_deferral_rate": _rate([x.get("over_deferral") for x in answerable_with_facts]),
        "deferral_rate_answerable": _rate([x.get("deferral") for x in answerable_with_facts]),
        "md_bold_rate": _rate([x.get("md_bold") for x in d]),
        "md_any_rate": _rate([x.get("md_bold") or x.get("md_heading") for x in d]),
        "bullet_reply_rate": _rate([(x.get("bullet_lines") or 0) >= 2 for x in d]),
        "avg_reply_chars": _avg([x.get("reply_chars") for x in d]),
        "avg_emoji": _avg([x.get("emoji") for x in d]),
        "anda_rate": _rate([x.get("says_anda") for x in d]),
        "kak_rate": _rate([x.get("says_kak") for x in d]),
        "mirror_regional_rate": _rate([x.get("mirrors_regional") for x in d]),
        "total_cost_usd": round(sum(cost) / 1e9, 5) if cost else None,
        "avg_latency_ms": _avg([u.get("latency_ms") for u in us]),
        "avg_out_tokens": _avg([u.get("out_tok") for u in us]),
    }


@kbench.task(
    name="umkm-bench-__VARIANT__",
    description="WhatsApp customer service in Indonesian (__MESSAGE_KEY__ messages__DESC_EXTRA__). Checks whether the model "
                "understood the order and whether its reply sticks to the shop info.",
)
def umkm_main(llm) -> tuple[float, float]:
    df = pd.DataFrame({"item_id": [it["id"] for it in ITEMS]})
    with kbench.client.enable_cache():
        runs = umkm_item.evaluate(
            llm=[llm],
            evaluation_data=df,
            n_jobs=2,
            timeout=900,
            on_failure="continue",
            remove_run_files=True,
        )
    results = list(runs.completed_runs.as_dataframe().result)
    # Second pass: retry anything that still errored, one item at a time.
    done = {r["id"] for r in results}
    for item_id in [i for i in df.item_id if i not in done]:
        try:
            run = umkm_item.run(llm, item_id=item_id)
            if isinstance(run.result, dict):
                results.append(run.result)
        except Exception as e:  # noqa: BLE001
            ERRORS.append(f"second pass {item_id}: {type(e).__name__}: {str(e)[:200]}")
    n_total = len(ITEMS)
    if ERRORS:
        print(f"API errors seen: {len(ERRORS)}. First few:")
        for msg in ERRORS[:5]:
            print("  ", msg)
    summary = _summarise(results, n_total) if results else {"variant": VARIANT, "n_items": n_total,
                                                         "n_completed": 0, "score": 0.0,
                                                         "json_parse_rate": 0.0}

    # Machine-readable line for analysis/analyze.py (errored items count as 0).
    print("UMKM_BENCH_SUMMARY=" + json.dumps(summary, ensure_ascii=False, default=str))

    kbench.assertions.assert_true(
        summary["json_parse_rate"] >= 0.9,
        expectation="Model returns valid JSON for at least 90% of messages.",
    )
    kbench.assertions.assert_true(
        summary["n_completed"] == n_total,
        expectation="Every item completed without infrastructure errors.",
    )

    scores = [r["score"] for r in results] + [0.0] * (n_total - len(results))
    mean = sum(scores) / n_total
    sd = math.sqrt(sum((s - mean) ** 2 for s in scores) / max(n_total - 1, 1))
    ci95 = 1.96 * sd / math.sqrt(n_total)
    return round(mean, 4), round(ci95, 4)


# %%
umkm_main.run(kbench.llm)

# %% On a Kaggle notebook, uncomment the next line so only the main task goes to the leaderboard:
# %choose umkm_main
