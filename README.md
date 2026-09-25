# UMKM-Bench

A small benchmark that asks one question: what happens to an LLM customer service bot when Indonesian customers type the way they really type on WhatsApp?

I build WhatsApp bots for small Indonesian businesses (UMKM). The messages look like "kak arabika yg setengah kilo ready gk?", not "Apakah Arabika Merapi 500 gram masih tersedia?". This repo holds the data, the scorer and the Kaggle tasks I used to measure the difference.

The write up with all the results is on DEV: [[DEV ARTICLE LINK]]
The Kaggle benchmark is here: [[KAGGLE BENCHMARK LINK]]

## What's in it

Three fictional shops, each with a small knowledge base (catalog, stock, shipping table, policies):

- Kopi Lereng, a coffee roaster in Sleman. The customers mix in Javanese.
- Sekar Hijab, a hijab shop in Bandung. The customers mix in Sundanese.
- Dapur Bu Tini, a frozen food kitchen in Semarang. The customers mix in Javanese.

84 customer messages. 60 are everyday questions, and 17 of those can't be answered from the shop data on purpose. The other 24 are traps: a city right next to one in the shipping table, totals the model has to calculate, a customer who "remembers" a discount that never existed, and prompt injections.

Every message comes in three versions. Formal is polite standard Indonesian. Slang is written by hand the way people type on WhatsApp. Noisy is the slang version with typos added from a fixed seed, so every model sees the same typos.

The model plays the shop admin and returns JSON with the intent, SKU, quantity, city, a flag that says whether the shop data can answer the question, and the reply it would send.

## How the scoring works

There is no LLM judge. Half of the score is understanding: intent, SKU, quantity and city. The other half is grounding. The flag has to be right, the reply has to contain the needed fact (85.000, 85rb and 85k all count as the same price), and when the shop data can't answer the question, the reply may not contain a number that isn't in the data or in the customer's message.

On top of the score, each run logs things that matter on WhatsApp: markdown in the reply, "Anda" or "kak", how often the model hands an answerable question to the admin, and cost and latency per reply.

## Files

```
data/stores.json          the three shops
data/items.json           60 everyday messages with gold labels
data/items_hard.json      24 trap messages
data/messages_preview.md  every message in all three versions, generated
src/perturb.py            the typo generator
src/prompt.py             builds the knowledge base text and the prompt
src/scoring.py            JSON parser and scorer
src/task_template.py      the Kaggle task
build.py                  bundles everything into one notebook per task in dist/
dist/                     the generated notebooks, ready to import into Kaggle
tests/                    scorer tests and a dry run with a fake model
results/                  the summary for each model and task from my runs
analysis/                 report and charts built from results/
```

Kaggle wants one file per task, so the sources stay split up here and build.py writes the notebooks. If you change the data or the scorer, run build.py again.

## Running it

Locally, without any API calls:

```bash
pip install pytest matplotlib pandas
pip install git+https://github.com/Kaggle/kaggle-benchmarks
python build.py
python -m pytest -q tests
python -m tests.mock_run dist/umkm_bench_slang.py oracle
```

The oracle run should score 1.0. Try `sloppy` instead of `oracle` to see a model that makes things up.

On Kaggle, open https://www.kaggle.com/personal-benchmarks/new, use File and then Import Notebook to upload one of the files in dist/, run it once with the default model, and click Build Task. Then add models from the task page.

To rebuild the tables and charts from the results folder:

```bash
python analysis/analyze.py results
python analysis/charts.py results
```

## Adding your own shop or language

A shop is one entry in data/stores.json and a message is one entry in the items files. The three version idea works for any language where people text differently from how they write: write the message formally, then the way people really type it, then let the typo generator mess it up.

## License

MIT. The shops and messages are made up. Any resemblance to a real shop in Sleman is a coincidence, though I'd buy their coffee.
