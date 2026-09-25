"""Deterministic scoring for one model response. No LLM judge involved.

Item score = 0.5 * understanding + 0.5 * grounding
  understanding = mean over the gold fields that apply: intent, sku, quantity, city
  grounding     = mean over: can_answer_from_kb flag, required facts in reply,
                  and (for unanswerable items) "no invented numbers" in reply
                  and (for trap items) not saying the specific wrong thing the trap targets
A response that is not parseable JSON scores 0 on everything.
"""

import json
import re

# ---------------------------------------------------------------- patterns
NAMED_PATTERNS = {
    "@OOS": (
        r"habis|kosong|sold ?out|stok(nya)? (0|nol)"
        r"|(tidak|tdk|belum|blm|gak|ga|gk|nggak|ngga|enggak) (ada|ready|tersedia)"
    ),
    "@NOT_YET": (
        r"\b(tidak|tdk|belum|blm|gak|ga|gk|nggak|ngga|enggak)\b|kurang|di bawah|dibawah"
        r"|tetap (kena|dikenakan|bayar)|harga normal|harga satuan"
    ),
    "@NO_VERB": (
        r"\b(tidak|tdk|belum|blm|gak|ga|gk|nggak|ngga|enggak)\s+"
        r"(bisa|dapat|melayani|menerima|tersedia|ada|support|menyediakan)\b"
    ),
}

CITY_ALIASES = {
    "Yogyakarta": ["yogyakarta", "jogja", "jogjakarta", "yogya", "jogya", "diy", "yk", "djogja"],
    "Jakarta": ["jakarta", "jkt", "dki jakarta", "dki"],
    "Surabaya": ["surabaya", "sby"],
    "Semarang": ["semarang", "smg"],
    "Bandung": ["bandung", "bdg"],
    "Makassar": ["makassar", "makasar", "mks", "ujung pandang"],
    "Balikpapan": ["balikpapan", "bpp"],
    "Pontianak": ["pontianak", "ptk"],
    "Medan": ["medan", "mdn"],
    "Denpasar": ["denpasar", "dps"],
    "Surakarta": ["surakarta", "solo"],
    "Singapura": ["singapura", "singapore", "sg", "spore"],
    "Magelang": ["magelang", "mgl"],
    "Cimahi": ["cimahi"],
    "Kendal": ["kendal"],
}
_ALIAS_TO_CITY = {a: c for c, aliases in CITY_ALIASES.items() for a in aliases}

_NUM_WORDS = {
    "satu": 1, "dua": 2, "tiga": 3, "empat": 4, "lima": 5, "enam": 6,
    "tujuh": 7, "delapan": 8, "sembilan": 9, "sepuluh": 10, "sebelas": 11,
    "selusin": 12, "sekodi": 20, "seratus": 100, "seribu": 1000,
}
_UNITS = (
    r"hari|minggu|bulan|tahun|jam|menit|persen|ribu|rb|pcs|botol|bungkus|pack|pak"
    r"|buah|biji|lembar|kotak|level|kali"
)
_DIGIT_RE = re.compile(r"(\d+(?:[.,]\d{3})+|\d+)(\s*(?:rb|ribu|jt|juta|k(?![a-z])))?", re.I)
_WORDNUM_RE = re.compile(r"\b(" + "|".join(_NUM_WORDS) + r")\s+(" + _UNITS + r")\b", re.I)


# ---------------------------------------------------------------- helpers
def extract_numbers(text: str) -> set:
    """All numeric values mentioned in text, normalised (85.000 / 85rb / 85k -> 85000)."""
    out = set()
    if not text:
        return out
    for m in _DIGIT_RE.finditer(text):
        n = int(re.sub(r"[.,]", "", m.group(1)))
        suffix = (m.group(2) or "").strip().lower()
        if suffix in ("rb", "ribu", "k"):
            n *= 1000
        elif suffix in ("jt", "juta"):
            n *= 1_000_000
        out.add(n)
    for m in _WORDNUM_RE.finditer(text):
        n = _NUM_WORDS[m.group(1).lower()]
        if m.group(2).lower() in ("ribu", "rb"):
            n *= 1000
        out.add(n)
    return out


def normalize_city(value):
    if value is None:
        return None
    v = str(value).lower().split(",")[0]
    v = re.sub(r"^(kota|kab\.?|kabupaten)\s+", "", v.strip())
    v = re.sub(r"[^a-z ]", "", v).strip()
    if v in _ALIAS_TO_CITY:
        return _ALIAS_TO_CITY[v]
    for alias, city in _ALIAS_TO_CITY.items():  # e.g. "jakarta selatan"
        if len(alias) > 3 and v.startswith(alias):
            return city
    return v.title() if v else None


def _to_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("true", "ya", "yes", "1"):
            return True
        if v in ("false", "tidak", "no", "0"):
            return False
    return None


def _to_int(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None


def _extract_json(text: str):
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:i + 1]
                    for fix in (lambda s: s, lambda s: re.sub(r",\s*([}\]])", r"\1", s)):
                        try:
                            obj = json.loads(fix(candidate))
                            if isinstance(obj, dict):
                                return obj
                        except json.JSONDecodeError:
                            pass
                    break
        start = text.find("{", start + 1)
    return None


def parse_response(raw):
    """Pull the first JSON object out of a model response. Returns dict or None.
    Curly quotes are only normalised as a fallback: inside a valid JSON string
    they are ordinary characters (e.g. a reply quoting “Mks”)."""
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return None
    obj = _extract_json(raw)
    if obj is None:
        obj = _extract_json(raw.replace("\u201c", '"').replace("\u201d", '"'))
    return obj


def _fact_ok(fact: dict, reply: str, reply_numbers: set) -> bool:
    if "num" in fact:
        return fact["num"] in reply_numbers
    if "any_num" in fact:
        return any(n in reply_numbers for n in fact["any_num"])
    return re.search(expand_pattern(fact["re"]), reply, re.I) is not None


def expand_pattern(pattern: str) -> str:
    """Replace @NAME tokens (e.g. "@NO_VERB|hanya") with their named regex."""
    for name, regex in NAMED_PATTERNS.items():
        pattern = pattern.replace(name, f"(?:{regex})")
    return pattern


def allowed_numbers(kb_text: str, item: dict) -> set:
    allowed = extract_numbers(kb_text) | {0, 1}
    allowed |= extract_numbers(item["formal"]) | extract_numbers(item["slang"])
    if item["gold"].get("qty") is not None:
        allowed.add(item["gold"]["qty"])
    return allowed


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


# ---------------------------------------------------------------- diagnostics
# These never change the score. They describe *how* a model answers, which is
# what a shop owner actually feels after deploying it on WhatsApp.
DEFERRAL_RE = re.compile(
    r"admin (akan )?(cek|mengecek|konfirmasi|bantu cek)|(di|ter)?cek (dulu|manual|terlebih)"
    r"|mengecek(nya)? (dulu|terlebih)|saya cek|kami cek|dicek(kan)? dulu|konfirmasi (dulu|ke)"
    r"|diteruskan ke|tanyakan (dulu )?ke|mohon (di)?tunggu", re.I)
REGIONAL_WORDS = {
    "regional_javanese": re.compile(r"\b(nggih|njih|nggeh|monggo|matur nuwun|sampun|mboten|niki|nopo|inggih|nyuwun|sugeng)\b", re.I),
    "regional_sundanese": re.compile(r"\b(punten|hatur nuhun|mangga|sawios|teh|akang|teteh|nuhun|atuh|mah)\b", re.I),
}
EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF\u2600-\u27BF]")


def reply_diagnostics(reply: str, item: dict, flag, facts_ok) -> dict:
    gold = item["gold"]
    deferral = bool(DEFERRAL_RE.search(reply))
    regional_tag = next((t for t in item["tags"] if t in REGIONAL_WORDS), None)
    return {
        "reply_chars": len(reply),
        "md_bold": "**" in reply,                       # WhatsApp bold is *single*, so ** shows up raw
        "md_heading": bool(re.search(r"(^|\n)#{1,4} ", reply)),
        "bullet_lines": len(re.findall(r"(^|\n)\s*([-\u2022*]|\d+[.)])\s", reply)),
        "emoji": len(EMOJI_RE.findall(reply)),
        "says_anda": bool(re.search(r"\banda\b", reply, re.I)),
        "says_kak": bool(re.search(r"\bka(k|kak)?\b", reply, re.I)),
        "deferral": deferral,
        # the KB had the answer, yet the model handed the question to a human without giving it
        "over_deferral": bool(gold["answerable"] and gold["facts"] and deferral and not facts_ok),
        "facts_ok": facts_ok,
        "mirrors_regional": (bool(REGIONAL_WORDS[regional_tag].search(reply)) if regional_tag else None),
    }


# ---------------------------------------------------------------- main scorer
def score_response(raw, item: dict, kb_text: str) -> dict:
    gold = item["gold"]
    parsed = parse_response(raw)
    res = {
        "id": item["id"], "store": item["store"], "tags": item["tags"],
        "answerable": gold["answerable"], "parse_ok": parsed is not None,
        "trap": (not gold["answerable"]) or bool(gold.get("forbid")),
    }
    if parsed is None:
        res.update(understanding=0.0, grounding=0.0, score=0.0,
                   hallucinated=res["trap"], reply=str(raw)[:300])
        return res

    reply = str(parsed.get("reply") or "")
    reply_numbers = extract_numbers(reply)

    # ---- understanding
    u = {}
    u["intent"] = float(str(parsed.get("intent", "")).strip().lower() in gold["intent"])
    if gold.get("sku"):
        u["sku"] = float(str(parsed.get("sku") or "").strip().upper() in gold["sku"])
    if gold.get("qty") is not None:
        u["qty"] = float(_to_int(parsed.get("quantity")) == gold["qty"])
    if gold.get("city") is not None:
        u["city"] = float(normalize_city(parsed.get("city")) == gold["city"])

    # ---- grounding
    g = {}
    flag = _to_bool(parsed.get("can_answer_from_kb"))
    # flag_any: items where "can it be answered from the KB" is genuinely ambiguous
    if not gold.get("flag_any"):
        g["flag"] = float(flag == gold["answerable"])
    facts_ok = None
    if gold["facts"]:
        facts_ok = all(_fact_ok(f, reply, reply_numbers) for f in gold["facts"])
        g["facts"] = float(facts_ok)
    invented = []
    if (not gold["answerable"]) or gold.get("check_fabrication"):
        invented = sorted(reply_numbers - allowed_numbers(kb_text, item))
        g["no_invented_numbers"] = float(not invented)

    forbidden_hit = any(_fact_ok(f, reply, reply_numbers) for f in gold.get("forbid", []))
    if gold.get("forbid"):
        g["no_forbidden"] = float(not forbidden_hit)

    understanding = _mean(list(u.values()))
    grounding = _mean(list(g.values())) if g else understanding
    res.update(
        fields={**u, **g},
        understanding=understanding,
        grounding=grounding,
        score=0.5 * understanding + 0.5 * grounding,
        invented_numbers=invented,
        # hallucination = claims it can answer something it can't, invents numbers,
        # or says the specific wrong thing a trap was built to provoke
        hallucinated=((not gold["answerable"]) and (flag is True or bool(invented))) or forbidden_hit,
        reply=reply[:300],
        # raw extraction, kept so results can be re-scored offline without re-running models
        raw={k: parsed.get(k) for k in ("intent", "sku", "quantity", "city", "can_answer_from_kb")},
        diag=reply_diagnostics(reply, item, flag, facts_ok),
    )
    return res
