"""Deterministic "noisy thumb" perturbation for WhatsApp-style messages.

Turns the hand-written slang message into a typo-heavy variant with a fixed
seed per item, so every model sees exactly the same noisy text.
Digits are never touched (changing a number would change the gold label).
"""

import random
import re

# Neighbouring keys on a QWERTY phone keyboard (letters only).
_QWERTY_NEIGHBOURS = {
    "a": "qwsz", "b": "vghn", "c": "xdfv", "d": "serfcx", "e": "wsdr",
    "f": "drtgvc", "g": "ftyhbv", "h": "gyujnb", "i": "ujko", "j": "huikmn",
    "k": "jiolm", "l": "kop", "m": "njk", "n": "bhjm", "o": "iklp",
    "p": "ol", "q": "wa", "r": "edft", "s": "awedxz", "t": "rfgy",
    "u": "yhji", "v": "cfgb", "w": "qase", "x": "zsdc", "y": "tghu",
    "z": "asx",
}
_VOWELS = set("aiueo")
_WORD_RE = re.compile(r"[A-Za-z]+")


def _swap(word: str, rng: random.Random) -> str:
    if len(word) < 4:
        return word
    i = rng.randrange(1, len(word) - 2)
    return word[:i] + word[i + 1] + word[i] + word[i + 2:]


def _drop_vowel(word: str, rng: random.Random) -> str:
    idx = [i for i, ch in enumerate(word) if i > 0 and ch.lower() in _VOWELS]
    if not idx:
        return word
    i = rng.choice(idx)
    return word[:i] + word[i + 1:]


def _repeat_last(word: str, rng: random.Random) -> str:
    return word + word[-1] * rng.randint(1, 3)


def _fat_finger(word: str, rng: random.Random) -> str:
    idx = [i for i, ch in enumerate(word) if ch.lower() in _QWERTY_NEIGHBOURS]
    if not idx:
        return word
    i = rng.choice(idx)
    repl = rng.choice(_QWERTY_NEIGHBOURS[word[i].lower()])
    return word[:i] + repl + word[i + 1:]


_OPS = (_swap, _drop_vowel, _repeat_last, _fat_finger)


def make_noisy(text: str, seed: str, p_word: float = 0.35, p_join: float = 0.25) -> str:
    """Return a deterministic typo-heavy version of ``text``.

    - each alphabetic word of length >= 4 is corrupted with probability ``p_word``
    - adjacent short words are glued together with probability ``p_join``
    - question marks are dropped half of the time
    """
    rng = random.Random(f"umkmbench::{seed}")

    def corrupt(match: re.Match) -> str:
        word = match.group(0)
        if len(word) >= 4 and rng.random() < p_word:
            return rng.choice(_OPS)(word, rng)
        return word

    out_lines = []
    for line in text.lower().split("\n"):
        line = _WORD_RE.sub(corrupt, line)
        tokens = line.split(" ")
        glued = [tokens[0]] if tokens else []
        for tok in tokens[1:]:
            prev = glued[-1]
            if (
                prev.isalpha() and tok.isalpha()
                and len(prev) <= 4 and len(tok) <= 4
                and rng.random() < p_join
            ):
                glued[-1] = prev + tok
            else:
                glued.append(tok)
        line = " ".join(glued)
        if rng.random() < 0.5:
            line = line.replace("?", "")
        out_lines.append(line.strip())
    noisy = "\n".join(out_lines)

    # Guarantee at least one corrupted word so no message passes through untouched.
    if noisy == text.lower():
        words = [m for m in _WORD_RE.finditer(noisy) if len(m.group(0)) >= 4]
        if words:
            m = rng.choice(words)
            noisy = noisy[:m.start()] + _fat_finger(m.group(0), rng) + noisy[m.end():]
    return noisy
