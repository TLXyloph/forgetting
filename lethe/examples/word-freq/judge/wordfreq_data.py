"""Deterministic benchmark data for the word-freq judge: the reference, corpus and gate cases.

Everything is generated from fixed seeds with `random.Random(seed).random()` (a stream Python
guarantees), so every machine and Python version sees identical text. Standard library only.
"""

from __future__ import annotations

import bisect
import itertools
import random
import re
import zlib
from collections import Counter
from typing import Any

CORPUS_WORDS = 300_000
VOCAB_SIZE = 1500
ZIPF_S = 1.1
WORD_RE = re.compile(r"[a-z']+")


def reference(text: str, k: int) -> list[tuple[str, int]]:
    """The specification: lowercase, split into [a-z'] runs, count, order by (-count, word)."""
    counts = Counter(WORD_RE.findall(text.lower()))
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:k]


# ---------------------------------------------------------------------------------------------
# deterministic generation
# ---------------------------------------------------------------------------------------------


class Rng:
    def __init__(self, seed: int) -> None:
        self.random = random.Random(seed).random

    def randint(self, lo: int, hi: int) -> int:
        return lo + min(hi - lo, int(self.random() * (hi - lo + 1)))

    def pick(self, seq: tuple[str, ...] | list[str]) -> str:
        return seq[self.randint(0, len(seq) - 1)]

    def shuffle(self, items: list[Any]) -> None:
        for i in range(len(items) - 1, 0, -1):
            j = self.randint(0, i)
            items[i], items[j] = items[j], items[i]


def word_list(spec: str) -> list[str]:
    return spec.split()


ONSETS = word_list(
    "b c d f g h j k l m n p r s t v w z br ch cl cr dr fl gr pl pr sh sl sp st th tr"
)
VOWELS = word_list("a e i o u ai ea ee io oa oo ou")
CODAS = ["", "", "", "n", "r", "s", "t", "l", "d", "m", "ng", "nt", "st", "rd"]
CONTRACTIONS = ("'s", "'t", "'ll", "'re", "'d", "'ve")
HEAD = word_list(
    "the of and a to in is it's that for was on with as i don't at by this be are or from"
    " an have not but had they you he she we his her its their there which will would can"
)
PUNCT = (",", ".", ";", ":", "!", "?", ")", '"')
_DIGITS = str.maketrans("0123456789", "abcdefghij")


def make_vocab(rng: Rng, size: int) -> list[str]:
    """`size` distinct lowercase pseudo-words: common function words first, then short to long."""
    seen = set(HEAD)
    extra: list[str] = []
    while len(seen) < size:
        word = "".join(
            rng.pick(ONSETS) + rng.pick(VOWELS) + rng.pick(CODAS)
            for _ in range(1 + rng.randint(0, 9) // 4)
        )
        if rng.randint(0, 24) == 0:
            word += rng.pick(CONTRACTIONS)
        if word not in seen:
            seen.add(word)
            extra.append(word)
    extra.sort(key=lambda w: (len(w), w))
    return list(HEAD) + extra


def zipf_cumulative(size: int) -> list[float]:
    return list(itertools.accumulate(1.0 / (r + 1) ** ZIPF_S for r in range(size)))


def noisy_text(rng: Rng, vocab: list[str], cum: list[float], words: int) -> str:
    """Zipf-distributed words with case, punctuation, numbers, hyphens and varied whitespace."""
    total = cum[-1]
    out: list[str] = []
    for _ in range(words):
        word = vocab[bisect.bisect_left(cum, rng.random() * total)]
        style = rng.random()
        if style < 0.08:
            word = word.capitalize()
        elif style < 0.09:
            word = word.upper()
        elif style < 0.095:
            word = "'" + word
        if rng.random() < 0.012:
            word += "-" + vocab[bisect.bisect_left(cum, rng.random() * total)]
        if rng.random() < 0.10:
            word += PUNCT[int(rng.random() * len(PUNCT))]
        out.append(word)
        gap = rng.random()
        if gap < 0.84:
            out.append(" ")
        elif gap < 0.92:
            out.append("\n")
        elif gap < 0.94:
            out.append("  ")
        elif gap < 0.95:
            out.append("\t")
        elif gap < 0.97:
            out.append(" - ")
        elif gap < 0.99:
            out.append(f" {rng.randint(1, 2999)} ")
        else:
            out.append(f" {rng.randint(0, 99)}.{rng.randint(0, 99)}, ")
    return "".join(out)


def build_corpus() -> str:
    rng = Rng(zlib.crc32(b"word-freq/corpus"))
    vocab = make_vocab(rng, VOCAB_SIZE)
    return noisy_text(rng, vocab, zipf_cumulative(len(vocab)), CORPUS_WORDS)


def gate_cases() -> list[tuple[str, str, int]]:
    """(kind, text, k) triples. Kinds are what the feedback may name."""
    rng = Rng(zlib.crc32(b"word-freq/gate"))
    cases: list[tuple[str, str, int]] = []

    def add(kind: str, text: str, k: int) -> None:
        cases.append((kind, text, k))

    def zipf(words: int, vocab_size: int) -> str:
        vocab = make_vocab(rng, vocab_size)
        return noisy_text(rng, vocab, zipf_cumulative(len(vocab)), words)

    add("empty", "", 5)
    add("empty", "", 0)
    add("no-words", "123 456 !!! --- 7.89 ... ??", 5)
    add("no-words", "\n\n\t  \r\n", 3)
    add("single-word", "hello", 1)
    add("single-word", "hello " * 1000, 3)
    add("single-word", "a", 5)
    add("k-zero", "the quick brown fox jumps over the lazy dog the end", 0)
    add("k-zero", zipf(300, 60), 0)
    add("k-large", "b a b c", 1000)
    add("k-large", zipf(400, 80), 10_000)
    add("k-large", "x y z", 3)

    add("ties", "delta alpha charlie bravo echo", 3)
    add("ties", "b b a a c c", 2)
    fruits = word_list("quartz apple mango zebra kiwi banana cherry date elder fig grape honey")
    twice = fruits * 2
    rng.shuffle(twice)
    for k in (5, 7, 12, 30):
        add("ties", " ".join(twice), k)
    big_tie = [
        f"{rng.pick(ONSETS)}{rng.pick(VOWELS)}{i:03d}".translate(_DIGITS) for i in range(300)
    ]
    rng.shuffle(big_tie)
    add("ties", " ".join(big_tie), 17)
    mixed = ["pear"] * 3 + ["fig"] * 3 + ["plum"] * 3 + ["kiwi"] * 2 + ["date"] * 2
    mixed += ["lime", "apple", "nut"]
    rng.shuffle(mixed)
    for k in (2, 4, 6, 9):
        add("ties", " ".join(mixed), k)

    add("case", "The the THE tHe ThE", 1)
    add("case", "Apple apple APPLE aPPle banana BANANA Banana cherry", 5)
    add(
        "apostrophes",
        "don't Don't DON'T dont 'tis ''' '' ' o'clock rock 'n' roll it's its it's",
        20,
    )
    add("apostrophes", "'a' 'b' 'a' ''quoted'' rock'n'roll can't can''t", 20)
    add("apostrophes", "it's a dog's life, isn't it? the dogs' bones; 'twas brillig", 30)
    add("separators", "foo,bar.baz;qux!x2y z_w well-known (hello)[world]{x} a/b c|d e+f g=h", 50)
    add("separators", "tab\tsep\r\nline\x0bvt\x0cff nbsp\u00a0here em\u2003space zw\u200bspace", 50)
    add("separators", "word1word word2word 3word word_word WORD-WORD word.word", 50)
    # Non-ASCII text: accented letters, CJK and emoji are separators; str.lower() maps the Kelvin
    # sign (U+212A) to ASCII k and turns U+0130 into 'i' plus a combining dot (a separator).
    add(
        "unicode",
        "caf\u00e9 na\u00efve r\u00e9sum\u00e9 \u65e5\u672c\u8a9e emoji \U0001f600 "
        "stra\u00dfe \u0130stanbul \u212aelvin KELVIN \u0391\u0392\u0393 "
        "\u03b1\u03b2\u03b3 \u00c0\u00c9\u00ce cafe",
        50,
    )
    add(
        "unicode",
        "\u00c5\u00c4\u00d6 \u00e5\u00e4\u00f6 abc ABC \u01c6 \u01c4 \u00df "
        "\u017f \ufb01ne fine \u0153uvre oeuvre",
        50,
    )
    add("long-word", "a" * 100_000 + " " + "b" * 10, 5)
    add("long-word", "x" * 5000 + "'" * 5000 + " " + "x" * 5000, 2)

    for words, vocab_size, ks in ((200, 30, (1, 5, 10)), (1000, 100, (10, 50)), (3000, 300, (25,))):
        text = zipf(words, vocab_size)
        for k in ks:
            add("random-small", text, k)
    distinct = sorted(
        {
            rng.pick(ONSETS)
            + rng.pick(VOWELS)
            + rng.pick(CODAS)
            + rng.pick(ONSETS)
            + rng.pick(VOWELS)
            for _ in range(4000)
        }
    )[:1500]
    rng.shuffle(distinct)
    add("many-distinct", " ".join(distinct), 100)
    add("many-distinct", " ".join(distinct + distinct[:700] + distinct[:300]), 200)
    big = zipf(20_000, 800)
    add("large", big, 25)
    add("large", big, 500)
    return cases
