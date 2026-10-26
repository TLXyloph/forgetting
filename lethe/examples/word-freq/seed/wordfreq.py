"""Word frequency: the k most common words of a text."""

from __future__ import annotations


def top_words(text: str, k: int) -> list[tuple[str, int]]:
    """Return the k most frequent words as (word, count) pairs, most frequent first.

    A word is a maximal run of the characters a-z and ' in the lowercased text. Ties are broken
    by the word in ascending order. Fewer than k pairs are returned when there are fewer words.
    """
    # Split the lowercased text into words, one character at a time.
    words: list[str] = []
    current = ""
    for ch in text.lower():
        if "a" <= ch <= "z" or ch == "'":
            current += ch
        elif current:
            words.append(current)
            current = ""
    if current:
        words.append(current)

    # Count: keep a list of [word, count] pairs and scan it for every word.
    counts: list[list] = []
    for word in words:
        for pair in counts:
            if pair[0] == word:
                pair[1] += 1
                break
        else:
            counts.append([word, 1])

    # Select: repeatedly take the best remaining pair.
    result: list[tuple[str, int]] = []
    while counts and len(result) < k:
        best = counts[0]
        for pair in counts:
            if pair[1] > best[1] or (pair[1] == best[1] and pair[0] < best[0]):
                best = pair
        result.append((best[0], best[1]))
        counts.remove(best)
    return result
