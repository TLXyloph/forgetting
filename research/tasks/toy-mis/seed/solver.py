"""Maximum independent set: read a graph from stdin and print a large independent set.

Input:  "n m", then m lines "u v" (0-based vertex ids, one undirected edge per line).
Output: one line with the chosen vertex ids separated by spaces.
"""

from __future__ import annotations

import random
import sys


def main() -> None:
    data = sys.stdin.read().split()
    n, m = int(data[0]), int(data[1])
    adjacent: list[list[int]] = [[] for _ in range(n)]
    for i in range(m):
        u, v = int(data[2 + 2 * i]), int(data[3 + 2 * i])
        adjacent[u].append(v)
        adjacent[v].append(u)

    # Greedy: scan the vertices in a random order (fixed seed) and take every vertex that is
    # not adjacent to one already taken.
    order = list(range(n))
    random.Random(12345).shuffle(order)
    blocked = [False] * n
    chosen: list[int] = []
    for v in order:
        if not blocked[v]:
            chosen.append(v)
            blocked[v] = True
            for u in adjacent[v]:
                blocked[u] = True
    print(" ".join(map(str, chosen)))


if __name__ == "__main__":
    main()
