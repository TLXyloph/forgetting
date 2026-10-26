"""Bin packing: pack items into as few bins as possible."""

from __future__ import annotations


def solve(sizes: list[int], capacity: int) -> list[list[int]]:
    """Return the packing as a list of bins; each bin is a list of item indices.

    Current approach: first-fit in input order.
    """
    bins: list[list[int]] = []
    loads: list[int] = []
    for index, size in enumerate(sizes):
        for b, load in enumerate(loads):
            if load + size <= capacity:
                bins[b].append(index)
                loads[b] += size
                break
        else:
            bins.append([index])
            loads.append(size)
    return bins
