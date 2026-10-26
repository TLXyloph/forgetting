"""Experiment files: one task, several arms (configuration variants) and a replicate count.

    name = "paper-toy-mis"                     # [A-Za-z0-9_.-]+, becomes a directory name
    task = "../../tasks/toy-mis/lethe.toml"    # relative to the experiment file
    replicates = 3                             # optional, default 1
    [overrides]                                # deep-merged into every arm (lists replace)
    budget = { usd = 3.0 }
    [[arms]]
    name = "solo"
    preset = "solo"                            # optional lethe preset
    [[arms]]
    name = "anchored"
    preset = "team"
    overrides = { forget = { when = [ { type = "anchored" } ] } }

Overrides are shaped like a `lethe.toml` file and are handed to `lethe.api.run_task` unchanged, so
lethe itself validates their keys and values. Here only the structure is checked: unknown keys
are errors, and so are override sections or presets that the config schema does not define,
because a typo there would otherwise surface hours into a matrix.
"""

from __future__ import annotations

import copy
import re
import tomllib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "OVERRIDE_SECTIONS",
    "PRESETS",
    "Arm",
    "Experiment",
    "ExperimentError",
    "arm_overrides",
    "deep_merge",
    "is_valid_name",
    "load_experiment",
    "select_arms",
]

NAME_PATTERN = re.compile(r"[A-Za-z0-9_.-]+")

# Mirrors the lethe.toml schema (config-schema.md): the presets and the top-level sections.
PRESETS: tuple[str, ...] = ("solo", "parallel", "team", "team-forget")
OVERRIDE_SECTIONS: tuple[str, ...] = ("task", "judge", "team", "forget", "agent", "budget", "run")

_TOP_LEVEL_KEYS = frozenset({"name", "task", "replicates", "overrides", "arms"})
_ARM_KEYS = frozenset({"name", "preset", "overrides"})
_MISSING = object()


class ExperimentError(ValueError):
    """An experiment file, or a selection of its arms, is invalid."""


@dataclass(frozen=True)
class Arm:
    """One configuration variant. `preset=None` leaves the preset to the task's own config."""

    name: str
    preset: str | None = None
    overrides: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Experiment:
    """A parsed experiment. `task` is absolute; `overrides` apply to every arm."""

    name: str
    task: Path
    replicates: int
    overrides: Mapping[str, Any]
    arms: tuple[Arm, ...]


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Merge `override` into `base`: nested tables merge key by key, anything else replaces.

    Lists are values, not tables, so an overriding list replaces the base list entirely
    (an empty list therefore clears it). Neither argument is modified, and the result shares
    no mutable state with either.
    """
    merged = copy.deepcopy(dict(base))
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(value, Mapping) and isinstance(current, Mapping):
            merged[key] = deep_merge(current, value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def arm_overrides(experiment: Experiment, arm: Arm) -> dict[str, Any]:
    """The overrides one arm runs with: experiment-level overrides, then the arm's own."""
    return deep_merge(experiment.overrides, arm.overrides)


def select_arms(experiment: Experiment, names: Iterable[str] | None) -> tuple[Arm, ...]:
    """The arms called `names`, in file order (all arms when `names` is None)."""
    if names is None:
        return experiment.arms
    wanted = [names] if isinstance(names, str) else list(names)
    known = [arm.name for arm in experiment.arms]
    unknown = list(dict.fromkeys(name for name in wanted if name not in known))
    if unknown:
        raise ExperimentError(
            f"unknown arm(s): {', '.join(unknown)} (known arms: {', '.join(known)})"
        )
    if not wanted:
        raise ExperimentError("no arms selected")
    return tuple(arm for arm in experiment.arms if arm.name in wanted)


def load_experiment(path: Path | str) -> Experiment:
    """Read and validate an experiment file. Raises `ExperimentError` describing the problem."""
    path = Path(path)
    with path.open("rb") as fh:
        try:
            raw = tomllib.load(fh)
        except tomllib.TOMLDecodeError as exc:
            raise ExperimentError(f"{path}: invalid TOML: {exc}") from exc
    return _parse(raw, path)


# ---------------------------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------------------------


def is_valid_name(value: object) -> bool:
    """True for names usable as a single directory component: [A-Za-z0-9_.-]+, not only dots."""
    return (
        isinstance(value, str)
        and NAME_PATTERN.fullmatch(value) is not None
        and bool(value.strip("."))
    )


def _check_name(value: object, what: str, where: str) -> str:
    if value is _MISSING:
        raise ExperimentError(f"{where}: missing required key: {what}")
    if not isinstance(value, str) or not is_valid_name(value):
        raise ExperimentError(
            f"{where}: {what} must match [A-Za-z0-9_.-]+ and not consist only of dots "
            f"(it becomes a directory name), got {value!r}"
        )
    return value


def _check_unknown_keys(raw: Mapping[str, Any], allowed: frozenset[str], where: str) -> None:
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ExperimentError(f"{where}: unknown key(s) {unknown}; allowed keys: {sorted(allowed)}")


def _check_overrides(value: object, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ExperimentError(f"{where}: overrides must be a table, got {value!r}")
    unknown = sorted(set(value) - set(OVERRIDE_SECTIONS))
    if unknown:
        hint = " (the preset is set per arm with `preset = ...`)" if "preset" in unknown else ""
        raise ExperimentError(
            f"{where}: unknown override section(s) {unknown}; "
            f"sections of lethe.toml are {list(OVERRIDE_SECTIONS)}{hint}"
        )
    for section, body in value.items():
        if not isinstance(body, dict):
            raise ExperimentError(f"{where}: overrides.{section} must be a table, got {body!r}")
    return copy.deepcopy(value)


def _parse_arm(raw: object, index: int, where: str) -> Arm:
    if not isinstance(raw, dict):
        raise ExperimentError(f"{where}: arms must be an array of tables ([[arms]]), got {raw!r}")
    _check_unknown_keys(raw, _ARM_KEYS, f"{where}: arm #{index + 1}")
    name = _check_name(raw.get("name", _MISSING), f"arm #{index + 1} name", where)
    label = f"{where}: arm {name!r}"
    preset: str | None = None
    if "preset" in raw:
        preset = raw["preset"]
        if not isinstance(preset, str) or preset not in PRESETS:
            raise ExperimentError(f"{label}: preset must be one of {list(PRESETS)}, got {preset!r}")
    overrides = _check_overrides(raw.get("overrides", {}), label)
    return Arm(name=name, preset=preset, overrides=overrides)


def _parse(raw: dict[str, Any], path: Path) -> Experiment:
    where = str(path)
    _check_unknown_keys(raw, _TOP_LEVEL_KEYS, where)
    name = _check_name(raw.get("name", _MISSING), "name", where)

    task = raw.get("task", _MISSING)
    if task is _MISSING:
        raise ExperimentError(f"{where}: missing required key: task")
    if not isinstance(task, str) or not task.strip():
        raise ExperimentError(f"{where}: task must be a non-empty path string, got {task!r}")

    replicates = raw.get("replicates", 1)
    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates < 1:
        raise ExperimentError(f"{where}: replicates must be an integer >= 1, got {replicates!r}")

    overrides = _check_overrides(raw.get("overrides", {}), where)

    raw_arms = raw.get("arms")
    if not isinstance(raw_arms, list) or not raw_arms:
        raise ExperimentError(f"{where}: at least one arm ([[arms]] table) is required")
    arms = tuple(_parse_arm(item, index, where) for index, item in enumerate(raw_arms))

    seen: set[str] = set()
    for arm in arms:
        # arm names become directories; macOS volumes are case-insensitive by default
        if arm.name.lower() in seen:
            raise ExperimentError(
                f"{where}: duplicate arm name {arm.name!r} (names must be unique, ignoring case)"
            )
        seen.add(arm.name.lower())

    return Experiment(
        name=name,
        task=(path.absolute().parent / task).resolve(),
        replicates=replicates,
        overrides=overrides,
        arms=arms,
    )
