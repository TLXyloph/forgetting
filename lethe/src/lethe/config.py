"""Loading and validating `lethe.toml` task/run configuration.

Precedence: built-in defaults < preset < config file < overrides (nested dicts, deep-merged;
lists replace). Unknown keys are errors so typos never silently change an experiment.
"""

from __future__ import annotations

import copy
import math
import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

from lethe.agents.claude_code import ClaudeCodeOptions
from lethe.judge.runner import JudgeSpec
from lethe.policy.triggers import trigger_from_dict
from lethe.runtime.budget import BudgetSpec
from lethe.workspace.paths import PathFilter, glob_to_regex

NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
ALWAYS_EXCLUDE = (".git/**", "**/__pycache__/**", "**/*.pyc", "**/.DS_Store", ".lethe/**")
HARNESS_FILES = ("TASK.md", ".context/**")
EFFORTS = ("", "low", "medium", "high", "xhigh", "max")

DEFAULTS: dict[str, Any] = {
    "preset": "team-forget",
    "task": {
        "name": None,
        "description": None,
        "description_file": None,
        "seed": None,
        "editable": None,
        "exclude": [],
    },
    "judge": {
        "run": None,
        "direction": None,
        "timeout": 600,
        "build": "",
        "concurrency": 2,
        "cache": True,
        "max_feedback_chars": 4000,
        "split": "dev",
        "env": {},
    },
    "team": {"agents": 4, "peers": "all", "session": "continue"},
    "forget": {"seed": "best", "carry_scratch": True, "when": []},
    "agent": {
        "backend": "claude-code",
        "model": "sonnet",
        "effort": "",
        "isolation": "safe",
        "shell": False,
        "turn_timeout": 1800,
        "extra_args": [],
    },
    "budget": {
        "usd": 5.0,
        "proposals": 0,
        "wall": 0,
        "primary": "usd",
        "pause_at_utilization": 0.95,
    },
    "run": {"runs_dir": ".lethe/runs"},
}


class ConfigError(ValueError):
    """The configuration is invalid; the message says which key and why."""


@dataclass(frozen=True)
class TaskConfig:
    name: str
    description: str
    config_dir: Path
    seed_dir: Path
    editable: tuple[str, ...]
    exclude: tuple[str, ...]

    @property
    def judged(self) -> PathFilter:
        return PathFilter(self.editable, self.exclude)

    @property
    def archive_exclude(self) -> PathFilter:
        return PathFilter((*HARNESS_FILES, *self.exclude))


@dataclass(frozen=True)
class TeamConfig:
    agents: int
    peers: str
    session: str

    @property
    def agent_ids(self) -> tuple[str, ...]:
        return tuple(f"a{i}" for i in range(1, self.agents + 1))


@dataclass(frozen=True)
class ForgetConfig:
    seed: str
    carry_scratch: bool
    when: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class AgentConfig:
    backend: str
    model: str | None
    effort: str | None
    isolation: str
    shell: bool
    turn_timeout: float
    extra_args: tuple[str, ...]

    def claude_options(self) -> ClaudeCodeOptions:
        return ClaudeCodeOptions(
            model=self.model,
            effort=self.effort,
            shell=self.shell,
            isolation=self.isolation,  # type: ignore[arg-type]
            turn_timeout_s=self.turn_timeout,
            extra_args=self.extra_args,
        )


@dataclass(frozen=True)
class RunConfig:
    preset: str
    task: TaskConfig
    judge: JudgeSpec
    team: TeamConfig
    forget: ForgetConfig
    agent: AgentConfig
    budget: BudgetSpec
    runs_dir: Path
    resolved: dict[str, Any]


def list_presets() -> list[str]:
    folder = resources.files("lethe").joinpath("presets")
    return sorted(p.name[:-5] for p in folder.iterdir() if p.name.endswith(".toml"))


def load_preset(name: str) -> dict[str, Any]:
    if name not in list_presets():
        raise ConfigError(f"unknown preset {name!r}; available: {', '.join(list_presets())}")
    text = resources.files("lethe").joinpath("presets", f"{name}.toml").read_text()
    return tomllib.loads(text)


def load_config(
    path: Path, *, preset: str | None = None, overrides: Mapping[str, Any] | None = None
) -> RunConfig:
    path = Path(path).resolve()
    try:
        file_data = tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"config file not found: {path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path.name} is not valid TOML: {exc}") from exc
    overrides = dict(overrides or {})
    _check_keys(file_data, DEFAULTS, "")
    _check_keys(overrides, DEFAULTS, "")
    preset_name = preset or overrides.get("preset") or file_data.get("preset") or DEFAULTS["preset"]
    merged = _merge(copy.deepcopy(DEFAULTS), load_preset(preset_name))
    merged = _merge(merged, file_data)
    merged = _merge(merged, overrides)
    merged["preset"] = preset_name
    return _build(merged, path.parent)


def _check_keys(data: Mapping[str, Any], schema: Mapping[str, Any], where: str) -> None:
    unknown = sorted(set(data) - set(schema))
    if unknown:
        section = f"[{where}]" if where else "the top level"
        raise ConfigError(f"unknown key(s) in {section}: {', '.join(unknown)}")
    for key, value in data.items():
        # Only top-level sections are checked; nested tables like judge.env are user-defined.
        if where == "" and isinstance(schema[key], dict):
            if not isinstance(value, Mapping):
                raise ConfigError(f"[{key}] must be a table")
            _check_keys(value, schema[key], key)


def _merge(base: dict[str, Any], extra: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in extra.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict) and key != "env":
            base[key] = _merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def _build(cfg: dict[str, Any], config_dir: Path) -> RunConfig:
    task, judge, team = cfg["task"], cfg["judge"], cfg["team"]
    forget, agent, budget = cfg["forget"], cfg["agent"], cfg["budget"]
    return RunConfig(
        preset=cfg["preset"],
        task=_task(task, config_dir),
        judge=_judge(judge),
        team=_team(team),
        forget=_forget(forget),
        agent=_agent(agent),
        budget=_budget(budget),
        runs_dir=(config_dir / _str(cfg["run"], "runs_dir", "run")).resolve(),
        resolved=cfg,
    )


def _task(t: dict[str, Any], config_dir: Path) -> TaskConfig:
    name = _str(t, "name", "task")
    if not NAME_RE.match(name):
        raise ConfigError("task.name must match [A-Za-z0-9_.-]+")
    if (t["description"] is None) == (t["description_file"] is None):
        raise ConfigError("task: set exactly one of description or description_file")
    if t["description"] is not None:
        description = _str(t, "description", "task")
    else:
        desc_path = config_dir / _str(t, "description_file", "task")
        if not desc_path.is_file():
            raise ConfigError(f"task.description_file not found: {desc_path}")
        description = desc_path.read_text(encoding="utf-8")
    seed_dir = (config_dir / _str(t, "seed", "task")).resolve()
    if not seed_dir.is_dir():
        raise ConfigError(f"task.seed directory not found: {seed_dir}")
    editable = _globs(t["editable"], "task.editable")
    if not editable:
        raise ConfigError("task.editable must be a non-empty list of globs")
    exclude = _globs(t["exclude"], "task.exclude")
    return TaskConfig(
        name,
        description,
        config_dir.resolve(),
        seed_dir,
        editable,
        tuple(dict.fromkeys((*ALWAYS_EXCLUDE, *exclude))),
    )


def _judge(j: dict[str, Any]) -> JudgeSpec:
    run = _str(j, "run", "judge")
    if j["direction"] not in ("max", "min"):
        raise ConfigError("judge.direction must be 'max' or 'min'")
    env = j["env"]
    if not isinstance(env, Mapping) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in env.items()
    ):
        raise ConfigError("judge.env must map names to string values")
    return JudgeSpec(
        run=run,
        direction=j["direction"],
        timeout_s=_num(j, "timeout", "judge", minimum=0, strict=True),
        build=_str(j, "build", "judge", allow_empty=True) or None,
        concurrency=int(_num(j, "concurrency", "judge", minimum=1, integer=True)),
        cache=_bool(j, "cache", "judge"),
        max_feedback_chars=int(_num(j, "max_feedback_chars", "judge", minimum=100, integer=True)),
        split=_str(j, "split", "judge"),
        env=dict(env),
    )


def _team(t: dict[str, Any]) -> TeamConfig:
    agents = t["agents"]
    if not isinstance(agents, int) or isinstance(agents, bool) or not 1 <= agents <= 64:
        raise ConfigError("team.agents must be an integer between 1 and 64")
    if t["peers"] not in ("all", "none", "ring"):
        raise ConfigError("team.peers must be 'all', 'none' or 'ring'")
    if t["session"] not in ("continue", "fresh"):
        raise ConfigError("team.session must be 'continue' or 'fresh'")
    return TeamConfig(agents, t["peers"], t["session"])


def _forget(f: dict[str, Any]) -> ForgetConfig:
    if f["seed"] not in ("best", "hedged"):
        raise ConfigError("forget.seed must be 'best' or 'hedged'")
    when = f["when"]
    if not isinstance(when, list) or not all(isinstance(w, Mapping) for w in when):
        raise ConfigError("forget.when must be a list of trigger tables")
    for spec in when:
        try:
            trigger_from_dict(spec)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"forget.when: invalid trigger: {exc}") from exc
    return ForgetConfig(
        f["seed"], _bool(f, "carry_scratch", "forget"), tuple(dict(w) for w in when)
    )


def _agent(a: dict[str, Any]) -> AgentConfig:
    if a["backend"] != "claude-code":
        raise ConfigError(
            "agent.backend must be 'claude-code' (scripted backends come via the API)"
        )
    if a["effort"] not in EFFORTS:
        raise ConfigError(f"agent.effort must be one of {', '.join(repr(e) for e in EFFORTS)}")
    if a["isolation"] not in ("safe", "bare", "none"):
        raise ConfigError("agent.isolation must be 'safe', 'bare' or 'none'")
    extra = a["extra_args"]
    if not isinstance(extra, list) or not all(isinstance(x, str) for x in extra):
        raise ConfigError("agent.extra_args must be a list of strings")
    model = _str(a, "model", "agent", allow_empty=True)
    return AgentConfig(
        backend="claude-code",
        model=model or None,
        effort=a["effort"] or None,
        isolation=a["isolation"],
        shell=_bool(a, "shell", "agent"),
        turn_timeout=_num(a, "turn_timeout", "agent", minimum=0, strict=True),
        extra_args=tuple(extra),
    )


def _budget(b: dict[str, Any]) -> BudgetSpec:
    usd = _num(b, "usd", "budget", minimum=0)
    proposals = int(_num(b, "proposals", "budget", minimum=0, integer=True))
    wall = _num(b, "wall", "budget", minimum=0)
    if usd <= 0 and proposals <= 0 and wall <= 0:
        raise ConfigError("budget: at least one budget cap (usd, proposals, wall) must be > 0")
    primary = b["primary"]
    caps = {"usd": usd, "proposals": proposals, "wall": wall}
    if primary not in caps:
        raise ConfigError("budget.primary must be 'usd', 'proposals' or 'wall'")
    if caps[primary] <= 0:
        raise ConfigError(f"budget.primary is {primary!r} but budget.{primary} is 0")
    pause = _num(b, "pause_at_utilization", "budget", minimum=0, strict=True)
    if pause > 1:
        raise ConfigError("budget.pause_at_utilization must be in (0, 1]")
    return BudgetSpec(
        usd=usd, proposals=proposals, wall_s=wall, primary=primary, pause_at_utilization=pause
    )


def _str(d: Mapping[str, Any], key: str, section: str, *, allow_empty: bool = False) -> str:
    value = d.get(key)
    if value is None or not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise ConfigError(f"{section}.{key} is required and must be a non-empty string")
    return value


def _bool(d: Mapping[str, Any], key: str, section: str) -> bool:
    if not isinstance(d[key], bool):
        raise ConfigError(f"{section}.{key} must be true or false")
    return d[key]


def _num(
    d: Mapping[str, Any],
    key: str,
    section: str,
    *,
    minimum: float,
    strict: bool = False,
    integer: bool = False,
) -> float:
    value = d[key]
    ok = isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)
    if ok and integer and isinstance(value, float) and not value.is_integer():
        ok = False
    if not ok or value < minimum or (strict and value <= minimum):
        op = ">" if strict else ">="
        kind = "an integer" if integer else "a number"
        raise ConfigError(f"{section}.{key} must be {kind} {op} {minimum:g}")
    return float(value)


def _globs(value: Any, where: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ConfigError(f"{where} must be a list of glob strings")
    for pattern in value:
        try:
            glob_to_regex(pattern)
        except ValueError as exc:
            raise ConfigError(f"{where}: {exc}") from exc
    return tuple(value)
