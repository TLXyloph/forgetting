"""`lethe` command line: run, check, report, export, presets."""

from __future__ import annotations

import argparse
import sys
import tomllib
from importlib import resources
from pathlib import Path
from typing import Any

from lethe.api import check_task, load_run, run_archive, run_filters, run_task
from lethe.config import ConfigError, list_presets
from lethe.policy.prompts import format_score
from lethe.telemetry.live import format_event
from lethe.telemetry.report import build_report, format_report, write_csvs

FLAG_PATHS = {
    "agents": ("team", "agents"),
    "peers": ("team", "peers"),
    "session": ("team", "session"),
    "usd": ("budget", "usd"),
    "proposals": ("budget", "proposals"),
    "wall": ("budget", "wall"),
    "primary": ("budget", "primary"),
    "model": ("agent", "model"),
    "effort": ("agent", "effort"),
    "isolation": ("agent", "isolation"),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lethe",
        description="Autoresearch with forgetting: teams of coding agents that periodically "
        "discard their history and restart from the best candidate.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run a task (lethe.toml) to completion")
    run.add_argument("config", type=Path)
    run.add_argument("--preset", choices=list_presets())
    run.add_argument("--agents", type=int)
    run.add_argument("--peers", choices=("all", "none", "ring"))
    run.add_argument("--session", choices=("continue", "fresh"))
    run.add_argument("--usd", type=float, help="API-equivalent USD cap (default from config)")
    run.add_argument("--proposals", type=int, help="cap on agent turns (becomes the primary unit)")
    run.add_argument("--wall", type=float, help="wall-clock cap in seconds")
    run.add_argument("--primary", choices=("usd", "proposals", "wall"))
    run.add_argument("--model")
    run.add_argument("--effort", choices=("low", "medium", "high", "xhigh", "max"))
    run.add_argument("--isolation", choices=("safe", "bare", "none"))
    run.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="any config value, e.g. --set judge.timeout=30 (TOML value syntax)",
    )
    run.add_argument("--runs-dir", type=Path)
    run.add_argument("--name", help="run directory name (default: timestamp-preset-id)")
    run.add_argument("--quiet", action="store_true", help="no live progress lines")

    check = sub.add_parser("check", help="validate a task and measure its judge on the seed")
    check.add_argument("config", type=Path)
    check.add_argument("--repeats", type=int, default=3)

    report = sub.add_parser("report", help="summarize a run directory")
    report.add_argument("run_dir", type=Path)
    report.add_argument("--csv", type=Path, help="also write turns/best_curve/rounds CSVs here")

    export = sub.add_parser("export", help="write a candidate's files (default: the best)")
    export.add_argument("run_dir", type=Path)
    export.add_argument("--out", type=Path, required=True)
    export.add_argument("--candidate")
    export.add_argument("--judged-only", action="store_true", help="only the editable files")

    sub.add_parser("presets", help="list built-in presets")
    return parser


def _parse_value(text: str) -> Any:
    try:
        return tomllib.loads(f"v = {text}")["v"]
    except tomllib.TOMLDecodeError:
        return text


def overrides_from_args(args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {}

    def put(path: tuple[str, ...], value: Any) -> None:
        node = overrides
        for key in path[:-1]:
            node = node.setdefault(key, {})
        node[path[-1]] = value

    for flag, path in FLAG_PATHS.items():
        value = getattr(args, flag, None)
        if value is not None:
            put(path, value)
    if args.proposals is not None and args.primary is None:
        put(("budget", "primary"), "proposals")
    for item in args.set:
        key, sep, raw = item.partition("=")
        if not sep or "." not in key:
            raise ConfigError(f"--set expects SECTION.KEY=VALUE, got {item!r}")
        put(tuple(key.split(".")), _parse_value(raw))
    return overrides


def _cmd_run(args: argparse.Namespace) -> int:
    def progress(envelope, state):
        line = format_event(envelope, state)
        if line and not args.quiet:
            print(line, flush=True)

    try:
        run_dir = run_task(
            args.config,
            preset=args.preset,
            overrides=overrides_from_args(args),
            runs_dir=args.runs_dir,
            run_name=args.name,
            on_event=progress,
        )
    except KeyboardInterrupt:
        print("\ninterrupted; the partial run is kept in its run directory", file=sys.stderr)
        return 130
    print()
    print(format_report(build_report(load_run(run_dir))))
    print(f"\nrun directory: {run_dir}")
    return 0


def _cmd_check(args: argparse.Namespace) -> int:
    report = check_task(args.config, repeats=args.repeats)
    cfg = report.config
    print(
        f"task {cfg.task.name}: preset {cfg.preset}, {cfg.team.agents} agent(s), "
        f"peers {cfg.team.peers}, budget usd={cfg.budget.usd:g} proposals={cfg.budget.proposals}"
    )
    print(
        f"seed: {len(report.seed_files)} files, "
        f"editable: {', '.join(report.editable_matches) or '-'}"
    )
    for i, result in enumerate(report.results, 1):
        status = format_score(result.score) if result.valid else "INVALID"
        print(f"judge run {i}: {status} in {result.duration_s:.1f}s")
    print(f"claude: {report.claude_path or 'not found'} {report.claude_version or ''}".rstrip())
    for warning in report.warnings:
        print(f"warning: {warning}")
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    state = load_run(args.run_dir)
    print(format_report(build_report(state)))
    if args.csv:
        for path in write_csvs(state, args.csv):
            print(f"wrote {path}")
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    state = load_run(args.run_dir)
    cand = state.candidates.get(args.candidate) if args.candidate else state.best()
    if cand is None:
        print("no such candidate (or no valid best yet)", file=sys.stderr)
        return 1
    keep = run_filters(state)[0] if args.judged_only else None
    run_archive(args.run_dir, state).checkout(cand.sha, args.out, keep=keep)
    print(f"wrote {cand.cand_id} (score {format_score(cand.score)}) to {args.out}")
    return 0


def _cmd_presets(_: argparse.Namespace) -> int:
    for name in list_presets():
        text = resources.files("lethe").joinpath("presets", f"{name}.toml").read_text()
        first = text.splitlines()[0].lstrip("# ").strip() if text else ""
        print(f"{name:<12} {first}")
    return 0


COMMANDS = {
    "run": _cmd_run,
    "check": _cmd_check,
    "report": _cmd_report,
    "export": _cmd_export,
    "presets": _cmd_presets,
}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return COMMANDS[args.command](args)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
