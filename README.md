# lethe: autoresearch with forgetting

lethe runs a team of coding agents (headless Claude Code) that iteratively improve a candidate
against an objective judge, and periodically **forgets**: every agent's session and workspace is
discarded and the team restarts, fresh, from the best candidate found so far.

It implements the method of [Tan et al., *Scaling Discovery Requires Forgetting* (arXiv
2610.20563, 2026)](https://arxiv.org/abs/2610.20563) as a general, configurable harness that runs
on a Claude **subscription**. Any task works if a command can score a candidate: heuristic
programs, kernels, performance work, configs, regexes and other text artifacts.

> Status: research prototype (v0.1). The `research/` package reproduces the paper's experiments
> and tests forgetting policies beyond it.

## Why forgetting

As an agent's history grows it anchors on its own past attempts: its edits shrink, its reasoning
gets shorter, and it starts declaring "practical limits" that later turn out to be far from the
real limit. In the paper, running one agent longer, running agents in parallel, and letting them
share candidates all plateau. Restarting fresh agents from the best candidate unblocks the search
at the same budget:

| Mean best (3 runs, same $) | AHC014 ↑ | VLIW cycles ↓ | Frontier-CS Q171 ↑ | nanochat val BPB ↓ |
| --- | --- | --- | --- | --- |
| 1 agent | 795k | 1,426 | 64.0 | 0.980 |
| 4 parallel agents | 964k | 1,220 | 70.2 | 0.980 |
| team of 4 + forgetting | **1,103k** | **1,097** | **80.8** | **0.961** |

<sub>Numbers from the paper's Table 1 (Pi agent + DeepSeek V4.1 Flash).</sub>

## How it works

```
             +-------------- control loop (plain code, never an LLM) ----------------+
 lethe.toml  | judge the seed (c0000) -> round 1: N agents in parallel -> forget ->   |
 seed/  ---> | round 2 from the best -> ... until the budget is spent                 |
 judge/      |                                                                        |
             | each turn: reducer builds the agent's view (TASK.md + .context/)       |
             |   -> `claude -p` (one turn) -> snapshot into the archive -> judge      |
             |   -> events.jsonl                                                      |
             +------------------------------------------------------------------------+
```

- **Event-sourced.** Every state change is appended to `events.jsonl`, and the run state is a pure
  fold over it. Runs can be replayed and analyzed, and forgetting never deletes anything: it starts
  a new *epoch*, with a fresh session and a fresh workspace seeded from the best candidate.
- **Every strategy is configuration.** The paper's three configurations and team-with-forgetting are
  presets of one reducer, which decides what each agent sees and whether its session continues.
- **Agents see only their workspace.** That workspace holds `TASK.md`, their current candidate and
  `.context/`: the judge's feedback, their lineage, and peers' pools and best candidates.
- **Judging is isolated.** Each candidate is scored on a pristine copy of the seed with only its
  editable files overlaid, so edits to tests or simulators can't change the score.

## Install

Requirements: Python 3.11+, git, [uv](https://docs.astral.sh/uv/), and the Claude Code CLI (`claude`)
logged in with your subscription.

```bash
git clone git@github.com:TLXyloph/forgetting.git && cd forgetting
uv sync --all-packages --all-extras      # lethe + research (+ matplotlib for plots)
uv run lethe presets
```

## Quickstart

```bash
# 1. Validate a task: config, seed files, judge on the seed (noise), claude CLI
uv run lethe check lethe/examples/bin-packing/lethe.toml

# 2. Run it: 2 agents, 8 proposals, one team forget at half the budget
uv run lethe run lethe/examples/bin-packing/lethe.toml --agents 2 --proposals 8 --model haiku \
    --set 'forget.when=[{type="budget_fractions", fractions=[0.5]}]'

# 3. Inspect and export
uv run lethe report .lethe/runs/<run-id> --csv out/
uv run lethe export .lethe/runs/<run-id> --out best/ --judged-only
```

A run directory holds `events.jsonl` (the log), `archive.git` (every candidate as a commit),
`ws/<agent>/e<epoch>/` (agent workspaces) and `transcripts/` (raw agent output per turn).
Run directories default to `.lethe/runs/` next to the task file; add `.lethe/` to your `.gitignore`.

## Write your own task

```
my-task/
  lethe.toml       # task + judge (+ optional team/forget/agent/budget overrides)
  seed/            # the starting candidate: the files agents begin from
  judge/score.py   # hidden from agents; reached via $LETHE_TASK_DIR
```

```toml
[task]
name = "my-task"
description = """Rules, input/output format and objective. Give no search strategy."""
seed = "seed"
editable = ["solver.py"]          # globs of judged files; `**` spans directories

[judge]
run = 'python3 "$LETHE_TASK_DIR/judge/score.py"'   # cwd = pristine seed + candidate's editable files
direction = "max"                 # or "min"
timeout = 300
```

**Judge contract.** The last non-empty stdout line is either a bare number or a JSON object:
`{"score": 12.5, "valid": true, "feedback": "shown to agents", "metrics": {...}, "cases": {"t1": 3.0}}`.
- Use `"valid": false` (and omit the score) to reject a candidate.
- Exit code 75 means the judge itself broke; it is retried once.
- Any other nonzero exit makes the candidate invalid.
- `judge.output = "file"` switches to reading `$LETHE_OUT/result.json` instead.

Judges get `LETHE_TASK_DIR`, `LETHE_CANDIDATE`, `LETHE_SPLIT`, `LETHE_SEED`, `LETHE_CAND_ID`, an
allowlisted environment and anything in `judge.env`.

Keep judges deterministic. Run candidate code in a subprocess with a timeout and a minimal
environment (`PYTHONHASHSEED=0` for Python), and don't put hidden test data in feedback.

## Configuration

Precedence: built-in defaults < preset < `lethe.toml` < CLI flags (`--set section.key=value`).
Unknown keys are errors.

| Section | Key (default) | Meaning |
| --- | --- | --- |
| `team` | `agents` (4), `peers` (`all`/`none`/`ring`), `session` (`continue`/`fresh`) | who sees whose candidates; whether sessions continue between turns |
| `forget` | `when` ([] = never), `seed` (`best`/`hedged`), `carry_scratch` (true) | triggers, what each agent restarts from, whether notes left in the best candidate survive |
| `agent` | `model` (`sonnet`), `effort`, `isolation` (`safe`), `shell` (false: a sandboxed Bash when true), `turn_timeout` (1800), `allow_api_billing` (false), `extra_args` | the Claude Code agent |
| `judge` | `run`, `direction`, `timeout` (600), `build`, `concurrency` (2), `cache` (true), `output` (`stdout`), `env` | scoring |
| `budget` | `usd` (5.0), `proposals` (0), `wall` (0), `primary` (`usd`), `pause_at_utilization` (0.95), `pause_at_weekly_utilization` (1.0) | caps: the first one reached stops the run; pause thresholds for the 5-hour and 7-day windows |

**Presets** (`lethe presets`): `solo` (1 agent), `parallel` (4 independent agents), `team` (4 agents
that see each other's candidates), and `team-forget` (the paper: a team of 4 forgetting at 25/50/75%
of the budget; the default).

**Forget triggers** (`forget.when`, any of):

| Trigger | Scope | Fires when |
| --- | --- | --- |
| `budget_fractions` | team | spend crosses each fraction (the paper's schedule) |
| `plateau` | team | the team best hasn't improved by `min_gain` in the last `window` proposals |
| `anchored` | team | at least `quorum` of agents' recent edit sizes have shrunk to ≤ `ratio` × their session's opening size, and the team is stalled |
| `stalled` | one agent | `n` of the agent's last `of` turns were empty or claimed a plateau ("practical limit", ...) |

## Budgets and the subscription meter

- `budget.usd` is **API-equivalent USD**. Claude Code reports every turn's cost at API list price
  even on a subscription, so this is a consistent compute meter, not your bill.
- `budget.proposals` counts agent turns, and `budget.wall` is seconds.
- `primary` picks the unit that forget schedules use.
- Every turn also logs the subscription's **5-hour and 7-day window utilization**. All agents pause
  at `pause_at_utilization` of the 5-hour window and resume after it resets.
- `pause_at_weekly_utilization` does the same for the 7-day window. Set it below 1 (say 0.8) on long
  runs to keep part of the week's allowance for other work. Pausing never changes results, only
  wall-clock time.
- `lethe report` shows spend, proposals and the latest window reading.

## Isolation and trust model

- **`isolation = "safe"`** (default, works with a subscription). Agents run with
  `claude -p --safe-mode --restricted --disable-slash-commands --tools Read,Edit,Write,Glob,Grep`:
  - no CLAUDE.md, hooks, plugins, MCP servers or skills;
  - file tools confined to the workspace, and no shell by default;
  - API-key and provider variables are removed from the agent's environment, and a turn billed to an
    API key stops the run unless `agent.allow_api_billing = true`;
  - each workspace is its own empty git repo, so Claude Code never shows agents the enclosing
    repository's history.
- **`agent.shell = true`** adds Bash, always inside Claude Code's sandbox:
  - commands are approved automatically, write only inside the workspace and have no network;
  - they can still read files elsewhere.

  Headless, an unsandboxed Bash would need an approval that no one is there to give.
- **`bare`** uses `--bare` and needs `ANTHROPIC_API_KEY`. **`none`** adds no isolation.
- **No sandbox for judged code.** Forgetting is enforced on the agents' tools, not on code the judge
  executes. Candidate code runs with your user permissions, so treat tasks like any code you run
  locally. `lethe check` and `lethe run` print this warning.

## Research package

`research/` reproduces the paper and tests new forgetting policies:

```bash
research/scripts/fetch_vliw.sh       # Anthropic's performance take-home (pinned commit)
research/scripts/fetch_ahc014.sh     # AtCoder AHC014 tools (checksum-pinned; builds with cargo)

uv run python -m research.matrix research/config/experiments/paper-vliw.toml   # arms × replicates
uv run python -m research.analyze research/runs/paper-vliw --compare team-forget:parallel --plots
uv run python -m research.bandit research/config/experiments/bets-toy-mis.toml --pulls 12
```

- **Tasks:** `toy-mis` (cheap development task), `vliw` and `ahc014` (two of the paper's four tasks).
- **Experiments** (`research/config/experiments/`): the paper's three configurations and a set of
  "bets": anchored, stalled, hedged seeding, clean forget, fresh-every-turn and plateau.
- **Analysis:** the paper's Tables 1 and 2, best-vs-spend curves, anchoring traces (lines changed and
  thinking tokens per spend bin), Welch t-tests and bootstrap CIs.

Third-party material is fetched at setup and never committed:
- The VLIW take-home has no license and asks that solutions not be published.
- AHC014 tools are AtCoder's.

## Development

```bash
uv run pytest            # ~1,300 tests
uv run ruff check . && uv run ruff format --check .
```

Source layout: `lethe/src/lethe/{domain,policy,runtime,judge,workspace,agents,telemetry}` and
`research/src/research/`. Stdlib-only at runtime.

## Acknowledgements

Method: Tan, Zhou, Agrawal, Zhang, Chen, Mang, Lee, Jacob, Sen, Dimakis, Zaharia, *Scaling Discovery
Requires Forgetting*, arXiv 2610.20563 (2026). Tasks: [Anthropic's original performance
take-home](https://github.com/anthropics/original_performance_takehome) and [AtCoder Heuristic
Contest 014](https://atcoder.jp/contests/ahc014).
