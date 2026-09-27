#!/usr/bin/env python3
"""Sidecar monitor for slime coding-agent RL runs (MiMo-style ops telemetry).

Round1 post-mortem (2026-09-16): 2,162 rollouts produced 2,161 zero rewards and
40 zero-gradient train steps, yet nothing alarmed -- the wandb reward curve was
flat zero and the failure detail lived only in per-rollout log lines nobody
watched. Xiaomi's public MiMo RL dashboard treats ``dynsam/infra_error/seq_rate``
as a first-class metric for exactly this failure class. This sidecar is our
version, parsed from the trainer log with zero trainer-side changes.

Per train step it aggregates (from generate.py / model.py log lines):
  - pass rate and its delta vs step 1            (dynsam/avg@n analog)
  - agent-exit taxonomy: completed_fail / cli_error / time_budget / abort:* / pass
    and the infra fraction (everything except genuine completed_fail)   (infra_error analog)
  - per-repo rollout composition                 (batch composition analog)
  - train-actor health metrics (loss / grad_norm / entropy / kl / logprob diff)

Outputs under --out: steps.jsonl (machine), summary.md (human, regenerated per
flush), alerts.txt (append-only alert events). One-shot mode replays an
existing log (post-mortem / validation); --follow tails it live next to the
trainer. stdlib only; wandb push is optional and non-fatal.

Usage:
  .venv-train-rl/bin/python scripts/monitor_rl_run.py \
      --log logs/rl-round1-gspo.log --out runtime/agent-rl/round1-monitor [--follow]
Exit code (one-shot): 0 clean, 1 alerts fired (usable as a cron/gate check).
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# generate.py:259 info line -- one per finished rollout
RE_RESULT = re.compile(
    r"\[coding_agent_rl\] (?P<iid>[^:]+): reward=(?P<reward>[\d.]+) "
    r"applied=(?P<applied>True|False) agent_exit_code=(?P<exit>-?\d+) "
    r"elapsed=(?P<elapsed>[\d.]+)s segments=(?P<segments>\d+)\s*$")
# generate.py:332 warning line -- orchestrator-level aborts
RE_ABORT = re.compile(
    r"\[coding_agent_rl\] (?P<iid>[^:]+?) aborted: (?P<reason>\S+)")
# model.py:918 train-step metrics dict
RE_STEP = re.compile(r"model\.py:\d+ - step (?P<step>\d+): (?P<dict>\{.*\})\s*$")

# Alert thresholds (constants, not flags: tune here after evidence)
MIN_ROLLOUTS_FOR_RATE = 8      # a rate over fewer samples is noise
ALL_ZERO_STREAK_ALERT = 2      # consecutive all-zero-reward steps
INFRA_FRAC_ALERT = 0.80        # see name; fraction of non-genuine outcomes
ENTROPY_SPIKE = 2.0            # entropy_loss above this = divergence smell
KL_DRIFT = 1.0                 # ppo_kl above this = policy moving too fast


def repo_of(iid: str) -> str:
    """getmoto__moto-7647 -> getmoto/moto; falls back to the raw id."""
    if "__" in iid:
        owner, rest = iid.split("__", 1)
        return f"{owner}/{rest.rsplit('-', 1)[0]}"
    return iid


class Monitor:
    def __init__(self, args):
        self.args = args
        self.out = Path(args.out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.buffer: list[dict] = []       # rollouts since last step line
        self.steps: list[dict] = []        # emitted step records
        self.alerts: list[str] = []
        self.zero_streak = 0
        self.infra_armed = True   # edge-triggered alert latches
        self.grad0_armed = True
        self.first_rate: float | None = None
        self.wandb = None
        if args.wandb:
            try:
                import wandb  # noqa: optional, from the train venv
                self.wandb = wandb.init(
                    project=args.wandb_project or "code-agent-dense-mainline",
                    group=(args.wandb_group or "stage2-gspo-round1") + "-sidecar",
                    job_type="sidecar",
                    config={"log": str(args.log), "note": "sidecar parsed from trainer log"})
            except Exception as exc:  # wandb is a convenience, never a dependency
                print(f"[monitor] wandb disabled ({type(exc).__name__}: {exc})",
                      file=sys.stderr)
                self.wandb = None

    # ------------------------------------------------------------- parsing
    def feed_line(self, line: str) -> None:
        m = RE_RESULT.search(line)
        if m:
            self.buffer.append({
                "iid": m["iid"], "reward": float(m["reward"]),
                "applied": m["applied"] == "True", "exit": int(m["exit"]),
                "elapsed": float(m["elapsed"]), "segments": int(m["segments"])})
            return
        m = RE_ABORT.search(line)
        if m:
            self.buffer.append({
                "iid": m["iid"], "reward": 0.0, "applied": False,
                "abort": m["reason"], "exit": None,
                "elapsed": None, "segments": None})
            return
        m = RE_STEP.search(line)
        if m:
            try:
                train = {k: v for k, v in ast.literal_eval(m["dict"]).items()
                         if isinstance(v, (int, float))}
            except Exception:
                train = {}
            self.emit_step(int(m["step"]), train)

    # ---------------------------------------------------------- aggregation
    def emit_step(self, step: int, train: dict) -> None:
        rolls = self.buffer
        self.buffer = []
        n = len(rolls)
        passes = sum(1 for r in rolls if r["reward"] >= 1.0)
        tax = Counter(self.outcome(r) for r in rolls)
        repos: dict[str, dict] = {}
        for r in rolls:
            repo = repo_of(r["iid"])
            d = repos.setdefault(repo, {"n": 0, "pass": 0})
            d["n"] += 1
            d["pass"] += 1 if r["reward"] >= 1.0 else 0
        genuine = tax["pass"] + tax["completed_fail"]
        infra_frac = (1.0 - genuine / n) if n else 0.0
        rate = passes / n if n else 0.0
        if self.first_rate is None and n >= MIN_ROLLOUTS_FOR_RATE:
            self.first_rate = rate
        rec = {
            "step": step, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "n_rollouts": n, "n_pass": passes, "pass_rate": round(rate, 4),
            "pass_rate_delta_vs_step1": (round(rate - self.first_rate, 4)
                                         if self.first_rate is not None else None),
            "outcomes": dict(sorted(tax.items())),
            "infra_frac": round(infra_frac, 4),
            "repos": repos,
            "elapsed_mean_s": (round(sum(r["elapsed"] for r in rolls if r["elapsed"]) /
                                     max(1, sum(1 for r in rolls if r["elapsed"])), 1)
                               if any(r["elapsed"] for r in rolls) else None),
            "train": train,
        }
        self.steps.append(rec)
        with (self.out / "steps.jsonl").open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.check_alerts(rec)
        self.write_summary()
        if self.wandb:
            flat = {f"sidecar/{k}": v for k, v in rec.items()
                    if isinstance(v, (int, float))}
            flat["sidecar/infra_error_rate"] = infra_frac
            self.wandb.log(flatten_repo_counts(repos) | flat, step=step)

    @staticmethod
    def outcome(r: dict) -> str:
        """Taxonomy documented, not guessed: reward>=1 wins; else orchestrator
        aborts by reason; else exit 0 = genuine completed-but-failed; exit 1
        bundles CLI errors (round1 evidence: mostly max-tokens budget deaths);
        exit<0 = the harness's own time budget."""
        if r["reward"] >= 1.0:
            return "pass"
        if "abort" in r:
            return f"abort:{r['abort'].split(':')[0]}"
        if r["exit"] == 0:
            return "completed_fail"
        if r["exit"] == 1:
            return "cli_error"
        return "time_budget"

    # --------------------------------------------------------------- alerts
    def check_alerts(self, rec: dict) -> None:
        """Edge-triggered: a condition fires when it (re)arms and trips, not
        on every step it persists -- a level-triggered alert at 3 alerts/step
        over a 100-step run is 300 lines of noise nobody reads again."""
        def fire(msg: str) -> None:
            stamp = f"[{rec['utc']}] step {rec['step']}: {msg}"
            self.alerts.append(stamp)
            with (self.out / "alerts.txt").open("a") as f:
                f.write(stamp + "\n")
            print(f"[monitor][ALERT] {stamp}", file=sys.stderr, flush=True)

        n, passes = rec["n_rollouts"], rec["n_pass"]
        train = rec.get("train", {})
        if n >= MIN_ROLLOUTS_FOR_RATE:
            if passes == 0:
                self.zero_streak += 1
                if self.zero_streak == ALL_ZERO_STREAK_ALERT:
                    fire(f"all-zero rewards for {self.zero_streak} consecutive steps "
                         f"(n={n}) -- round1 incident signature, investigate before "
                         f"burning more compute")
            else:
                self.zero_streak = 0
            if rec["infra_frac"] >= INFRA_FRAC_ALERT and self.infra_armed:
                fire(f"infra outcome fraction {rec['infra_frac']:.2f} "
                     f"(tax={rec['outcomes']}) -- rollout path unhealthy")
                self.infra_armed = False
            elif rec["infra_frac"] < INFRA_FRAC_ALERT * 0.75:
                self.infra_armed = True
        loss, grad = train.get("train/loss"), train.get("train/grad_norm")
        if (isinstance(loss, float) and loss != loss) or \
           (isinstance(grad, float) and grad != grad):
            fire("NaN in train/loss or train/grad_norm")
        if isinstance(grad, (int, float)) and grad == 0 and self.grad0_armed:
            fire("train/grad_norm == 0 -- zero advantage (all-zero rewards?) "
                 "or broken graph")
            self.grad0_armed = False
        elif isinstance(grad, (int, float)) and grad != 0:
            self.grad0_armed = True
        ent = train.get("train/entropy_loss")
        if isinstance(ent, (int, float)) and ent > ENTROPY_SPIKE:
            fire(f"entropy_loss {ent:.2f} above {ENTROPY_SPIKE} -- divergence smell")
        kl = train.get("train/ppo_kl")
        if isinstance(kl, (int, float)) and kl > KL_DRIFT:
            fire(f"ppo_kl {kl:.2f} above {KL_DRIFT} -- policy drifting fast")

    # --------------------------------------------------------------- output
    def write_summary(self) -> None:
        lines = ["# RL run sidecar summary", "",
                 f"- log: `{self.args.log}`",
                 f"- updated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
                 f"- steps parsed: {len(self.steps)}",
                 f"- alerts: **{len(self.alerts)}**", ""]
        if self.alerts:
            lines += ["## Alerts (latest 10)", ""]
            lines += [f"- {a}" for a in self.alerts[-10:]] + [""]
        lines += ["## Per-step", "",
                  "| step | n | pass | rate | Δvs1 | exit0f | cli | tb | abort | infra% | grad | loss |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in self.steps[-40:]:
            o = r["outcomes"]
            t = r["train"]
            lines.append(
                f"| {r['step']} | {r['n_rollouts']} | {r['n_pass']} | "
                f"{r['pass_rate']:.3f} | {fmt(r['pass_rate_delta_vs_step1'])} | "
                f"{o.get('completed_fail', 0)} | {o.get('cli_error', 0)} | "
                f"{o.get('time_budget', 0)} | "
                f"{sum(v for k, v in o.items() if k.startswith('abort:'))} | "
                f"{r['infra_frac']:.0%} | {fmt(t.get('train/grad_norm'))} | "
                f"{fmt(t.get('train/loss'))} |")
        tot = Counter()
        for r in self.steps:
            tot.update(r["outcomes"])
        n_all = sum(tot.values())
        n_pass_all = sum(r["n_pass"] for r in self.steps)
        lines += ["", "## Cumulative", "",
                  f"- rollouts: {n_all} | passes: {n_pass_all} "
                  f"({n_pass_all / n_all:.1%})" if n_all else "- rollouts: 0", ""]
        repos: Counter = Counter()
        repos_pass: Counter = Counter()
        for r in self.steps:
            for repo, d in r["repos"].items():
                repos[repo] += d["n"]
                repos_pass[repo] += d["pass"]
        if repos:
            lines += ["## Repo composition (cumulative)", "",
                      "| repo | rollouts | pass | rate |", "|---|---|---|---|"]
            for repo, n in repos.most_common():
                lines.append(f"| {repo} | {n} | {repos_pass[repo]} | "
                             f"{repos_pass[repo] / n:.1%} |")
        (self.out / "summary.md").write_text("\n".join(lines) + "\n")

    def finish(self) -> int:
        if self.buffer:  # rollouts after the last step line (run cut mid-batch)
            self.emit_step(-1, {})
            self.steps[-1]["note"] = "trailing rollouts before any step line"
        if self.wandb:
            self.wandb.finish()
        return 1 if self.alerts else 0


def flatten_repo_counts(repos: dict) -> dict:
    return {f"sidecar/repo/{repo}": d["n"] for repo, d in repos.items()}


def fmt(v) -> str:
    return "—" if v is None else f"{v:.3g}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--log", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=ROOT / "runtime/agent-rl/round1-monitor")
    ap.add_argument("--follow", action="store_true",
                    help="keep tailing after EOF (live mode)")
    ap.add_argument("--wandb", action="store_true", help="optional wandb push")
    ap.add_argument("--wandb-project", default=None)
    ap.add_argument("--wandb-group", default=None)
    args = ap.parse_args()
    mon = Monitor(args)
    with args.log.open("r", errors="replace") as f:
        while True:  # --follow parses existing content first, then tails
            line = f.readline()
            if line:
                mon.feed_line(line)
                continue
            if not args.follow:
                break
            time.sleep(1.0)
    sys.exit(mon.finish())


if __name__ == "__main__":
    main()
