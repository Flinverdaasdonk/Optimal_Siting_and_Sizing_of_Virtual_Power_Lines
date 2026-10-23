"""Readers shared by the DF_with_w_current notebooks: the LDF -> DistFlow (SOCP) -> pgm study.

The runs come from slurm_jobs/DF_wc003_experiments/ -- exp1 through its slurm scripts
(exp1_validate_two_stage.slurm + packed_long_runs.slurm for the 24 h single-stage cells), exp2
and exp3 through experiments/DF_wc003_server_sweep.py (`--which even` / `--which odd` on two
servers, the same _lib.sh underneath). Both lay the runs out as

    <sweep root>/job_<slurm id | host_stamp>/{single_DF,stage1_LDF,stage2_DF,baseline}/exp_<cell>_<tag>_<stamp>/

`load(root)` globs every run folder under a sweep root (any job, smoke runs skipped) and returns
one row per run: the config knobs that define the cell, the solve as it ended (results.json) and
the pgm re-pricing of the dispatch (post_solve_pgm_objective/metrics.json). A run that is still
going has no results.json yet: it is kept, with `finished = False` and NaN numbers.

THE COMPARISON COLUMNS ARE THE CONGESTION TERM, NOT THE TOTAL. The DF block prices the squared
current excess (`current_congestion_objective`) plus a flat `w_current * sum(ell)` (`w_loss` is 0
in these runs, so the cone is priced by `w_current` alone), and the pgm phase prices both terms at
the AC state of the same dispatch, on the shunt-free grid (`pgm_grid_capacitances_zeroed`). So

    objective      the model's current_congestion_objective (stage 1, LDF: its total)
    pgm_objective  pgm's current_congestion_objective -- the real target
    objective_total / pgm_total   the full objectives, w_current term included
    bound          the solver's bound, on the TOTAL (compare it with objective_total only)
    model_<term> / pgm_<term>     every component, as each side reports it

Deliberately not `load_runs`: that globs one level only, and flattens the pgm `evaluations`
list into an opaque tuple.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

PGM_DIRNAME = "post_solve_pgm_objective"
_EXP_ID = re.compile(r"^exp_(\d+)_")
CONGESTION_TERM = "current_congestion_objective"
GRID_PAIRS = ("C0,S0", "S0,R0", "R0,C0")   # the cell order of every sweep here


def grid_pairs(used):
    """The notebooks' USED_GRIDS (["C0S0", ...], commas optional) as `grids` column values, in
    GRID_PAIRS order. Refuses a name that is not one of GRID_PAIRS."""
    by_name = {g.replace(",", ""): g for g in GRID_PAIRS}
    names = [u.replace(",", "") for u in used]
    unknown = [u for u in names if u not in by_name]
    if unknown or not names:
        raise ValueError(f"USED_GRIDS must be a non-empty subset of {list(by_name)}, not {list(used)!r}")
    return tuple(g for g in GRID_PAIRS if g.replace(",", "") in names)

# run folder's parent -> the phase name the notebooks use
PHASES = {"single_DF": "single", "single_LDF_S": "single",
          "stage1_LDF": "stage1",
          "stage2_DF": "stage2", "stage2_LDF_S": "stage2",
          "baseline": "baseline"}

# The notebooks' "which objective" toggle: "model" (DF, as solved), "pgm" (AC re-pricing) or "both".
OBJECTIVE_MODES = {"pgm": ("pgm_objective",), "model": ("objective",), "both": ("pgm_objective", "objective")}


def objective_columns(mode):
    if mode not in OBJECTIVE_MODES:
        raise ValueError(f"PLOT_OBJECTIVE must be one of {list(OBJECTIVE_MODES)}, not {mode!r}")
    return OBJECTIVE_MODES[mode]


# The notebooks' "which term" toggle: "total" (the full objective, w_current * sum(ell) included)
# or "congestion" (current_congestion_objective alone).
OBJECTIVE_TERMS = ("total", "congestion")


def select_term(df, term):
    """`df` with `objective` / `pgm_objective` holding `term`, so the code downstream is term-agnostic."""
    if term not in OBJECTIVE_TERMS:
        raise ValueError(f"OBJECTIVE_TERM must be one of {list(OBJECTIVE_TERMS)}, not {term!r}")
    if term == "congestion":
        return df
    return df.assign(objective=df["objective_total"], pgm_objective=df["pgm_total"])


def _read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _read_kv(path):
    """`stage1_run.txt`: one `key: value` per line, as run_two_stage writes it."""
    out = {}
    try:
        for line in Path(path).read_text().splitlines():
            if ": " in line:
                k, v = line.split(": ", 1)
                out[k.strip()] = v.strip()
    except OSError:
        pass
    return out


def find_runs(root):
    """Every run folder under `root`, smoke runs excluded."""
    root = Path(root)
    return sorted(p for p in root.glob("**/exp_*")
                  if p.is_dir() and (p / ".hydra" / "config.yaml").is_file()
                  and not any(part.startswith("smoke_") for part in p.relative_to(root).parts))


def read_pgm(run_dir):
    """The pgm re-pricing of the extensive solve's dispatch, or the reason there is none."""
    missing = {"pgm_objective": np.nan, "pgm_total": np.nan}
    metrics = _read_json(Path(run_dir) / PGM_DIRNAME / "metrics.json")
    if not metrics:
        return {**missing, "pgm_note": "no metrics.json"}
    zeroed = (metrics.get("instance") or {}).get("pgm_grid_capacitances_zeroed")
    for ev in metrics.get("evaluations", []):
        if ev.get("siting_source") != "extensive":
            continue
        pgm = ev.get("pgm_objective")
        if not pgm:
            return {**missing, "pgm_note": ev.get("skipped", "not priced")}
        comps = pgm.get("objective_components", {})
        row = {"pgm_total": float(pgm["total"]),
               "pgm_objective": float(comps.get(CONGESTION_TERM, np.nan)),
               "pgm_note": None,
               "pgm_capacitances_zeroed": zeroed}
        row.update({f"pgm_{k}": float(v) for k, v in comps.items()})
        return row
    return {**missing, "pgm_note": "no extensive evaluation"}


def read_run(run_dir):
    run_dir = Path(run_dir)
    cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
    res = _read_json(run_dir / "results.json")
    finished = res is not None
    res = res or {}
    gamma = {int(k): int(v) for k, v in ((res.get("gamma_n") or {}).get("per_node") or {}).items()}
    m = _EXP_ID.match(run_dir.name)
    C_base = float(cfg.objective.C_base)
    units = int(cfg.objective.max_units_total)
    comps = ((res.get("solves") or {}).get("extensive") or {}).get("objective_components") or {}
    total = res.get("objective")
    row = {
        "run": run_dir.name,
        "run_dir": str(run_dir),
        "phase": PHASES.get(run_dir.parent.name, run_dir.parent.name),   # single | stage1 | stage2 | baseline
        "job": run_dir.parent.parent.name,
        "task": int(m.group(1)) if m else None,
        "experiment": str(cfg.experiment.name),
        "physics": str(cfg.physics.name),
        "objective_name": str(cfg.objective.name),
        "grids": ",".join(cfg.solve.grid_names),
        "days": int(cfg.solve.days),
        "C_base": C_base,
        "units": units,
        "C_total": C_base * units / 1000,      # MWh
        "vcc": bool(cfg.solve.block.use_virtual_cable_constraint),
        "w_current": float(cfg.objective.weights.get("w_current", np.nan) or 0.0),
        "facets": str(cfg.physics.get("apparent_power_facets", None)),
        "not_candidate_set": str(list(cfg.solve.not_candidate_set or [])),
        "candidate_set": str(list(cfg.solve.candidate_set or [])),
        "pinned": cfg.solve.fix_gamma_n is not None,
        "finished": finished,
        "status": res.get("termination_reason"),
        "error": res.get("error"),
        # the MODEL's congestion term (stage 1, which has none, keeps its total)
        "objective": comps.get(CONGESTION_TERM, total),
        "objective_total": total,
        "bound": res.get("bound"),             # on the TOTAL
        "mip_gap": res.get("mip_gap"),
        "wall_time_s": res.get("wall_time_s"),
        "gamma_n": gamma,
        "n_sites": len(gamma),
        "units_placed": sum(gamma.values()),
    }
    row.update({f"model_{k}": float(v) for k, v in comps.items()})
    row.update(read_pgm(run_dir))
    s1 = _read_kv(run_dir / "stage1_run.txt")
    row["stage1_run"] = Path(s1["stage1_run_dir"].replace("\\", "/")).name if "stage1_run_dir" in s1 else None
    row["two_stage_wall_s"] = float(s1["two_stage_wall_s"]) if "two_stage_wall_s" in s1 else np.nan
    return row


def load(root, grids=None):
    """One row per run under a sweep root (every job folder, smoke runs skipped).
    `grids` (e.g. `grid_pairs(USED_GRIDS)`) keeps only those grid pairs; None keeps all."""
    rows = [read_run(p) for p in find_runs(root)]
    if not rows:
        raise FileNotFoundError(f"no run folders under {root}")
    df = pd.DataFrame(rows)
    if grids is not None:
        df = df[df["grids"].isin(grids)].reset_index(drop=True)
        if df.empty:
            raise FileNotFoundError(f"no runs of {list(grids)} under {root}")
    for col in ("objective", "objective_total", "bound", "mip_gap", "wall_time_s"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    running = df[~df["finished"]]
    if len(running):
        print(f"NOTE: {len(running)} run(s) without results.json (still running, or killed):")
        print(running[["phase", "run"]].to_string(index=False))
    unpriced = df[df["finished"] & (df["phase"] != "stage1") & df["pgm_objective"].isna()]
    if len(unpriced):
        print(f"WARNING: {len(unpriced)} DF run(s) without a pgm objective:")
        print(unpriced[["run", "status", "pgm_note", "error"]].to_string(index=False))
    return df


def one_per(frame, key):
    """Latest run per key (by folder name, which ends in the run stamp), with a warning on repeats."""
    frame = frame.sort_values("run")
    dup = frame.duplicated(key, keep="last")
    if dup.any():
        print(f"WARNING: {dup.sum()} repeated run(s); keeping the latest of",
              frame.loc[dup, key].drop_duplicates().values.tolist())
    return frame[~dup].set_index(key)


def two_stage(df):
    """Stage-2 rows with their stage-1 run joined on (prefix s1_).

    `time_s` is the summed SOLVE wall clock of both stages (what `single` is compared on);
    `two_stage_wall_s` is the script's end to end, builds and pgm included.
    """
    s1 = (df[df["phase"] == "stage1"]
          .set_index("run")[["status", "objective", "mip_gap", "wall_time_s", "gamma_n", "n_sites"]]
          .add_prefix("s1_"))
    s2 = df[df["phase"] == "stage2"].join(s1, on="stage1_run")
    missing = s2["s1_wall_time_s"].isna()
    if missing.any():
        print(f"WARNING: {missing.sum()} stage-2 run(s) whose stage-1 run is not under this root")
    s2["time_s"] = s2["s1_wall_time_s"] + s2["wall_time_s"]
    return s2


def experiment_mode(name):
    """exp3's arm, read off experiment.name (..._<mode>)."""
    for mode in ("cess_plus_slack", "no_slack"):
        if name.endswith(mode):
            return mode
    return None


def units_at(gamma, positions):
    """Units of a {position: units} siting at the given positions."""
    return sum(u for k, u in gamma.items() if k in set(positions))


def curves_frame(curves, names, reference_MWh=None):
    """A figure's curves as numbers: one column per curve, one row per C_total [MWh].

    `curves` maps a legend key (a tuple, one entry per level in `names`) to the Series the plot drew
    (index C_total, values y). A curve without a point at some C_total is NaN there. With
    `reference_MWh` the secondary x-axis (C_total as % of it) is a second index level.
    """
    df = pd.concat(curves, axis=1, names=list(names)).sort_index()
    x = "C_total [MWh]"
    if reference_MWh is None:
        df.index.name = x
    else:
        df.index = pd.MultiIndex.from_arrays([df.index, 100 * df.index / reference_MWh],
                                             names=[x, "C_total [% of avg daily demand]"])
    return df


@lru_cache(maxsize=None)
def avg_daily_abs_demand_MWh(grids, days, every_nth=4):
    """sum_n |p_n| over the horizon, per day [MWh]: the secondary x-axis' reference.

    The hourly profiles (`every_nth=4` of the 15-min data), so each row is 1 h. This is how
    v5/04b's AVG_ABS_FAILY_GRID_DEMAND_KWH = 377697.34 was made: it is [C0,S0] x 4 days.
    `grids` is the comma-joined `grids` column ("C0,S0").
    """
    from aoss.utilities import grid_utilities as grut
    _, p, _ = grut.get_grids_and_profiles(grid_names=grids.split(","), timespan=f"{days}_days",
                                          every_nth=every_nth, ensure_radial_grid=True,
                                          merge_slackbusses=False)
    hours_per_row = every_nth / 4
    return float(np.abs(p).sum() * hours_per_row / days / 1000)
