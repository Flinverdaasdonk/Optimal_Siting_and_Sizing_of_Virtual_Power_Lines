"""Site and size storage on C0+S0 (4 days): the two-stage procedure of the paper.

    stage 1  LinDistFlow, all candidate nodes           -> the sites
    stage 2  DistFlow, only the stage-1 sites, with the stage-1 siting as MIP start
    each DistFlow dispatch is then re-priced under AC power flow (power-grid-model)

Examples (from the paper_companion folder):
    python scripts/demo.py --C-total-MWh 5 --gamma 10 --storage vpl
    python scripts/demo.py --C-total-MWh 5 --gamma 10 --storage cess
    python scripts/demo.py --C-total-MWh 5 --gamma 10 --storage vpl --variant no_slack
    python scripts/demo.py --C-total-MWh 5 --gamma 10 --storage vpl --variant cess_plus_slack
    python scripts/demo.py --baseline                      # DistFlow without storage
    python scripts/demo.py --C-total-MWh 5 --gamma 10 --storage vpl --full   # single-stage DistFlow (slow: hours)

Variants (experiment III):
    standard         both stages use all candidates
    no_slack         the slack nodes are removed from the candidates, in both stages
    cess_plus_slack  stage 1 is solved as CESS; stage 2 places VPLs on those sites + the slack nodes
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # paper_companion/

from optimizations.DistFlow import solve_distflow  # noqa: E402
from optimizations.LinDistFlow import solve_lindistflow  # noqa: E402
from utilities.ac_replay import ac_replay  # noqa: E402
from utilities.instance import load_instance  # noqa: E402


def report(name, res, ac=None):
    print(f"\n=== {name} ===")
    print(f"  objective {res['objective']:.6f}   bound {res['bound']:.6f}   gap {100 * res['mip_gap']:.3f}%   "
          f"{res['runtime_s']:.0f} s")
    print("  components:", {k: round(v, 6) for k, v in res["components"].items()})
    print("  siting {node position: units}:", res["gamma"])
    if ac is not None:
        print(f"  AC replay objective {ac['objective']:.6f}  components:",
              {k: round(v, 6) for k, v in ac["components"].items()},
              f" |V| in [{ac['v_min_pu']:.4f}, {ac['v_max_pu']:.4f}] pu")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--C-total-MWh", type=float, default=5.0, help="total installed capacity")
    ap.add_argument("--gamma", type=int, default=10, help="max number of units (unit size = C_total / gamma)")
    ap.add_argument("--storage", choices=["vpl", "cess"], default="vpl")
    ap.add_argument("--variant", choices=["standard", "no_slack", "cess_plus_slack"], default="standard")
    ap.add_argument("--full", action="store_true", help="also solve the single-stage DistFlow on all candidates")
    ap.add_argument("--baseline", action="store_true", help="only solve DistFlow without storage")
    ap.add_argument("--time-limit", type=float, default=None, help="per solve [s]")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--quiet", action="store_true", help="no gurobi log")
    args = ap.parse_args()

    inst = load_instance()
    common = dict(time_limit=args.time_limit, threads=args.threads, verbose=not args.quiet)
    C_total_kWh = 1000 * args.C_total_MWh

    if args.baseline:
        res = solve_distflow(inst, C_total_kWh, args.gamma, vpl=False, fix_gamma=[0] * inst.N, **common)
        report("DistFlow, no storage", res, ac_replay(inst))
        return

    vpl = args.storage == "vpl"
    candidates = inst.candidates
    if args.variant == "no_slack":
        candidates = [n for n in candidates if n not in inst.slacks]
    vpl_stage1 = False if args.variant == "cess_plus_slack" else vpl
    if args.variant == "cess_plus_slack" and not vpl:
        raise SystemExit("--variant cess_plus_slack is a VPL variant (stage 1 is CESS)")

    s1 = solve_lindistflow(inst, C_total_kWh, args.gamma, vpl=vpl_stage1, candidates=candidates, **common)
    report(f"stage 1: LinDistFlow, {'VPL' if vpl_stage1 else 'CESS'}", s1)

    sites = sorted(s1["gamma"])
    if args.variant == "cess_plus_slack":
        sites = sorted(set(sites) | set(inst.slacks))
    s2 = solve_distflow(inst, C_total_kWh, args.gamma, vpl=vpl, candidates=sites, gamma_start=s1["gamma"], **common)
    report(f"stage 2: DistFlow on the stage-1 sites, {'VPL' if vpl else 'CESS'}", s2, ac_replay(inst, s2["dC"], s2["dQ"]))

    if args.full:
        full = solve_distflow(inst, C_total_kWh, args.gamma, vpl=vpl, candidates=candidates, **common)
        report(f"single stage: DistFlow on all candidates, {'VPL' if vpl else 'CESS'}", full,
               ac_replay(inst, full["dC"], full["dQ"]))


if __name__ == "__main__":
    main()
