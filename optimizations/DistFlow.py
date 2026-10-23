"""Stage 2 (and the single-stage benchmark): siting and sizing under DistFlow, the second-order-cone
relaxation of AC power flow on a radial grid, minimizing branch-current congestion (a MISOCP).

Variables per operating hour t = 0..T-2 (the last hour only closes the state of charge):
    P[t, e], Q[t, e]   active / reactive flow on edge e = (i, j), sending end, oriented away from the slack
    v[t, n]            squared voltage magnitude [pu^2]
    l[t, e]            squared current magnitude [pu^2]
    X[t, e] >= 0       overload of l above the squared soft current rating

Network (j any non-slack node, e = (i, j) the line feeding it, k over the lines j feeds):
    P[t, e] - r_e l[t, e] = dC[t, j] + p[t, j] + sum_k P[t, k]                 active power balance
    Q[t, e] - x_e l[t, e] = dQ[t, j] + q[t, j] + sum_k Q[t, k]                 reactive power balance
    v[t, j] = v[t, i] - 2 (r_e P[t, e] + x_e Q[t, e]) + (r_e^2 + x_e^2) l[t, e] voltage drop
    P[t, e]^2 + Q[t, e]^2 <= l[t, e] v[t, i]                                   the cone (relaxed l = |S|^2 / v)
    v[t, s] = 1                                                                at both slack nodes s
    0.95^2 <= v <= 1.05^2
    l[t, e] <= I_rated_e^2 + X[t, e]                                           soft current rating
    l[t, e] <= (2 S_e)^2 / 0.95^2     a loose numerical guard (never binding): S_e = sum over the nodes
                                      behind e of max_t |p| + max_t |q|, plus the whole storage budget

Objective
    min  sum_{t,e} X[t, e]  +  0.03 * sum_{t,e} l[t, e]
The second term prices current everywhere. It is what keeps the cone tight: without it the
relaxation could inflate l (and with it the losses r l) for free.

As in LinDistFlow, the slack nodes have no balance row.
"""
from __future__ import annotations

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from optimizations.storage import add_first_stage, add_storage, siting

W_CURRENT = 0.03
V2_MIN, V2_MAX = 0.9025, 1.1025
ELL_GUARD_SAFETY = 2.0


def solve_distflow(inst, C_total_kWh, Gamma, vpl, candidates=None, fix_gamma=None, gamma_start=None,
                   time_limit=None, threads=None, verbose=True):
    """`candidates` restricts the siting (stage 2: the stage-1 sites); `gamma_start` is a MIP start
    ({position: units}); `fix_gamma` pins the siting (e.g. all zeros: the no-storage baseline)."""
    C_base = C_total_kWh / Gamma / inst.S_base_kW
    candidates = inst.candidates if candidates is None else candidates
    Tm, N, E = inst.T - 1, inst.N, len(inst.edges)
    p, q = inst.p[:Tm], inst.q[:Tm]

    m = gp.Model("DistFlow")
    m.Params.OutputFlag = int(verbose)
    m.Params.NumericFocus = 1
    if time_limit is not None:
        m.Params.TimeLimit = time_limit
    if threads is not None:
        m.Params.Threads = threads

    gamma, C_max = add_first_stage(m, inst, Gamma, C_base, candidates, fix_gamma=fix_gamma)
    if gamma_start is not None:
        start = np.zeros(N)
        for n, units in gamma_start.items():
            start[n] = units
        gamma.Start = start
    C, dC, dQ = add_storage(m, inst, C_max, vpl)

    # numerical guard on l (see module docstring); uses every hour of the profiles
    node_peak = np.abs(inst.p).max(axis=0) + np.abs(inst.q).max(axis=0)
    l_guard = np.array([(ELL_GUARD_SAFETY * (node_peak[sub].sum() + C_base * Gamma)) ** 2 / V2_MIN
                        for sub in inst.subtree])

    P = m.addMVar((Tm, E), lb=-GRB.INFINITY, name="P")
    Q = m.addMVar((Tm, E), lb=-GRB.INFINITY, name="Q")
    v = m.addMVar((Tm, N), lb=V2_MIN, ub=V2_MAX, name="v")
    ell = m.addMVar((Tm, E), lb=0, ub=np.broadcast_to(l_guard, (Tm, E)), name="ell")
    X = m.addMVar((Tm, E), lb=0, name="X")

    for j, e in inst.parent_edge.items():
        i = inst.edges[e][0]
        r, x = inst.r[e], inst.x[e]
        kids = inst.child_edges[j]
        m.addConstr(P[:, e] - r * ell[:, e] == dC[:, j] + p[:, j] + P[:, kids].sum(axis=1), name=f"P_balance[{j}]")
        m.addConstr(Q[:, e] - x * ell[:, e] == dQ[:, j] + q[:, j] + Q[:, kids].sum(axis=1), name=f"Q_balance[{j}]")
        m.addConstr(v[:, j] == v[:, i] - 2 * (r * P[:, e] + x * Q[:, e]) + (r ** 2 + x ** 2) * ell[:, e],
                    name=f"v_drop[{j}]")
    for s in inst.slacks:
        m.addConstr(v[:, s] == 1.0, name=f"v_slack[{s}]")

    for e, (i, _) in enumerate(inst.edges):
        m.addConstr(P[:, e] * P[:, e] + Q[:, e] * Q[:, e] <= ell[:, e] * v[:, i], name=f"cone[{e}]")
        m.addConstr(ell[:, e] <= inst.I_rated[e] ** 2 + X[:, e], name=f"rating[{e}]")

    congestion = X.sum()
    current = W_CURRENT * ell.sum()
    m.setObjective(congestion + current, GRB.MINIMIZE)
    m.optimize()

    return {
        "status": m.Status, "objective": m.ObjVal, "bound": m.ObjBound,
        "mip_gap": m.MIPGap if m.IsMIP else 0.0, "runtime_s": m.Runtime,
        "components": {"current_congestion": congestion.getValue().item(), "w_current*l": current.getValue().item()},
        "gamma": siting(gamma),
        "dC": dC.X, "dQ": dQ.X,     # the dispatch, per unit, (T-1, N): input to the AC replay
    }
