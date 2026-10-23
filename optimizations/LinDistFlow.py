"""Stage 1: siting and sizing under LinDistFlow, minimizing active-power congestion (a MILP).

Variables per operating hour t = 0..T-2 (the last hour only closes the state of charge):
    P[t, e], Q[t, e]   active / reactive flow on edge e = (i, j), sending end, oriented away from the slack
    v[t, n]            squared voltage magnitude [pu^2]
    X[t, e] >= 0       overload of |P| above the soft rating
    A[t, e] >= 0       |P[t, e]|

Network (j any non-slack node, e = (i, j) the line feeding it, k over the lines j feeds):
    P[t, e] = dC[t, j] + p[t, j] + sum_k P[t, k]               active power balance
    Q[t, e] = dQ[t, j] + q[t, j] + sum_k Q[t, k]               reactive power balance
    v[t, j] = v[t, i] - 2 (r_e P[t, e] + x_e Q[t, e])          linearized voltage drop
    v[t, s] = 1                                                at both slack nodes s
    0.75^2 <= v <= 1.25^2
    |P[t, e]| <= P_rated_e + X[t, e]                           soft line rating

Objective
    min  sum_{t,e} X[t, e]  +  0.03 * sum_{t,e} |P[t, e]|

The slack nodes have no balance row: whatever is injected there is absorbed by the upstream
grid. So storage at a slack node only matters through the VPL row sum_n dC[t, n] = 0.
"""
from __future__ import annotations

import gurobipy as gp
from gurobipy import GRB

from optimizations.storage import add_first_stage, add_storage, siting

W_REAL_POWER = 0.03
V2_MIN, V2_MAX = 0.5625, 1.5625


def solve_lindistflow(inst, C_total_kWh, Gamma, vpl, candidates=None, time_limit=None, threads=None,
                      verbose=True):
    C_base = C_total_kWh / Gamma / inst.S_base_kW
    candidates = inst.candidates if candidates is None else candidates
    Tm, N, E = inst.T - 1, inst.N, len(inst.edges)
    p, q = inst.p[:Tm], inst.q[:Tm]

    m = gp.Model("LinDistFlow_stage1")
    m.Params.OutputFlag = int(verbose)
    if time_limit is not None:
        m.Params.TimeLimit = time_limit
    if threads is not None:
        m.Params.Threads = threads

    gamma, C_max = add_first_stage(m, inst, Gamma, C_base, candidates)
    C, dC, dQ = add_storage(m, inst, C_max, vpl)

    P = m.addMVar((Tm, E), lb=-GRB.INFINITY, name="P")
    Q = m.addMVar((Tm, E), lb=-GRB.INFINITY, name="Q")
    v = m.addMVar((Tm, N), lb=V2_MIN, ub=V2_MAX, name="v")
    X = m.addMVar((Tm, E), lb=0, name="X")
    A = m.addMVar((Tm, E), lb=0, name="absP")

    for j, e in inst.parent_edge.items():
        i = inst.edges[e][0]
        kids = inst.child_edges[j]
        m.addConstr(P[:, e] == dC[:, j] + p[:, j] + P[:, kids].sum(axis=1), name=f"P_balance[{j}]")
        m.addConstr(Q[:, e] == dQ[:, j] + q[:, j] + Q[:, kids].sum(axis=1), name=f"Q_balance[{j}]")
        m.addConstr(v[:, j] == v[:, i] - 2 * (inst.r[e] * P[:, e] + inst.x[e] * Q[:, e]), name=f"v_drop[{j}]")
    for s in inst.slacks:
        m.addConstr(v[:, s] == 1.0, name=f"v_slack[{s}]")

    for e in range(E):
        m.addConstr(P[:, e] <= inst.P_rated[e] + X[:, e], name=f"rating_pos[{e}]")
        m.addConstr(-P[:, e] <= inst.P_rated[e] + X[:, e], name=f"rating_neg[{e}]")
        m.addConstr(A[:, e] >= P[:, e])
        m.addConstr(A[:, e] >= -P[:, e])

    congestion = X.sum()
    flow = W_REAL_POWER * A.sum()
    m.setObjective(congestion + flow, GRB.MINIMIZE)
    m.optimize()

    return {
        "status": m.Status, "objective": m.ObjVal, "bound": m.ObjBound, "mip_gap": m.MIPGap,
        "runtime_s": m.Runtime,
        "components": {"congestion": congestion.getValue().item(), "w_real_power*|P|": flow.getValue().item()},
        "gamma": siting(gamma),
    }
