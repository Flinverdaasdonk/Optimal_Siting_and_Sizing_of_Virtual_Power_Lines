"""The first stage (siting and sizing) and the storage model, shared by both physics.

Notation: t = 0..T-1 hours, n = 0..N-1 nodes, Gamma = max number of units, C_base = unit size.

First stage
    gamma_n in {0, 1, ..., Gamma}           integer number of units at node n
    sum_n gamma_n <= Gamma
    gamma_n = 0                             for n not a candidate
    C_max_n = C_base * gamma_n              installed energy capacity

Storage (a lossless device; one hour per step, rate r = 1, i.e. it can (dis)charge its full
capacity in one hour)
    0 <= C[t, n] <= C_max_n                 state of charge, t = 0..T-1
    dC[t, n] = C[t+1, n] - C[t, n]          net charging (> 0) / discharging (< 0), t = 0..T-2
    -C_max_n <= dC[t, n] <= C_max_n
    C[0, n] = C[T-1, n] = C_max_n / 2       half full at the start and end of the horizon
    reactive power dQ: (dC, dQ) inside a regular octagon around the circle of radius C_max_n,
        cos(th_k) dC + sin(th_k) dQ <= C_max_n   for th_k = 45, 90, 135, 225, 270, 315 deg
        (the 0 and 180 deg facets are the dC rate limits above)
    VPL only ("virtual power line"):  sum_n dC[t, n] = 0   -- the fleet as a whole never
        charges or discharges, it only moves energy between sites. CESS has no such row.

Note: the 4 days of the horizon (day of year 0, 91, 182, 274) form ONE state-of-charge chain:
the storage carries energy across the day boundaries.
"""
from __future__ import annotations

import gurobipy as gp
import numpy as np
from gurobipy import GRB

# facet angles of the octagon; 0 and 180 deg are omitted (they coincide with the rate limits)
FACET_ANGLES = [2 * np.pi * k / 8 for k in (1, 2, 3, 5, 6, 7)]


def add_first_stage(m, inst, gamma_max, C_base_pu, candidates, fix_gamma=None):
    """gamma_n (integer) and C_max_n = C_base * gamma_n. `fix_gamma` (length-N) pins the siting."""
    N = inst.N
    is_candidate = np.zeros(N, dtype=bool)
    is_candidate[list(candidates)] = True
    ub = np.where(is_candidate, gamma_max, 0)
    gamma = m.addMVar(N, lb=0, ub=ub, vtype=GRB.INTEGER, name="gamma")
    if fix_gamma is not None:
        gamma.lb = gamma.ub = np.asarray(fix_gamma, dtype=float)
    m.addConstr(gamma.sum() <= gamma_max, name="max_units_total")
    C_max = m.addMVar(N, lb=0, name="C_max")
    m.addConstr(C_max == C_base_pu * gamma, name="capacity")
    return gamma, C_max


def add_storage(m, inst, C_max, vpl):
    """State of charge C (T, N), net charging dC and reactive power dQ (T-1, N)."""
    T, N = inst.T, inst.N
    C = m.addMVar((T, N), lb=0, name="C")
    dC = m.addMVar((T - 1, N), lb=-GRB.INFINITY, name="dC")
    dQ = m.addMVar((T - 1, N), lb=-GRB.INFINITY, name="dQ")

    m.addConstr(dC == C[1:, :] - C[:-1, :], name="soc_dynamics")
    for t in range(T):
        m.addConstr(C[t, :] <= C_max, name=f"soc_max[{t}]")
    for t in range(T - 1):
        m.addConstr(dC[t, :] <= C_max, name=f"charge_max[{t}]")
        m.addConstr(-dC[t, :] <= C_max, name=f"discharge_max[{t}]")
        for k, th in enumerate(FACET_ANGLES):
            m.addConstr(np.cos(th) * dC[t, :] + np.sin(th) * dQ[t, :] <= C_max, name=f"inverter[{t},{k}]")
    m.addConstr(C[0, :] == 0.5 * C_max, name="half_full_start")
    m.addConstr(C[T - 1, :] == 0.5 * C_max, name="half_full_end")

    if vpl:
        m.addConstr(dC.sum(axis=1) == 0, name="virtual_power_line")
    return C, dC, dQ


def siting(gamma):
    """{node position: units} of a solved gamma."""
    return {int(n): int(round(g)) for n, g in enumerate(gamma.X) if round(g) > 0}
