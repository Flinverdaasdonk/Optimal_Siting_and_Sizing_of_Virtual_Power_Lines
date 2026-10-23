"""Price a DistFlow dispatch under AC power flow (power-grid-model, Newton-Raphson).

Storage enters the operational problem only through its injections dC, dQ. Fixing them turns the
second stage into a load flow: run it on the load p + dC (and q + dQ), read the AC flows and
voltages back, and evaluate the SAME objective as the model,

    sum_{t,e} max(0, l[t, e] - I_rated_e^2)  +  0.03 * sum_{t,e} l[t, e],   l = (P^2 + Q^2) / v_sending

so the model's value and this one are directly comparable. The grid in grid_data/ is already set
up for this: no line capacitances (the models have none either), both sources ideal at 1.0 pu
(the models' slack voltage), and a sym_load on every node so a dispatch can be applied anywhere.
"""
from __future__ import annotations

import numpy as np

from optimizations.DistFlow import W_CURRENT

W_PER_KW = 1e3


def ac_replay(inst, dC=None, dQ=None):
    """The objective of the dispatch (dC, dQ) [per unit, (T-1, N)] under AC power flow.
    dC = dQ = None replays the grid without storage."""
    Tm = inst.T - 1
    dC = np.zeros((Tm, inst.N)) if dC is None else dC
    dQ = np.zeros((Tm, inst.N)) if dQ is None else dQ
    p_kW = inst.p_kW[:Tm] + dC * inst.S_base_kW
    q_kvar = inst.q_kvar[:Tm] + dQ * inst.S_base_kW

    g = inst.grid.grid
    node_pos = {int(n): k for k, n in enumerate(g["node"]["id"])}
    load_cols = [node_pos[int(n)] for n in g["sym_load"]["node"]]      # one sym_load per node
    batch = {"sym_load": {"id": np.tile(np.array(g["sym_load"]["id"]), (Tm, 1)),
                          "p_specified": p_kW[:, load_cols] * W_PER_KW,
                          "q_specified": q_kvar[:, load_cols] * W_PER_KW}}
    out = inst.grid.batch_load_flow(batch)
    node_out = next(v for k, v in out.items() if str(k).endswith("node"))
    line_out = next(v for k, v in out.items() if str(k).endswith("line"))

    # sending-end flows under the models' edge orientation (which need not be pgm's from -> to)
    from_node = np.array(g["line"]["from_node"])[inst.line_index]
    tail_id = np.array([int(g["node"]["id"][i]) for i, _ in inst.edges])
    same = from_node == tail_id
    li = inst.line_index
    P = np.where(same, line_out["p_from"][:, li], line_out["p_to"][:, li]) / W_PER_KW / inst.S_base_kW
    Q = np.where(same, line_out["q_from"][:, li], line_out["q_to"][:, li]) / W_PER_KW / inst.S_base_kW
    v = node_out["u_pu"] ** 2
    v_tail = v[:, [i for i, _ in inst.edges]]

    ell = (P ** 2 + Q ** 2) / v_tail
    congestion = float(np.maximum(ell - inst.I_rated ** 2, 0.0).sum())
    current = float(W_CURRENT * ell.sum())
    return {"objective": congestion + current,
            "components": {"current_congestion": congestion, "w_current*l": current},
            "v_min_pu": float(np.sqrt(v.min())), "v_max_pu": float(np.sqrt(v.max()))}
