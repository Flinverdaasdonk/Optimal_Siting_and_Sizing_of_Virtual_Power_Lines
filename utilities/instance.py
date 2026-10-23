"""Load the instance in grid_data/ and turn it into what the two models need.

Everything is converted to per unit here, once:
    S_base = instance.json["S_base_kW"]        (the peak apparent flow of the storage-free grid)
    V_base = node u_rated (20 kV)               Z_base = V_base^2 / S_base
    p, q, ratings, C_base   -> / S_base         r, x -> / Z_base
One timestep is one hour, so an energy in kWh / S_base and a power in kW / S_base are the same
per-unit number; that is why delta_C = C[t+1] - C[t] appears directly in the power balance.

Nodes are referred to by their POSITION in grid["node"]["id"] (0..N-1); that is also the column
of the load profiles and the index of gamma_n.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
from grid_processing_toolkit.builders import GridBuilder

GRID_DATA = Path(__file__).resolve().parents[1] / "grid_data"


@dataclass
class Instance:
    grid: object             # the gptk grid (used by the AC replay)
    T: int                   # timesteps (hours)
    N: int                   # nodes
    S_base_kW: float
    p_kW: np.ndarray         # (T, N), positive = load
    q_kvar: np.ndarray       # (T, N)
    p: np.ndarray            # (T, N) per unit
    q: np.ndarray            # (T, N) per unit
    edges: list              # [(i, j)] node positions, oriented away from the slack
    line_index: np.ndarray   # (E,) row of each edge in grid["line"]
    r: np.ndarray            # (E,) per unit
    x: np.ndarray            # (E,) per unit
    P_rated: np.ndarray      # (E,) soft active-power rating (LinDistFlow), per unit
    I_rated: np.ndarray      # (E,) soft current rating (DistFlow), per unit
    parent_edge: dict        # j -> edge index of the line feeding j (every non-slack node)
    child_edges: dict        # j -> edge indices of the lines j feeds
    subtree: list            # per edge (i, j): the node positions behind j, j included
    slacks: list             # node positions of the two slack (source) nodes
    candidates: list         # node positions where storage may be placed


def load_instance(grid_data=GRID_DATA):
    grid_data = Path(grid_data)
    with open(grid_data / "grid.json") as f:
        grid = GridBuilder().from_grid_dict(json.load(f))
    with open(grid_data / "instance.json") as f:
        meta = json.load(f)
    profiles = np.load(grid_data / "load_profiles.npz")
    p_kW, q_kvar = profiles["p_kW"], profiles["q_kvar"]

    g = grid.grid
    node_ids = [int(n) for n in g["node"]["id"]]
    pos = {n: k for k, n in enumerate(node_ids)}
    T, N = p_kW.shape

    S_base = float(meta["S_base_kW"])
    V_base_kV = float(g["node"]["u_rated"][0]) / 1e3
    Z_base = V_base_kV ** 2 * 1e3 / S_base

    # the grid is a forest: one radial tree per slack, edges oriented away from it
    line_of = {frozenset((int(a), int(b))): k for k, (a, b) in enumerate(zip(g["line"]["from_node"], g["line"]["to_node"]))}
    graph = nx.Graph()
    graph.add_nodes_from(node_ids)
    graph.add_edges_from(zip(g["line"]["from_node"], g["line"]["to_node"]))
    slacks = [pos[int(s)] for s in g["source"]["node"]]
    tree = nx.DiGraph()
    for s in g["source"]["node"]:
        tree.add_edges_from(nx.bfs_edges(graph, int(s)))
    edge_ids = list(tree.edges())

    edges = [(pos[i], pos[j]) for i, j in edge_ids]
    line_index = np.array([line_of[frozenset(e)] for e in edge_ids])
    parent_edge = {j: e for e, (_, j) in enumerate(edges)}
    child_edges = {n: [] for n in range(N)}
    for e, (i, _) in enumerate(edges):
        child_edges[i].append(e)
    subtree = [np.array([pos[n] for n in nx.descendants(tree, j) | {j}]) for _, j in edge_ids]

    return Instance(
        grid=grid, T=T, N=N, S_base_kW=S_base, p_kW=p_kW, q_kvar=q_kvar,
        p=p_kW / S_base, q=q_kvar / S_base,
        edges=edges, line_index=line_index,
        r=np.array(g["line"]["r1"], dtype=float)[line_index] / Z_base,
        x=np.array(g["line"]["x1"], dtype=float)[line_index] / Z_base,
        P_rated=np.array(meta["P_rated_kW"])[line_index] / S_base,
        I_rated=np.array(meta["I_rated_kVA"])[line_index] / S_base,
        parent_edge=parent_edge, child_edges=child_edges, subtree=subtree,
        slacks=slacks, candidates=list(meta["candidate_positions"]),
    )
