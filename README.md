# Paper companion

The results behind the paper and a minimal, self-contained re-implementation of its optimization
models: siting and sizing of storage as virtual power lines (VPL) or as conventional storage (CESS)
on the C0+S0 grid over 4 days.

## Layout

| path | what |
|---|---|
| `grid_data/grid.json` | the grid as a `grid_processing_toolkit` grid dict (load with `GridBuilder().from_grid_dict(json.load(f))`). Already prepared: the two grids merged (two islands, one slack each), made radial, line capacitances zeroed, sources ideal at 1.0 pu, a (zero) load on every node |
| `grid_data/load_profiles.npz` | `p_kW`, `q_kvar`: (96, 224) hourly load per node (positive = consumption), columns in `grid["node"]["id"]` order; days of year 0, 91, 182, 274 |
| `grid_data/instance.json` | the precomputed instance data: `S_base_kW`, the soft line ratings (`P_rated_kW` for LinDistFlow, `I_rated_kVA` for DistFlow, per line id), the candidate nodes and the slack nodes. Its `description` says how each was derived |
| `optimizations/storage.py` | first stage (integer units per node) and the storage model, shared by both physics |
| `optimizations/LinDistFlow.py` | stage 1: LinDistFlow, active-power congestion (MILP) |
| `optimizations/DistFlow.py` | stage 2: DistFlow (second-order-cone relaxation), current congestion (MISOCP) |
| `utilities/instance.py` | loads `grid_data/`, builds the radial trees, converts to per unit |
| `utilities/ac_replay.py` | re-prices a DistFlow dispatch under AC power flow (power-grid-model, Newton-Raphson) |
| `scripts/demo.py` | the two-stage procedure (and its variants), with AC re-pricing |
| `exp_I/`, `exp_II/`, `exp_III/`, `newton_raphson_and_DistFlow_parity_comparison/` | the notebooks and figures of the paper's experiments (these read the full result sets and use the original codebase) |

Each model file starts with its formulation written out; the code follows it line by line.

## Running

```bash
pip install -r requirements.txt
python scripts/demo.py --C-total-MWh 5 --gamma 10 --storage vpl      # stage 1 + stage 2, a few minutes
python scripts/demo.py --baseline                                     # no storage, seconds
python scripts/demo.py --help                                         # CESS, the slack-node variants, single stage
```

`--gamma` is the maximum number of units, both in total and per node; the unit size is
`C_total / gamma`.

## Reference values

C0+S0, 5 MWh, Gamma = 10, VPL, from the paper's runs (16 threads):

| run | model objective | AC re-priced |
|---|---|---|
| no storage (DistFlow) | 9.65340 | 9.71120 |
| stage 1 (LinDistFlow) | 34.66405 | |
| stage 2 (DistFlow) | 7.18953 | 7.25705 |

Expect small differences: the MIP gap is Gurobi's default (1e-4), the optimal siting is not always
unique, and the solver path depends on the number of threads.

## Modelling choices worth knowing

- One timestep is one hour. The 4 days form one horizon, so the state of charge carries over from
  one day to the next. The storage is half full at the first and last hour.
- The last hour of the profiles only closes the state of charge: the network is modelled for
  hours 0..94.
- The slack nodes have no power-balance row; the upstream grid absorbs whatever is injected there.
  Storage at a slack node therefore only acts through the VPL constraint `sum_n dC[t, n] = 0`.
  That is what experiment III varies.
- The device is lossless and has a reactive-power channel bounded by an octagon around its rated
  power.
