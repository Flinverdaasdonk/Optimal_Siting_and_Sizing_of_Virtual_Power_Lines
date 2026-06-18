"""
This file shows a gurobipy implementation of the proposed MILP. Note that it assumes some 'grid' object that has the function 'compute_DC_PTDF_matrix'.
The grid object is a format by the open source python package `grid_processing_toolkit`
"""
def milp(
    grid,  # object describing the grid (nodes, edges, etc)
    load_profiles,
    line_limits,
    Gamma,
    C_upper,
    C_total,
    r,  # c-rating
    possible_storage_device_locations, # candidate set
    halffull_at_nth_index=24,
    verbose_model=True,
    use_virtual_cable_constraint=True,
):
    model = gp.Model()
    F = grid.compute_DC_PTDF_matrix()

    T = len(load_profiles)  # number of timesteps
    E, N = F.shape  # number of edges, number of nodes

    P_rated_matrix = np.tile(line_limits, (T, 1))

    p = load_profiles

    ## variables
    # storage size per node
    C_max = model.addMVar(N, lb=0, vtype=GRB.CONTINUOUS, name="C_max")

    gamma = model.addMVar(N, vtype=GRB.BINARY, name="gamma")

    # state of charge per timestep per node
    C = model.addMVar((T, N), lb=0, vtype=GRB.CONTINUOUS, name="C")

    # delta C
    delta_C = model.addMVar((T - 1, N), lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="delta_C")

    # congestion per timestep per node
    X_abs = model.addMVar((T - 1, E), lb=0, vtype=GRB.CONTINUOUS, name="X_abs")
    X = model.addMVar((T - 1, E), lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="X_R")

    ## constraints

    ## limit number of storage devices
    model.addConstr(gp.quicksum(gamma) <= Gamma, name="total_storage_constraint")

    # # # convexification of siting+sizing multiplication
    model.addConstr(C_max <= gamma * C_upper, name=f"sized_and_sited_constraint")

    # # # limit total capacity
    model.addConstr(gp.quicksum(C_max) <= C_total, name="total_storage_size_constraint")

    # # # charge consistency constraint:
    model.addConstr(delta_C == C[1:] - C[:-1], name="delta_C_definition")

    # # state of charge constraints
    model.addConstr(C <= C_max, name=f"upperlimit_state_of_charge_constraints")

    # # # charge and discharge rate constraints
    model.addConstr(delta_C <= r * C_max, name="charge_rate_constraints")
    model.addConstr(-r * C_max <= delta_C, name="discharge_rate_constraints")

    # # intermediate state-of-charge constraints
    for t in range(T):
        if halffull_at_nth_index == 0:
            break
        if t % halffull_at_nth_index == 0 or (t + 1) % halffull_at_nth_index == 0:
            model.addConstr(C[t, :] == 0.5 * C_max, name=f"intermediate_SOC_t{t}")

    model.addConstr(F @ (p[:-1].T + C[1:].T - C[:-1].T) <= P_rated_matrix + X.T, name="overloading_lines_constraints")
    model.addConstr(-P_rated_matrix + -X.T <= F @ (p[:-1].T + C[1:].T - C[:-1].T), name="underloading_lines_constraints")


    # # auxiliary constraints to ensure that X is an absolute value
    model.addConstr(X_abs >= X, name="auxiliary_positive_congestion_constraints")
    model.addConstr(X_abs >= -X, name="auxiliary_negative_congestion_constraints")


    # # virtual cable constraints
    if use_virtual_cable_constraint:
        model.addConstr(gp.quicksum(delta_C.T) == 0, name=f"virtual_cable_constraint_t")

    for idx, value in enumerate(possible_storage_device_locations):
        if value == 0:
            model.addConstr(gamma[idx] == 0, name=f"no_storage_device_at_node_{idx}")


    objective = gp.quicksum(gp.quicksum(X_abs))

    model.setObjective(objective, GRB.MINIMIZE)

    if verbose_model is False:
        model.Params.OutputFlag = 0

    model.setParam("MIPGap", CONFIG_MIPGAP)  

    if CONFIG_MAX_SOLVETIME is not None:
        model.setParam("TimeLimit", CONFIG_MAX_SOLVETIME)

    model.update()
    return
