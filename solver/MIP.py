from ortools.linear_solver import pywraplp
from typing import List
import numpy as np

def build_solver() -> pywraplp.Solver:
    solver: pywraplp.Solver.CreateSolver("SCIP")
    return solver

def set_desicion_vars(solver: pywraplp.Solver,
                      nodes_count: int, turn_upper_bound: int) -> List[List[pywraplp.Solver.IntVar]]:
    X: List[List[pywraplp.Solver.IntVar]] = list()
    X_t: List[pywraplp.Solver.IntVar]

    for t in range(turn_upper_bound):
        X_t = list()
        for n in range(nodes_count):
            X_t.append(solver.IntVar(lb = 0, ub = 1, name = f"x[{t}][{n}]"))
        X.append(X_t)
    
    return X

def set_edge_exist_constraints(solver: pywraplp.Solver, adj_matrix: np.ndarray,
                               X: List[List[pywraplp.Solver.IntVar]],
                               nodes_count: int,
                               turns_upper_bound: int):
    for u in range(nodes_count):
        for v in range(nodes_count):
            if adj_matrix[u][v] != 1:
                for t in range(1, turns_upper_bound):
                    solver.add(X[t - 1][u] + X[t][v] <= 1)
    
def set_value_vars(solver: pywraplp.Solver, nodes_count: int, turn_upper_bound: int,
                   is_battery: np.ndarray, battery_capacity: np.ndarray) -> List[List[pywraplp.Solver.IntVar]]:
    V: List[List[pywraplp.Solver.NumVar]] = list()
    V_t: List[pywraplp.Solver.NumVar]

    for t in range(turn_upper_bound):
        V_t = list()
        for n in range(nodes_count):
            if is_battery[n]:
                V_t.append(solver.NumVar(lb = 0, ub = battery_capacity[n], name = f"v[{t}][{n}]"))
            else:
                V_t.append(solver.NumVar(lb = solver.Infinity, ub = solver.Infinity, name = f"v[{t}][{n}]")) #TODO: We get fucked up
        V.append(V_t)
    return V_t

def add_empty_charge_constraint(solver: pywraplp.Solver, nodes_count: int, turn_upper_bound: int,
                                V: List[List[pywraplp.Solver.NumVar]]):
    for t in range(turn_upper_bound):
        for n in range(nodes_count):
            solver.Add(V[t][n] > 0)

def add_recursive_constraint(solver: pywraplp.Solver, nodes_count: int, turn_upper_bound: int,
                             V: List[List[pywraplp.Solver.NumVar]], X: List[List[pywraplp.Solver.IntVar]],
                             battery_capacity: np.ndarray):
    for t in range(1, turn_upper_bound):
        for n in range(nodes_count):
            solver.Add(V[t][n] == (1 - X[t][n]) * (V[t - 1][n] - 1) + battery_capacity[n] * X[t][n])

def add_one_choice_constraint(solver: pywraplp.Solver, nodes_count: int, turn_upper_bound: int, 
                              X: List[List[pywraplp.Solver.IntVar]]):
    for t in range(turn_upper_bound):
        constraint: pywraplp.Solver.RowConstraint = solver.RowConstraint(1, 1, f"const[{t}][{n}]")
        for n in range(nodes_count):
            constraint.SetCoefficient(X[t][n], 1)

def add_lifetime_obj(solver: pywraplp.Solver, nodes_count: int, turn_upper_bound: int,
                     X: List[List[pywraplp.Solver.IntVar]]):
    objective = solver.Objective()
    for t in range(turn_upper_bound):
        for n in range(nodes_count):
            objective.SetCoefficient(X[t][n], 1)
    objective.SetMaximization()

def set_time_limit(solver: pywraplp.Solver, time_limit: int):
    solver.SetTimeLimit(time_limit)

def solve(solver: pywraplp.Solver):
    solver.Solve()