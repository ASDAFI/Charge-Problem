from ortools.linear_solver import pywraplp
from typing import List

def build_solver() -> pywraplp.Solver:
    solver: pywraplp.Solver.CreateSolver("SCIP")
    return solver

def set_desicion_vars(solver: pywraplp.Solver,
                      nodes_count: int, turn_upper_bound: int) -> List[List[pywraplp.Solver.IntVar]]:
    X: List[List[pywraplp.Solver.IntVar]] = list()
    X_t: List[pywraplp.Solver.IntVar]

    for t in range(turn_upper_bound):
        for n in range(nodes_count):
            X_t.append(solver.IntVar(lb = 0, ub = 1, name = f"x[{t}][{n}]"))
        X.append(X_t)
    
    return X

