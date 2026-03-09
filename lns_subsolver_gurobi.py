"""
Implement problems and solvers for LNS subproblems based on Gurobi. Here LNS subproblems mean that only a subset of variables are free, while the rest are fixed.
"""

### For those want to simplfy the code: you can easily put every lns_subsolver functions into one single file. However, some commercial solvers have limitations on running sessions, and even if you don't call them, importing the solver libaries may also count into the session limit. So it is better to separate different solvers into different files before you make sure this limitation not exist for your solvers.

import numpy as np
import gurobipy as gp
import utils_io
from utils_io import SolverStatus

gurobi_env = gp.Env()
gurobi_env.setParam("LogToConsole", 0)
gurobi_env.setParam("Threads", 1)  # limit Gurobi to use only one thread

__gurobi_status_dict__ = {
    gp.GRB.OPTIMAL: SolverStatus.OPTIMAL,
    gp.GRB.TIME_LIMIT: SolverStatus.TIME_LIMIT,
    gp.GRB.SOLUTION_LIMIT: SolverStatus.SOLUTION_LIMIT,
    gp.GRB.INTERRUPTED: SolverStatus.INTERRUPTED,
    gp.GRB.INFEASIBLE: SolverStatus.INFEASIBLE,
    gp.GRB.UNBOUNDED: SolverStatus.UNBOUNDED,
    gp.GRB.INF_OR_UNBD: SolverStatus.UNBOUNDED,
}


def gurobi_subproblem_solve(
    problem_info: dict,  # problem information obtained from instance_info_get
    all_vars: list,  # All variables
    cur_sol: np.ndarray,  # Current solution
    time_limit: float,  # Time limit for optimization
    neighborhood: np.ndarray = None,  # Neighborhood (0-1 vector)
    solution_limit: int = None,  # solution limit for solve
    gap_limit: float = None,  # gap limit for solve
    obj_limit: float = None,  # objective limit for solve
    problem_name: str = None,  # Name of the problem file
    logfile_path=None,
):
    """
    Formulating the sub-problem with high-degree terms and solve it using gurobi.

    Parameters:
        problem_info (dict): Problem object containing AMPL model and other info.
        all_vars (list): All variables
        cur_sol (np.ndarray): Current solution
        time_limit (float): Time limit for optimization
        neighborhood (np.ndarray): Neighborhood (0-1 vector)
        solution_limit (int): Solution limit for solve
        gap_limit (float): Gap limit for solve
        problem_name (str): Name of the problem file
        logfile_path: Path to the log file
    Returns:
        cur_sol (np.ndarray): Current solution
        cur_obj (float): Objective value of current solution
        status (int): Status of the optimization
    """
    if neighborhood is None:
        neighborhood = np.ones(len(all_vars))
    # unpack problem information
    variable_info = problem_info["variable_info"]
    objective_info = problem_info["objective_info"]
    constraint_info = problem_info["constraint_info"]

    # build the gurobi model corresponding to the problem
    model = gp.Model(env=gurobi_env)
    # add variables. Please note: variable orders may change, so do use variable names to call variables
    model_var_dict = {}
    for var_idx, var_name in enumerate(all_vars):
        var_info = variable_info[var_name]
        var_type = var_info["type"]
        if var_type == "B" and neighborhood[var_idx] == 0:
            # fix variables outside the neighborhood to their values in cur_sol
            var_lb, var_ub = cur_sol[var_idx], cur_sol[var_idx]
        else:
            # ----- original code -----
            # var_lb = problem_object.var_lb[var_idx]
            # var_ub = problem_object.var_ub[var_idx]
            # ----- temporary fix -----
            if var_type == "B":
                var_lb = 0
                var_ub = 1
            else:
                var_lb = -gp.GRB.INFINITY
                var_ub = gp.GRB.INFINITY
            # ----- temporary fix -----
        var = model.addVar(lb=var_lb, ub=var_ub, vtype=var_type, name=var_name)
        model_var_dict[var_name] = var
    # add the auxiliary variable for the nonlinear objective. Actually gurobi supports quadratic objective, but for consistency we still add the auxiliary variable here.
    obj_var = model.addVar(lb=-float("inf"), ub=float("inf"), vtype="C", name="obj")
    model_var_dict["obj"] = obj_var

    # add constraints
    constraint_terms, constraint_rhs, constraint_sense = constraint_info
    for constr_idx, constr in enumerate(constraint_terms):
        rhs, sense = constraint_rhs[constr_idx], constraint_sense[constr_idx]
        terms = []
        for term_list in constr:
            coeff, var_info = term_list
            term = coeff
            for var_name, var_degree in var_info.items():
                var = model_var_dict[var_name]
                if var_degree == 1:
                    term *= var
                else:
                    term *= var**var_degree
            terms.append(term)
        if sense == "<":
            model.addConstr(gp.quicksum(terms) <= rhs)
        elif sense == ">":
            model.addConstr(gp.quicksum(terms) >= rhs)
        else:
            model.addConstr(gp.quicksum(terms) == rhs)
    # set objective
    obj_term_list, obj_sense = objective_info
    obj_terms = []
    for term_list in obj_term_list:
        coeff, var_info = term_list
        term = coeff
        for var_name, var_degree in var_info.items():
            var = model_var_dict[var_name]
            if var_degree == 1:
                term *= var
            else:
                term *= var**var_degree
        obj_terms.append(term)
    model.addConstr(obj_var == gp.quicksum(obj_terms))
    if obj_sense == -1:
        model.setObjective(obj_var, sense=gp.GRB.MAXIMIZE)
    else:
        model.setObjective(obj_var, sense=gp.GRB.MINIMIZE)

    # set solver parameters
    model.setParam("NonConvex", 2)
    model.setParam("TimeLimit", time_limit)
    if solution_limit is not None:
        model.setParam("SolutionLimit", solution_limit)
    if gap_limit is not None:
        model.setParam("MIPGap", gap_limit)
    if obj_limit is not None:
        model.setParam("BestObjStop", obj_limit)
    model.setParam("OutputFlag", 0)  # disable inherent log prints of Gurobi
    model.update()
    model.optimize()

    status = SolverStatus(__gurobi_status_dict__[model.status])
    # return current solution
    try:
        cur_obj = model.ObjVal
        for var_idx, var_name in enumerate(all_vars):
            var = model_var_dict[var_name]
            cur_sol[var_idx] = var.X
    except Exception:
        cur_obj = -1
    if logfile_path is not None:
        utils_io.log(
            f"Subproblem: {problem_name}, obj: {cur_obj}, status: {status}, time: {model.Runtime}",
            logfile_path,
        )
    # free memory
    model.dispose()
    return (cur_sol, cur_obj, status)
