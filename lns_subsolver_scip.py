"""
Implement problems and solvers for LNS subproblems. Here LNS subproblems mean that only a subset of variables are free, while the rest are fixed.
"""

# log should be unified to use the same logger

import numpy as np
import pyscipopt
import utils_io
from utils_io import SolverStatus

__scip_status_dict__ = {
    "optimal": SolverStatus.OPTIMAL,
    "timelimit": SolverStatus.TIME_LIMIT,
    "sollimit": SolverStatus.SOLUTION_LIMIT,
    "gaplimit": SolverStatus.GAP_LIMIT,
    "infeasible": SolverStatus.INFEASIBLE,
    "unbounded": SolverStatus.UNBOUNDED,
    "inforunbd": SolverStatus.UNBOUNDED,
    "userinterrupt": SolverStatus.INTERRUPTED,
    "unknown": SolverStatus.UNDEFINED,
}


### currently, there is a very strange bug: var_lb becomes all zeros and var_ub becomes all ones. This should be wrong for CFLPTC.
### as a temporary solution, I directly change the lbs and ubs in the function in subproblem_solve. This works for most cases where nonbinary variables are auxiliary variables, but could fail for the rest.


def scip_subproblem_solve(
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
    Formulating the sub-problem with high-degree terms and solve it using scip.

    Parameters:
        problem_object (Neighborhoodsearch.Problem): Problem object containing AMPL model and other info.
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
    # build the pyscipopt model corresponding to the problem
    model = pyscipopt.Model()
    # add variables. Please note: variable orders may change, so do use variable names to call variables
    for var_idx, var_name in enumerate(all_vars):
        var_info = variable_info[var_name]
        var_type = var_info["type"]
        if var_type == "B" and neighborhood[var_idx] == 0:
            # fix variables outside the neighborhood to their values in cur_sol
            fix_value = cur_sol[var_idx]
            fix_value = int(round(fix_value))
            var_lb, var_ub = fix_value, fix_value
        else:
            # ----- original code -----
            # var_lb = problem_object.var_lb[var_idx]
            # var_ub = problem_object.var_ub[var_idx]
            # ----- temporary fix -----
            if var_type == "B":
                var_lb = 0
                var_ub = 1
            else:
                var_lb = None
                var_ub = None
            # ----- temporary fix -----
        model.addVar(
            name=var_name,
            vtype=var_type,
            lb=var_lb,
            ub=var_ub,
        )
    # add the auxiliary variable for the nonlinear objective. This is necessary as scip only supports linear objective.
    model.addVar("obj", vtype="C")
    model_var_dict = {var.name: var for var in model.getVars()}
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
            model.addCons(pyscipopt.quicksum(terms) <= rhs)
        elif sense == ">":
            model.addCons(pyscipopt.quicksum(terms) >= rhs)
        else:
            model.addCons(pyscipopt.quicksum(terms) == rhs)
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
    obj_var = model_var_dict["obj"]
    model.addCons(obj_var == pyscipopt.quicksum(obj_terms))
    if obj_sense == -1:
        model.setObjective(obj_var, sense="maximize")
    else:
        model.setObjective(obj_var, sense="minimize")

    # set solver parameters
    model.setRealParam("limits/time", time_limit)
    if solution_limit is not None:
        model.setIntParam("limits/solutions", solution_limit)
    if gap_limit is not None:
        model.setRealParam("limits/gap", gap_limit)
    if obj_limit is not None:
        model.setRealParam("limits/primal", obj_limit)

    model.hideOutput()
    model.optimize()
    status = SolverStatus(__scip_status_dict__[model.getStatus()])
    try:
        cur_obj = model.getObjVal()
        for var_idx, var_name in enumerate(all_vars):
            var = model_var_dict[var_name]
            cur_sol[var_idx] = model.getVal(var)
    except Exception:
        cur_obj = -1
    if logfile_path is not None:
        utils_io.log(
            f"Subproblem: {problem_name}, obj: {cur_obj}, status: {status}, time: {model.getSolvingTime()}",
            logfile_path,
        )
    # free memory
    model.freeProb()
    return (cur_sol, cur_obj, status)
