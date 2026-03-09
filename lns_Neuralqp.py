import numpy as np
import time
import random
import utils_io

INFINITY = 1e100


def get_solver(solver_name):
    if solver_name == "gurobi":
        from lns_subsolver_gurobi import gurobi_subproblem_solve

        return gurobi_subproblem_solve
    elif solver_name == "scip":
        from lns_subsolver_scip import scip_subproblem_solve

        return scip_subproblem_solve
    else:
        raise NotImplementedError(f"Solver {solver_name} is not implemented.")


class Problem(object):
    def __init__(
        self,
        solver: str,  # solver name
        problem_info: dict,  # problem information obtained from instance_info_get
        all_var: list,  # all variables
        cur_val: np.ndarray = None,  # current solution
        cur_obj: float = None,  # current objective value
    ):
        # set solver
        self.subproblem_solve = get_solver(solver)

        # set variable information
        self.problem_info = problem_info
        self.all_var = all_var
        self.n_var = len(all_var)

        # set current model status
        self.is_feasible = False
        self.is_unbounded = False
        self.cur_val = cur_val if cur_val is not None else np.zeros(self.n_var)
        self.cur_obj = cur_obj if cur_obj is not None else INFINITY

    def set_variables(self, indices: np.ndarray, values: np.ndarray):
        """
        Set the values of variables to the given values. Please note that in this function no checking or rounding is performed here to insure feasibility.
        """
        for i, val in zip(indices, values):
            self.cur_val[int(i)] = val

    def get_data(self):
        """
        Get the data of the problem.
        """
        return (
            self.all_var,
            self.problem_info,
            self.cur_val,
        )


class RepairPolicy(object):
    "Only for binary variables"

    def __init__(self, p: Problem):
        self.problem = p

    def get_repair(self, cur_sol, neiborhood):
        raise NotImplementedError("RepairPolicy has to be implemented by subclasses.")

    def check_constr_violation(
        self, all_var, var_lb, var_ub, constr_term_list, sense, rhs
    ):
        """
        Check whether a constraint is doomed to be violated given the variable bounds.

        Parameters:
        -----------
        all_var (list): List of all variable names.
        var_lb (np.ndarray): Lower bounds of variables.
        var_ub (np.ndarray): Upper bounds of variables.
        constr_term_list (list): List of terms in the constraint. [(coeff, {var_name: order, ...}), ...]
        sense (str): Sense of the constraint ("<", ">", "=").
        rhs (float): Right-hand side value of the constraint.
        """
        lhs_min, lhs_max = 0, 0
        constraint_vars = []
        for coeff, var_info in constr_term_list:
            constraint_vars += list(var_info.keys())
            term_min, term_max = 1, 1
            constraint_vars += list(var_info.keys())
            for var_name, var_order in var_info.items():
                var_idx = all_var.index(var_name)
                term_min *= var_lb[var_idx] ** var_order
                term_max *= var_ub[var_idx] ** var_order
            if coeff >= 0:
                lhs_min += coeff * term_min
                lhs_max += coeff * term_max
            else:
                lhs_min += coeff * term_max
                lhs_max += coeff * term_min

        # check if the constraint is deemed to be infeasible
        infeasible = False
        if sense == "<":
            if lhs_min > rhs:
                infeasible = True
        elif sense == ">":
            if lhs_max < rhs:
                infeasible = True
        elif sense == "=":
            if lhs_min > rhs or lhs_max < rhs:
                infeasible = True

        return infeasible, list(set(constraint_vars)), lhs_min, lhs_max


class QuickRepairPolicy(RepairPolicy):
    def get_repair(
        self,
        cur_sol: np.ndarray,
        neighborhood: np.ndarray,  # Neighborhood (0-1 vector)
        max_size: int,
    ) -> np.ndarray:
        """
        Check and repair the constraints that are doomed to be infeasible.
        """
        all_var, problem_info, _ = self.problem.get_data()

        # unpack problem information
        variable_info = problem_info["variable_info"]
        # objective_info = problem_info["objective_info"]
        # obj_term_list, obj_sense = objective_info
        constraint_info = problem_info["constraint_info"]
        constraint_terms, constraint_rhs, constraint_sense = constraint_info

        var_lb = np.array([variable_info[var_name]["lb"] for var_name in all_var])
        var_ub = np.array([variable_info[var_name]["ub"] for var_name in all_var])
        if neighborhood is None:
            neighborhood = np.zeros(self.problem.n_var)
        size = np.count_nonzero(neighborhood)
        mask = neighborhood == 0
        var_lb_neigh = var_lb.copy()
        var_ub_neigh = var_ub.copy()
        var_lb_neigh[mask] = cur_sol[mask]
        var_ub_neigh[mask] = cur_sol[mask]

        num_constraints = len(constraint_rhs)
        # Notice: code below only work for positive variables.
        for i in range(num_constraints):
            rhs, sense = constraint_rhs[i], constraint_sense[i]
            constr = constraint_terms[i]
            infeasible, constraint_vars, _, _ = self.check_constr_violation(
                all_var, var_lb_neigh, var_ub_neigh, constr, sense, rhs
            )
            # add all variables in this constraint to the neighborhood if the constraint is infeasible, until reaching max_size
            if infeasible:
                for k in constraint_vars:
                    var_idx = all_var.index(k)
                    if neighborhood[var_idx] == 0 and size < max_size:
                        neighborhood[var_idx] = 1
                        var_lb_neigh[var_idx] = var_lb[var_idx]
                        var_ub_neigh[var_idx] = var_ub[var_idx]
                        size += 1
                        if size >= max_size:
                            break
        return neighborhood


class CautiousRepairPolicy(RepairPolicy):
    def get_repair(
        self,
        cur_sol: np.ndarray,
        neighborhood: np.ndarray,  # Neighborhood (0-1 vector)
        max_size: int,
    ) -> np.ndarray:
        """
        Check and repair the constraints that are doomed to be infeasible.
        """
        all_var, problem_info, _ = self.problem.get_data()

        # unpack problem information
        variable_info = problem_info["variable_info"]
        # objective_info = problem_info["objective_info"]
        # obj_term_list, obj_sense = objective_info
        constraint_info = problem_info["constraint_info"]
        constraint_terms, constraint_rhs, constraint_sense = constraint_info

        var_lb = np.array([variable_info[var_name]["lb"] for var_name in all_var])
        var_ub = np.array([variable_info[var_name]["ub"] for var_name in all_var])
        if neighborhood is None:
            neighborhood = np.zeros(self.problem.n_var)
        size = np.count_nonzero(neighborhood)
        mask = neighborhood == 0
        var_lb_neigh = var_lb.copy()
        var_ub_neigh = var_ub.copy()
        var_lb_neigh[mask] = cur_sol[mask]
        var_ub_neigh[mask] = cur_sol[mask]

        num_constraints = len(constraint_rhs)
        # Notice: code below only work for positive variables.
        for i in range(num_constraints):
            rhs, sense = constraint_rhs[i], constraint_sense[i]
            constr = constraint_terms[i]
            infeasible, constraint_vars, _, _ = self.check_constr_violation(
                all_var, var_lb_neigh, var_ub_neigh, constr, sense, rhs
            )
            # if the constraint is infeasible, add variables in the constraint to the neighborhood one by one until reaching max_size or the constraint is no longer infeasible
            if infeasible:
                random.shuffle(constraint_vars)
                for k in constraint_vars:
                    var_idx = all_var.index(k)
                    if neighborhood[var_idx] == 0 and size < max_size:
                        neighborhood[var_idx] = 1
                        var_lb_neigh[var_idx] = var_lb[var_idx]
                        var_ub_neigh[var_idx] = var_ub[var_idx]
                        size += 1
                        # check if the constraint is still infeasible
                        infeasible, _, _, _ = self.check_constr_violation(
                            all_var,
                            var_lb_neigh,
                            var_ub_neigh,
                            constr,
                            sense,
                            rhs,
                        )
                        if not infeasible or size >= max_size:
                            break
        return neighborhood


class InitialSolutionPolicy(object):
    def __init__(self, p: Problem):
        self.problem = p

    def get_feasible_solution(cur_val):
        raise NotImplementedError("InitialPolicy has to be implemented by subclasses.")


class VariableRelaxationPolicy(InitialSolutionPolicy):
    def __init__(self, p: Problem):
        super().__init__(p)

    def get_feasible_solution(
        self,
        logfile_path: str,
        val_and_logit: np.ndarray,  # the index,predicted values and predicted loss of each variable
        repair_policy: RepairPolicy,  # repair policy
        alpha=0.1,  # the threshold of the percentage of variables to be optimized
        alpha_step=0.05,  # the step of alpha
        alpha_ub=0.25,  # the upper bound of alpha
        max_size=1000,
        initial_time_limit=30,  # time limit for each subproblem
        print_output=True, # whether to print output to console
    ):
        """
        Get a feasible solution from the given values by relaxing the variables.
        """
        start_time = time.time()
        sorted_indices = val_and_logit[:, 2].argsort()
        val_and_logit = val_and_logit[sorted_indices]

        # set the variables to the predicted values
        indices = val_and_logit[:, 0].reshape(-1)
        values = val_and_logit[:, 1].reshape(-1)
        self.problem.set_variables(indices, values)

        all_var, problem_info, cur_sol = self.problem.get_data()

        algha_exceeded = False
        while not self.problem.is_feasible and (alpha <= alpha_ub):
            num_to_optimize = int(alpha * len(val_and_logit))
            one_indices = indices[-num_to_optimize:].astype(int)
            neighborhood = np.zeros(self.problem.n_var, dtype=int)
            neighborhood[one_indices] = 1
            cur_sol = self.problem.cur_val
            neighborhood = repair_policy.get_repair(cur_sol, neighborhood, max_size)

            alpha = np.nonzero(neighborhood)[0].shape[0] / neighborhood.shape[0]
            cur_sol, obj, status = self.problem.subproblem_solve(
                problem_info,
                all_var,
                cur_sol,
                time_limit=initial_time_limit,
                neighborhood=neighborhood,
                solution_limit=1,
            )
            if (
                status == utils_io.SolverStatus.OPTIMAL
                or status == utils_io.SolverStatus.SOLUTION_LIMIT
            ):
                self.problem.is_feasible = True
                self.problem.cur_obj = obj
                self.problem.cur_val = cur_sol
                utils_io.log(
                    f"Feasible solution found alpha:{alpha} n_var:{np.nonzero(neighborhood)[0].shape[0]}",
                    logfile_path,
                    print_output=print_output,
                )
                utils_io.log(
                    f"Time used: {time.time() - start_time:.2f} seconds", logfile_path, print_output=print_output
                )
                break

            else:
                alpha += alpha_step
                if alpha >= alpha_ub:
                    if not algha_exceeded:
                        algha_exceeded = True
                        alpha = 1
                    else:
                        utils_io.log(
                            "Failed to find initial feasible solution", logfile_path, print_output=print_output
                        )
                        break
        return cur_sol, obj


class NeighborhoodPolicy(object):
    """
    Neighborhood search policy

    Attributes:
        size: number of total variables
        losses: loss of each variable, i.e., the likelihood of confidence
        rate: the rate of variables allowed to be optimized
        predictX: the predicted value of each variable
        curX: the current value of each variable
    """

    def __init__(self, p: Problem) -> None:
        self.problem = p

    def get_neighborhood(self, block_size):
        raise NotImplementedError(
            "NeighborhoodPolicy has to be implemented by subclasses."
        )


class ConstrRandomNeighborhoodPolicy(NeighborhoodPolicy):
    def __init__(self, p: Problem):
        super().__init__(p)

    def get_neighborhood(
        self, block_size, cur_val=None, val_and_logit=None, *args, **kwargs
    ) -> np.ndarray:
        """
        Get a partition of n_blocks based on the random model.
        """
        all_var, problem_info, _ = self.problem.get_data()

        # unpack problem information
        # variable_info = problem_info["variable_info"]
        # objective_info = problem_info["objective_info"]
        # obj_term_list, obj_sense = objective_info
        constraint_info = problem_info["constraint_info"]
        constraint_terms = constraint_info[0]

        n_vars, n_constrs = len(all_var), len(constraint_terms)
        perm = np.random.permutation(n_constrs)
        n_all = 0  # the number of elements in all neighborhooods

        constr_list = []
        for i in range(n_constrs):
            # get the variables in the constraint
            constraint_term = constraint_terms[i]
            vars = []
            for _, var_info in constraint_term:
                for var_name in var_info.keys():
                    vars.append(all_var.index(var_name))
            vars = list(set(vars))
            random.shuffle(vars)
            constr_list.append(vars)
            n_all += len(vars)

        n_blocks = n_all // block_size + 1
        binary_matrix = np.zeros((n_blocks, n_vars), dtype=int)
        count_var = np.zeros(n_blocks)
        neighborhood = 0
        for i in range(n_constrs):
            for j in constr_list[perm[i]]:
                if count_var[neighborhood] <= block_size:
                    binary_matrix[neighborhood, j] = 1
                    count_var[neighborhood] += 1
                else:
                    neighborhood += 1
                    binary_matrix[neighborhood, j] = 1
                    count_var[neighborhood] += 1
        return binary_matrix


def cross_neighborhood(
    p: Problem,
    repair_policy: RepairPolicy,
    block_size: int,
    neighborhood_1: np.ndarray,  # the neighborhood whose cur_obj is better
    neighborhood_2: np.ndarray,
    cur_sol_1: np.ndarray,  # the solution whose cur_obj is better
    cur_sol_2: np.ndarray,
    cur_obj_1: np.ndarray,
    cur_obj_2: np.ndarray,
    cross_time_limit: float = 30,  # the time limit for the crossover
):
    all_var, problem_info, _ = p.get_data()
    n_vars = len(all_var)
    obj_sense = problem_info["objective_info"][1]
    if (cur_obj_1 >= cur_obj_2 and obj_sense == -1) or (
        cur_obj_1 <= cur_obj_2 and obj_sense == 1
    ):
        cur_obj = cur_obj_1
        neighborhood = neighborhood_1
        neighborhood_ = neighborhood_2
        new_sol = cur_sol_1
        cur_sol = cur_sol_1
        cur_sol_ = cur_sol_2
    else:
        cur_obj = cur_obj_2
        neighborhood = neighborhood_2
        neighborhood_ = neighborhood_1
        new_sol = cur_sol_2
        cur_sol = cur_sol_2
        cur_sol_ = cur_sol_1
    for i in range(n_vars):
        if neighborhood[i] == 0 and neighborhood_[i] == 1:
            new_sol[i] = cur_sol_[i]
    new_neighborhood = repair_policy.get_repair(
        new_sol, neighborhood=None, max_size=block_size
    )
    if np.where(new_neighborhood == 1)[0].shape[0] > block_size:
        return (cur_sol, cur_obj, utils_io.SolverStatus.SIZE_LIMIT)
    else:
        new_sol, new_obj, status = p.subproblem_solve(
            problem_info, all_var, new_sol, cross_time_limit, new_neighborhood
        )
        return (new_sol, new_obj, status)


NEIGH_POLICY_DICT = {"constr_random": ConstrRandomNeighborhoodPolicy}
REPAIR_POLICY_DICT = {
    "quick": QuickRepairPolicy,
    "cautious": CautiousRepairPolicy,
}
INITIAL_POLICY_DICT = {"variable_relaxation": VariableRelaxationPolicy}


def optimize(
    logfile_path: str,
    solver: str,
    problem_data: list[
        np.ndarray
    ],  # problem data to build Problem object, including problem_info, all_var, cur_val (optional), cur_obj (optional)
    output_prob: np.ndarray,  # neural network output.
    hard_predict: np.ndarray,  # hard predict
    confidence: np.ndarray,  # confidence score
    initial_policy: str,
    repair_policy: str,
    neighborhood_policy: str,
    time_limit: float,
    obj_limit: float,
    block: float,
    crossover: bool = True,
    initial_time_limit: float = 30,
    # the time limit for the initial solution
    search_time_limit: float = 30,
    # the time limit for the neighborhood search
    cross_time_limit: float = 30,
    # the time limit for the crossover
    alpha_initial: float = 0.1,
    # the initial alpha for the initial solution
    only_repair: bool = False,
    # if True, only repair the solution and ignore the neighborhood search
    print_output: bool = True, # whether to print output to console
    *args,
    **kwargs,
):
    """
    Perform neighborhood search on the given problem.
    1. Predict a solution
    2. Get a feasible solution
    3. Generate the neighborhood of the solution
    4. solve
    """
    start_time = time.time()
    utils_io.log(f"time_limit {time_limit}", logfile_path, print_output=print_output)
    utils_io.log(f"block_size {block}", logfile_path, print_output=print_output)
    utils_io.log(f"repair_policy: {repair_policy}", logfile_path, print_output=print_output)
    utils_io.log(f"neighborhood_policy: {neighborhood_policy}", logfile_path, print_output=print_output)
    utils_io.log(f"crossover: {crossover}", logfile_path, print_output=print_output)
    p = Problem(solver, *problem_data)
    block_size = int(block * p.n_var)
    all_var, problem_info, cur_val = p.get_data()
    obj_sense = problem_info["objective_info"][1]
    initial_policy = INITIAL_POLICY_DICT[initial_policy](p)
    neighborhood_policy = NEIGH_POLICY_DICT[neighborhood_policy](p)
    repair_policy = REPAIR_POLICY_DICT[repair_policy](p)

    val_and_logit = np.zeros(
        (len(output_prob), 3)
    )  # store information of each variable: index, predicted value, confidence score
    val_and_logit[:, 0] = np.arange(len(output_prob))
    val_and_logit[:, 1] = hard_predict.reshape(-1)
    val_and_logit[:, 2] = confidence.reshape(-1)

    # step 2: get a feasible solution
    cur_val, cur_obj = initial_policy.get_feasible_solution(
        logfile_path,
        val_and_logit,
        repair_policy,
        alpha=alpha_initial,
        alpha_ub=1.0,
        max_size=p.n_var,
        initial_time_limit=initial_time_limit,
        print_output=print_output,
    )

    best_sol, best_obj = cur_val, cur_obj

    if only_repair:
        return best_sol, best_obj, time.time() - start_time

    round = 0
    while time.time() - start_time < time_limit:
        # step 3: generate the neighborhood of the solution
        neighborhoods = neighborhood_policy.get_neighborhood(
            block_size, cur_val, val_and_logit
        )
        results = []
        if time.time() - start_time >= time_limit:
            break
        for neighborhood in neighborhoods:
            neigh_result = p.subproblem_solve(
                problem_info, all_var, cur_val, search_time_limit, neighborhood
            )
            results.append(neigh_result)
            if time.time() - start_time >= time_limit:
                break

        try:
            if not results:
                utils_io.log("No valid results found", logfile_path, print_output=print_output)
                continue
            sol, obj, status = zip(*results)
            sol, obj = np.array(sol), np.array(obj)
            if obj_sense == -1:
                best_obj = np.max(obj[obj != -1])
                best_sol = sol[np.argmax(obj[obj != -1])]
            else:
                best_obj = np.min(obj[obj != -1])
                best_sol = sol[np.argmin(obj[obj != -1])]
            cur_val = best_sol
            utils_io.log(
                f"Round {round} neighborhood search best obj: {best_obj}", logfile_path, print_output=print_output
            )
        except Exception:
            if len(results) == 0:
                utils_io.log("No valid neighborhood search results", logfile_path, print_output=print_output)
            else:
                utils_io.log("Objectives are invalid", logfile_path, print_output=print_output)
                utils_io.log(f"Neighborhood search status:\n{status}", logfile_path, print_output=print_output)
                utils_io.log(f"Neighborhood search objective:\n{obj}", logfile_path, print_output=print_output)
            break

        if obj_sense == -1:
            if obj_limit is not None and best_obj >= obj_limit:
                break
        else:
            if obj_limit is not None and best_obj <= obj_limit:
                break

        # find the best crossover solution
        if crossover:
            if time.time() - start_time >= time_limit:
                break
            n_neighbor = neighborhoods.shape[0]
            cross_results = []
            for i in range(n_neighbor // 2):
                cur_sol_1, cur_sol_2 = sol[2 * i], sol[2 * i + 1]
                cur_obj_1, cur_obj_2 = obj[2 * i], obj[2 * i + 1]
                cross_result = cross_neighborhood(
                    p,
                    repair_policy,
                    block_size,
                    neighborhoods[2 * i],
                    neighborhoods[2 * i + 1],
                    cur_sol_1,
                    cur_sol_2,
                    cur_obj_1,
                    cur_obj_2,
                    cross_time_limit,
                )
                cross_results.append(cross_result)

                if time.time() - start_time >= time_limit:
                    break

            try:
                cross_sol, cross_obj, cross_status = zip(*cross_results)
                cross_sol, cross_obj = np.array(cross_sol), np.array(cross_obj)
                new_obj = np.hstack((obj, cross_obj))
                new_sol = np.vstack((sol, cross_sol))
                if obj_sense == -1:
                    best_obj = np.max(new_obj[new_obj != -1])
                    best_sol = new_sol[np.argmax(new_obj[new_obj != -1])]
                else:
                    best_obj = np.min(new_obj[new_obj != -1])
                    best_sol = new_sol[np.argmin(new_obj[new_obj != -1])]
                cur_val = best_sol
                utils_io.log(
                    f"Round {round} crossover best obj: {best_obj}", logfile_path, print_output=print_output
                )
            except Exception:
                if len(cross_results) == 0:
                    utils_io.log("No valid crossover results", logfile_path, print_output=print_output)
                else:
                    utils_io.log("Cross objectives are invalid", logfile_path, print_output=print_output)
                    utils_io.log(f"Crossover status:\n{cross_status}", logfile_path, print_output=print_output)
                    utils_io.log(f"Crossover objective:\n{cross_obj}", logfile_path, print_output=print_output)
                break

            if obj_sense == -1:
                if obj_limit is not None and best_obj >= obj_limit:
                    break
            else:
                if obj_limit is not None and best_obj <= obj_limit:
                    break
        round += 1
    duration = time.time() - start_time
    utils_io.log(
        f"Optimization finished. Best obj: {best_obj}, time used: {duration:.2f} seconds",
        logfile_path,
        print_output=print_output,
    )
    return best_sol, best_obj, duration
