"""
Get information of the optimization model from different solvers.
"""

from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import pickle
import gurobipy as gp
import sympy as sp
import argparse
import utils_io


class ProbModelExtractor(ABC):
    """Abstract base class for extracting information from optimization models"""

    @abstractmethod
    def get_vars_info(self, model) -> dict[str, dict[str, any]]:
        """
        Extract variable information from the model.

        Returns:
            Dict containing variable info like names, types, bounds, etc. Format: {
                "var1_name": {
                    "type": "C" | "B" | "I",
                    "lb": float,
                    "ub": float,
                    ...  # other attributes as needed
                },
                "var2_name": ...
            }
        """
        pass

    def get_solution_values(
        self, vars_info, sol_dict={}, default_for_missing=0
    ) -> list[float]:
        """
        Get solution values for the model.

        Parameters:
            vars_info (Dict[str, Dict[str, Any]]): Variable information dictionary.
            sol_dict (Dict[str, float]): Dictionary containing solution values.
            default_for_missing (float): Default value to use if a variable is not found in the solution dictionary.
        Returns:
            list[float]: List of solution values corresponding to the variables.
        """
        solution_values = []
        for var_name, _ in vars_info.items():
            if var_name in sol_dict:
                solution_values.append(sol_dict[var_name])
            else:
                # If no solution is found, use the default_for_missing value
                solution_values.append(default_for_missing)
        return solution_values

    @abstractmethod
    def get_expr_terms(self, expr) -> list[tuple[float, dict[str, int]]]:
        """
        Extract terms from an expression.

        Parameters:
            expr: the expression to extract terms from.
        Returns:
            List of terms in the expression, where each term is a tuple (coefficient, {variable_name: variable degree}).
            Example: [(1, {'x': 2}), (2, {'y': 1, 'z': 1})] represents two terms with coefficients 1 and 2, the former has one 2-order variable x, while the latter has a 1-order variable y and an 1-order variable z.
        """
        pass

    @abstractmethod
    def get_objective_info(
        self, model
    ) -> tuple[list[tuple[float, dict[str, int]]], int]:
        """
        Extract terms from the objective function of the model.

        Parameters:
            model: The optimization model object.
        Returns:
            Tuple containing:
                - List of terms in the objective function, where each term is a tuple (coefficient, {variable_name: variable degree}).
                - Objective sense (1 for minimization, -1 for maximization).
        """
        pass

    @abstractmethod
    def get_constraints_info(
        self, model
    ) -> tuple[list[list[tuple[float, dict[str, int]]]], list[float], list[str]]:
        """
        Extract information about the constraints in the model.

        Parameters:
            model: The optimization model object.

        Returns:
            Tuple containing:
                - List of constraints, where each constraint is represented as a list of terms (coefficient, {variable_name: variable degree}).
                - List of right-hand side values for each constraint.
                - List of constraint senses.
        """
        pass

    @abstractmethod
    def get_instance_info(self, model, sol_dict, default_for_missing=0) -> dict:
        """
        Extracts instance information from the model.

        Parameters:
            model: The optimization model object.
        Returns:
            dict: A dictionary containing the extracted instance information, including
                - variable_info
                - objective_info
                - constraint_info
                - solution_values
        """
        pass


# Concrete implementation for Gurobi QP models
class GurobiQPExtractor(ProbModelExtractor):
    """Extractor for Gurobi quadratic programming models"""

    def get_vars_info(self, model) -> dict[str, dict[str, any]]:
        """Extract variable information from a Gurobi QP model.

        Parameters:
            model (gurobipy.Model): The Gurobi model object.
        Returns:
            dict: A dictionary containing variable_name - variable attributes (lb, ub, type).
        """
        vars_info = {}
        for var in model.getVars():
            vars_info[var.VarName] = {
                "lb": var.LB,
                "ub": var.UB,
                "type": var.VType,
            }
        return vars_info

    def get_expr_terms(
        self, expr: gp.LinExpr | gp.QuadExpr
    ) -> list[tuple[float, dict[str, int]]]:
        """
        Extract terms from a Gurobi expression in a quadratic problem model, with linear or quadratic terms.

        Parameters:
            expr (gurobipy.LinExpr or gurobipy.QuadExpr): The Gurobi linear expression.

        Returns:
            List of terms in the expression, where each term is a tuple (coefficient, {variable_name: variable degree}).
        """
        terms = []
        if isinstance(expr, gp.LinExpr):
            for i in range(expr.size()):
                var = expr.getVar(i)
                coeff = expr.getCoeff(i)
                terms.append((coeff, {var.VarName: 1}))
        elif isinstance(expr, gp.QuadExpr):
            # handle linear part
            linear_expr = expr.getLinExpr()
            for i in range(linear_expr.size()):
                var = linear_expr.getVar(i)
                coeff = linear_expr.getCoeff(i)
                terms.append((coeff, {var.VarName: 1}))
            # handle quadratic part
            for i in range(expr.size()):
                var1, var2 = expr.getVar1(i), expr.getVar2(i)
                coeff = expr.getCoeff(i)
                if var1.VarName == var2.VarName:
                    # self-product, e.g., x^2
                    terms.append((coeff, {var1.VarName: 2}))
                else:
                    # cross-product, e.g., xy
                    terms.append((coeff, {var1.VarName: 1, var2.VarName: 1}))
        else:
            raise TypeError(
                "Expression must be a Gurobi LinExpr or QuadExpr. Received: {}".format(
                    type(expr)
                )
            )
        return terms

    def get_objective_info(
        self, model
    ) -> tuple[list[tuple[float, dict[str, int]]], int]:
        """
        Extract terms from the objective function of a Gurobi QP model.

        Parameters:
            model (gurobipy.Model): The Gurobi model object.

        Returns:
            objective_terms (list): List of terms in the objective function, where each term is a tuple (coefficient, {variable_name: variable degree}).
            objective_sense (int): 1 if the objective is to minimize, -1 if to maximize.
        """
        obj_expr = model.getObjective()
        objective_terms = self.get_expr_terms(obj_expr)
        objective_sense = model.getAttr("ModelSense")
        return objective_terms, objective_sense

    def get_constraints_info(
        self, model
    ) -> tuple[list[list[tuple[float, dict[str, int]]]], list[float], list[str]]:
        """
        Extract constraints information from a Gurobi QP model.

        Parameters:
            model (gurobipy.Model): The Gurobi model object.

        Returns:
            constraint_term_list (list[list[tuple[float, dict]]]): List of constraints, each containing a list of terms in the constraint expression, where each term is a tuple (coefficient, {variable_name: variable degree}).
            rhs_list (list[float]): List of right-hand side values for each constraint.
            sense_list (list[str]): List of constraint senses, where each sense is one of "<", ">", or "=".
        """
        constraint_term_list = []
        rhs_list = []
        sense_list = []
        # linear constraints
        for constr in model.getConstrs():
            rhs_list.append(constr.getAttr("RHS"))
            sense_list.append(constr.getAttr("Sense"))
            expr = model.getRow(constr)
            terms = self.get_expr_terms(expr)
            constraint_term_list.append(terms)
        # quadratic constraints
        for qconstr in model.getQConstrs():
            rhs_list.append(qconstr.getAttr("QCRHS"))
            sense_list.append(qconstr.getAttr("QCSense"))
            expr = model.getQCRow(qconstr)
            terms = self.get_expr_terms(expr)
            constraint_term_list.append(terms)
        return constraint_term_list, rhs_list, sense_list

    def get_instance_info(self, model, sol_dict, default_for_missing=0) -> dict:
        """
        Extracts instance information from the Gurobi QP model.

        Parameters:
            model (gurobipy.Model): The Gurobi model object.
        Returns:
            dict: A dictionary containing the extracted instance information, including
                - variable_info
                - objective_info
                - constraint_info
                - solution_values
        """
        variable_info = self.get_vars_info(model)
        solution_values = self.get_solution_values(
            variable_info, sol_dict, default_for_missing
        )
        objective_info = self.get_objective_info(model)
        constraint_info = self.get_constraints_info(model)

        return {
            "variable_info": variable_info,
            "solution_values": solution_values,
            "objective_info": objective_info,
            "constraint_info": constraint_info,
        }


class AMPLPolyExtractor(ProbModelExtractor):
    """Extractor for AMPL polynomial programming problems"""

    def _get_sympy_var(self, vars_info: dict[str, dict[str, any]]):
        """Generate sympy variables from variable information dictionary.

        Parameters:
        -----------
        vars_info : dict
            A dictionary containing variable_name - variable attributes (lb, ub, type, other information).

        Returns:
        --------
        sym_vars : dict
            A dictionary mapping variable names to sympy symbols.
        """

        sym_vars = {}
        created_base_names = {}
        for var_name, var_info in vars_info.items():
            if var_info["ampl_indexed"]:
                # indexed variable
                base_name = var_info["ampl_base_name"]
                if base_name not in created_base_names:
                    sym_vars[base_name] = sp.IndexedBase(base_name)
                    created_base_names[base_name] = True
            else:
                # non-indexed variable
                sym_vars[var_name] = sp.symbols(var_name)
        return sym_vars

    def _expr_split(self, expr, max_term_num=100):
        """Split the expression into several smaller expressions. Notice that this function now only realize a weak function from its ideal version.

        Note that the string expression should follow:
        1. variables, coefficients and operators are not separated by spaces
        2. terms are separated by '-'. For example, 'x^2+2*y^2-3*z' should be written as 'x^2+2*y^2+-3*z'
        3. coefficients should be at the beginning of the terms. When a coefficient is 1, it can be ommitted.
        4. multiplication is represented by '*' and can not be omitted.
        5. exponentiation is represented by '^'. Exponenents should be positive integers, "^1" and "^0" should not appear, while for other exponenets "^" should not be omitted.
        6. branckets could appear but only '()'

        Parameters:
        -----------
        expr : sympy expression
            A sympy expression representing the objective function or constraint
        max_term_num: int
            Maximum number of terms in each sub-expression

        Returns:
        --------
        sub_exprs: list[str]
            list of sub-expressions
        """
        raw_split0 = expr.split(
            ")+"
        )  # ')-' shoudl also be considered, but I'd leave it for now
        raw_split = []
        for raw_expr in raw_split0:
            if "(" in raw_expr:
                raw_split.append(raw_expr + ")")
            else:
                raw_split.append(raw_expr)
        sub_exprs = []
        while raw_split:
            sub_expr = ""
            num_term = 0  # counting terms is not correctly implemented. But I'd leave it for now.
            while num_term < max_term_num:
                if len(raw_split) == 0:
                    break
                sub_expr += f"{raw_split[0]}+"
                num_term += len(
                    raw_split[0].split("+")
                )  # cautious, for expressions like (a+b)(c+d), this count is wrong.
                raw_split = raw_split[1:]
            if sub_expr:
                sub_exprs.append(sub_expr[:-1])
        return sub_exprs

    def get_vars_info(self, model) -> dict[str, dict[str, any]]:
        """Extract variable information from a Gurobi QP model.

        Parameters:
            model (gurobipy.Model): The Gurobi model object.
        Returns:
            dict: A dictionary containing variable_name - variable attributes (lb, ub, type, other_info).
        """
        vars_info = {}

        for var_name, var_detail in model.get_variables():
            var_obj = model.getVariable(var_name)
            var_type = "C"
            if "binary" in str(var_detail):
                var_type = "B"
            elif "integer" in str(var_detail):
                var_type = "I"

            # indexed variables
            if var_obj.indexarity() > 0:
                for _, inst in enumerate(var_obj.instances()):
                    index_info = inst[0]
                    if isinstance(index_info, int):
                        var_individual_name = f"{var_name}[{index_info}]"
                    else:
                        var_individual_name = (
                            f"{var_name}[{','.join(str(i) for i in index_info)}]"
                        )
                    var = var_obj.get(index_info)
                    vars_info[var_individual_name] = {
                        "lb": var.lb(),
                        "ub": var.ub(),
                        "type": var_type,
                        "ampl_indexed": True,
                        "ampl_base_name": var_name,
                        "ampl_index": index_info,
                    }
            else:
                vars_info[var_name] = {
                    "lb": var_obj.lb(),
                    "ub": var_obj.ub(),
                    "type": var_type,
                    "ampl_indexed": False,
                }
        return vars_info

    def get_expr_terms(self, expr) -> list[tuple[float, dict[str, int]]]:
        """
        Extract terms from an AMPL polynomial expression.

        Parameters:
            A sympy expression representing the objective function or constraint
        Returns:
            List of terms in the expression, where each term is a tuple (coefficient, {variable_name: variable degree}).
        """
        expr_terms = []
        terms = expr.split("+")
        for term in terms:
            if len(term) == 0:
                # skip empty terms. For standard expression, this should not happen
                continue
            negative_flag = False
            if term[0] == "-":
                negative_flag = True
                term = term[1:]
            multipliers = term.split("*")
            # get coefficient
            try:
                coeff = float(multipliers[0])
                multipliers = multipliers[1:]
                if coeff == 0:
                    # skip zero terms. Generally, this should not happen
                    continue
            except ValueError:
                coeff = 1
            if negative_flag:
                coeff = -coeff
            # get variable information
            var_info = {}
            for multiplier in multipliers:
                var_expr = multiplier.split("^")
                if len(var_expr) == 1:
                    var_info[var_expr[0]] = 1
                elif len(var_expr) == 2:
                    var_info[var_expr[0]] = int(var_expr[1])
                else:
                    raise ValueError(
                        f"Invalid term format: {term}"
                    )  # usually this should not happen

            expr_terms.append((coeff, var_info))

        return expr_terms

    def get_objective_info(
        self, model, vars_info=None
    ) -> tuple[list[tuple[float, dict[str, int]]], int]:
        """
        Extract terms from the objective function of an AMPL polynomial model.

        Parameters:
            model (amplpy.AMPL): The AMPL model object.
            vars_info : dict
                A dictionary containing variable_name - variable attributes (lb, ub, type, other information).

        Returns:
            objective_terms (list): List of terms in the objective function, where each term is a tuple (coefficient, {variable_name: variable degree}).
            objective_sense (int): 1 if the objective is to minimize, -1 if to maximize.
        """
        if vars_info is None:
            vars_info = self.get_vars_info(model)
        sym_vars = self._get_sympy_var(vars_info)
        # Get the objective function and turn it into a standard form
        obj_names = [objective[0] for objective in model.get_objectives()]
        obj_expr = (
            model.getOutput(f"expand {obj_names[0]};")
            .split(";\n\n")[0]
            .split(":")[1]
            .replace("\n\t", "")
        )
        obj_sub_exprs = self._expr_split(obj_expr.replace(" ", ""), max_term_num=100)

        # Extract the terms from the objective function
        obj_terms = []
        for sub_expr in obj_sub_exprs:
            sub_expr_sp = sp.sympify(sub_expr, locals=sym_vars, convert_xor=True)
            sub_expand_sp = sp.expand(sub_expr_sp)
            sub_stand_str = (
                str(sub_expand_sp)
                .replace(" ", "")
                .replace("-", "+-")
                .replace("**", "^")
            )
            obj_terms += self.get_expr_terms(sub_stand_str)

        objective_sense = 1 if model.obj[obj_names[0]].minimization() else -1

        return obj_terms, objective_sense

    def get_constraints_info(
        self, model, vars_info=None
    ) -> tuple[list[list[tuple[float, dict[str, int]]]], list[float], list[str]]:
        """
        Extract constraints information from an AMPL polynomial model.

        Parameters:
            model (amplpy.AMPL): The AMPL model object.

        Returns:
            constraint_term_list (list[list[tuple[float, dict]]]): List of constraints, each containing a list of terms in the constraint expression, where each term is a tuple (coefficient, {variable_name: variable degree}).
            rhs_list (list[float]): List of right-hand side values for each constraint.
            sense_list (list[str]): List of constraint senses, where each sense is one of "<", ">", or "=".
            sos1_orbits (list[list[int]]): A list of orbits where each orbit contains indices of constraints that are part of the same SOS1 group. SOS1 constraints allow at most one variable in the group to be non-zero.
            independent_indices (list[int]): A list of indices of constraints that are independent (not part of any group of SOS1 constraints).
        """
        constraint_terms = []
        rhs_list = []
        sense_list = []
        sos1_groups = []
        if vars_info is None:
            vars_info = self.get_vars_info(model)
        sym_vars = self._get_sympy_var(vars_info)
        # get names of all constraints
        constraint_names = [constraint[0] for constraint in model.get_constraints()]

        ### parse each constraint. Note that in ampl models, constraints can be either an individual/scalar constraint or a set of constraints.
        for constraint_name in constraint_names:
            const = model.getConstraint(constraint_name)
            const_set = []
            if const.isScalar():
                # parse individual/scalar constraint
                const_set.append(model.getOutput(f"expand {constraint_name};"))
            else:
                # parse set of constraints
                consts = model.getOutput(f"expand {constraint_name};").split(";\n\n")[
                    :-1
                ]
                const_set.extend(consts)
            # parse each constraint under this constraint name
            for const_str in const_set:
                const_expr = (
                    const_str.split(":\n\t")[1].replace("\n\t", "").replace(" ", "")
                )
                # get sense and rhs of the constraint
                if "<" in const_expr or "<=" in const_expr:
                    sense = "<"
                    lhs, rhs = (
                        const_expr.split("<=")
                        if "<=" in const_expr
                        else const_expr.split("<")
                    )
                elif ">" in const_expr or ">=" in const_expr:
                    sense = ">"
                    lhs, rhs = (
                        const_expr.split(">=")
                        if ">=" in const_expr
                        else const_expr.split(">")
                    )
                elif "==" in const_expr or "=" in const_expr:
                    sense = "="
                    lhs, rhs = (
                        const_expr.split("==")
                        if "==" in const_expr
                        else const_expr.split("=")
                    )
                else:
                    raise ValueError(f"Invalid constraint sense: {const_str}")
                rhs_list.append(float(rhs))
                sense_list.append(sense)

                # Extract the terms from the constraint's lhs
                lhs_expr_sp = sp.sympify(lhs, locals=sym_vars, convert_xor=True)
                lhs_standard_str = (
                    str(sp.expand(lhs_expr_sp))
                    .replace(" ", "")
                    .replace("-", "+-")
                    .replace("**", "^")
                )
                # Extract the terms from the constraint's lhs
                lhs_terms = self.get_expr_terms(lhs_standard_str)
                constraint_terms.append(lhs_terms)
                # check whether it's a sos1 constraint. Such constraints should be of the form "sum(x) = 1" for binary variables
                sos1_falg = True
                if sense == "=" and float(rhs) == 1:
                    for coeff, var_info in lhs_terms:
                        # check if the coefficient is 1 and if the terms are linear
                        sos1_condition1 = (
                            (coeff == 1)
                            and (len(var_info) == 1)
                            and (list(var_info.values())[0] == 1)
                        )
                        # check if all variables are binary variables
                        if sos1_condition1:
                            var_name = list(var_info.keys())[0]
                            var_type = vars_info[var_name]["type"]
                            if var_type != "B":
                                sos1_falg = False
                                break
                        else:
                            sos1_falg = False
                            break
                else:
                    sos1_falg = False

                if sos1_falg:
                    # get the indices of the variables in this constraint
                    var_indices = []
                    for _, var_info in lhs_terms:
                        var_name = list(var_info.keys())[0]
                        var_indices.append(list(vars_info.keys()).index(var_name))
                    sos1_groups.append(var_indices)
        independent_indices = set(list(range(len(vars_info)))) - set(
            [i for group in sos1_groups for i in group]
        )
        return (
            constraint_terms,
            rhs_list,
            sense_list,
            sos1_groups,
            list(independent_indices),
        )

    def get_instance_info(self, model, sol_dict, default_for_missing=0) -> dict:
        """
        Extracts instance information from the AMPL polynomial model.

        Parameters:
            model (amplpy.AMPL): The AMPL model object.
        Returns:
            dict: A dictionary containing the extracted instance information, including
                - variable_info
                - objective_info
                - constraint_info
                - solution_values
        """
        variable_info = self.get_vars_info(model)
        solution_values = self.get_solution_values(
            variable_info, sol_dict, default_for_missing
        )
        objective_info = self.get_objective_info(model, variable_info)
        constraint_info = self.get_constraints_info(model, variable_info)

        return {
            "variable_info": variable_info,
            "solution_values": solution_values,
            "objective_info": objective_info,
            "constraint_info": constraint_info,
        }


def extract_one_instance(
    model_type: str, instance_path: str, has_sol: bool = False, **kwargs
):
    """Extract instance information from a given optimization model.

    Parameters:
        model_type (str): Type of the model ('gurobi_qp' or 'ampl_poly').
        instance_path (str): Path to the model file.
        has_sol (bool): Whether a solution file is provided.
        **kwargs: Additional arguments for solution extraction.
    Returns:
        dict: Extracted instance information.
    """
    sol_dict = {}
    if model_type in ["gurobi_qp", "GurobiQP"]:
        model = gp.read(instance_path)
        extractor = GurobiQPExtractor()
        if has_sol:
            sol_path = instance_path.rsplit(".", 1)[0] + ".sol"
            sol_dict = utils_io.load_solution(sol_path, model_type)
        instance_info = extractor.get_instance_info(model, sol_dict, **kwargs)
    elif model_type in ["ampl_poly", "AMPLPoly"]:
        model_structure_path = kwargs.get("model_structure_path", None)
        kwargs.pop("model_structure_path", None)
        if model_structure_path is None:
            raise ValueError(
                "For AMPL polynomial models, 'model_structure_path' must be provided in kwargs."
            )
        problem_type = kwargs.get("problem_type", "pcflp")
        kwargs.pop("problem_type", None)
        model = utils_io.load_ampl_poly_instance(
            model_structure_path, problem_type, instance_path, **kwargs
        )
        if has_sol:
            file_dir, file_name = instance_path.rsplit("/", 1)
            base_part = file_name.split("_", 1)[1] if "_" in file_name else file_name
            sol_name = f"solution_{base_part}"
            sol_path = f"{file_dir}/{sol_name}"
            sol_dict = utils_io.load_solution(sol_path, model_type)
        extractor = AMPLPolyExtractor()
        instance_info = extractor.get_instance_info(model, sol_dict, **kwargs)
    else:
        raise ValueError(f"Unsupported model type: {model_type}.")
    return instance_info


def _process_instance_task(instance_path, info_path, model_type, has_sol, kwargs):
    """Worker helper to extract and persist a single instance."""
    instance_info = extract_one_instance(
        model_type, instance_path, has_sol=has_sol, **kwargs
    )
    info_dirname = os.path.dirname(info_path)
    if info_dirname:
        os.makedirs(info_dirname, exist_ok=True)
    with open(info_path, "wb") as f:
        pickle.dump(instance_info, f)
    return instance_path, info_path


if __name__ == "__main__":
    # get instance information and save them into a pickle file
    parser = argparse.ArgumentParser(
        description="Extract instance information from optimization models."
    )
    parser.add_argument(
        "--instance_dir",
        type=str,
        required=False,
        help="Directory containing optimization model files.",
    )
    parser.add_argument(
        "--info_dir",
        type=str,
        required=False,
        help="Directory to save extracted instance information.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        required=False,
        help="Type of the model ('gurobi_qp' or 'ampl_poly').",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=16,
        help="Number of parallel workers for representation generation.",
    )
    parser.add_argument(
        "--has_sol", action="store_true", help="Whether solution files are provided."
    )
    args, other_args = parser.parse_known_args()
    kwargs = {}
    for i in range(0, len(other_args), 2):
        key = other_args[i].lstrip("-")
        value = other_args[i + 1]
        kwargs[key] = value

    instance_dir = args.instance_dir
    info_dir = args.info_dir
    model_type = args.model_type
    has_sol = args.has_sol
    # print(has_sol)
    os.makedirs(info_dir, exist_ok=True)
    instances = []
    if model_type in ["gurobi_qp", "GurobiQP"]:
        instances = [f for f in os.listdir(instance_dir) if f.endswith(".lp")]
    elif model_type in ["ampl_poly", "AMPLPoly"]:
        instances = [
            f
            for f in os.listdir(instance_dir)
            if f.endswith(".json") and f.startswith("PCFLP_")
        ]
    else:
        raise ValueError(f"Unsupported model type: {model_type}.")
    num_workers = max(1, args.num_workers)
    total_instances = len(instances)
    pending_tasks = []
    for file_idx, file in enumerate(instances):
        info_path = os.path.join(info_dir, file.rsplit(".", 1)[0] + ".pkl")
        file = os.path.join(instance_dir, file)
        if os.path.exists(info_path):
            print(
                f"{file_idx}/{total_instances}: Instance info for {file} already exists at {info_path}, skipping extraction."
            )
            continue  # skip already processed instances
        pending_tasks.append((file_idx, file, file, info_path))

    if not pending_tasks:
        print("All eligible instances already processed.")
    elif num_workers == 1:
        # sequential fallback when only one worker is requested
        for file_idx, file, instance_path, info_path in pending_tasks:
            _process_instance_task(
                instance_path, info_path, model_type, has_sol, dict(kwargs)
            )
            print(
                f"{file_idx}/{total_instances}: Extracted and saved instance info for {file} to {info_path}"
            )
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_to_meta = {}
            for file_idx, file, instance_path, info_path in pending_tasks:
                future = executor.submit(
                    _process_instance_task,
                    instance_path,
                    info_path,
                    model_type,
                    has_sol,
                    dict(kwargs),
                )
                future_to_meta[future] = (file_idx, file, info_path)
            for future in as_completed(future_to_meta):
                file_idx, file, info_path = future_to_meta[future]
                try:
                    future.result()
                    print(
                        f"{file_idx}/{total_instances}: Extracted and saved instance info for {file} to {info_path}"
                    )
                except Exception as exc:
                    print(
                        f"{file_idx}/{total_instances}: Failed to process {file}: {exc}"
                    )
                    raise

    """
    # test: "/home/mli2/programFiles/HNN_IPHD/data/train/problem/qis/easy/"
    instance_dir = "/home/mli2/programFiles/HNN_IPHD/data/train/problem/qis/easy/"
    info_dir = "/home/mli2/programFiles/HNN_IPHD/data/train/instance_info/qis/easy/"
    os.makedirs(info_dir, exist_ok=True)
    model_type = "GurobiQP"
    instances = [os.path.join(instance_dir, f) for f in os.listdir(instance_dir) if f.endswith(".lp")]
    for file_idx, file in enumerate(instances):
        if file.endswith(".lp"):
            instance_path = os.path.join(instance_dir, file)
            instance_info = extract_one_instance(model_type, instance_path, has_sol=True)
            # save instance_info into a pickle file
            info_path = os.path.join(info_dir, file.rsplit('.', 1)[0] + '.pkl')
            with open(info_path, 'wb') as f:
                pickle.dump(instance_info, f)
            print(f"{file_idx}/{len(instances)}: Extracted and saved instance info for {file} to {info_path}")
    """
