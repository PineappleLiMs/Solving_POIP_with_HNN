"""
Implementation of utility functions for input/output operations.
"""

import datetime
import pickle
import pandas as pd
import json
from enum import Enum


class SolverStatus(Enum):
    OPTIMAL = 2  # success
    INFEASIBLE = 3  # fail
    UNBOUNDED = 5  # fail
    TIME_LIMIT = 9  # success
    SOLUTION_LIMIT = 10  # success
    INTERRUPTED = 11  # fail
    OBJ_LIMIT = 15  # success
    GAP_LIMIT = 18  # success
    SIZE_LIMIT = 19  # fail
    UNDEFINED = 20  # undefined
    OTHER_FEASIBLE = 21  # success with feasible solution


def log(str, logfile=None, print_output=True):
    str = f"[{datetime.datetime.now()}] {str}"
    if print_output:
        print(str)
    if logfile is not None:
        with open(logfile, mode="a") as f:
            print(str, file=f)


def load_solution(sol_path, model_type):
    """
    Load solution from a file. Note that no checking is performed here.

    Parameters:
    -----------
    sol_path : str
        Path to the solution file.
    model_type : str
        Type of the model (e.g., 'gurobi_qp', 'ampl_poly', etc.)
    """
    sol_dict = {}
    if model_type in ["gurobi_qp", "GurobiQP"]:
        # Gurobi solution stored in a .sol file
        with open(sol_path, "r") as f:
            for line in f:
                if not line.startswith("#") and not line.startswith("obj"):
                    line = line.strip().split()
                    sol_dict[line[0]] = float(line[1])
    elif model_type in ["ampl_poly", "AMPLPoly"]:
        # AMPL's polynomial model solutions are stored in JSON files
        with open(sol_path, "r") as f:
            sol_dict = json.load(f)
            for var_name, var_value in sol_dict["variable_values"].items():
                sol_dict[var_name] = float(var_value)
    else:
        raise ValueError(f"Unsupported model type: {model_type}.")

    return sol_dict


def load_ampl_poly_instance(
    model_structure_path, problem_type, instance, **other_information
):
    """
    Reads a instance from a local file and builds an AMPL model.

    Parameters:
    -----------
    model_structure_path : str
        Path to the AMPL model file (e.g., 'PCFLP.mod')
    problem_type : str
        Type of the problem (e.g., 'pcflp' or "minlplib")
    instance : dict or str
        Dictionary that contains instance data, or path to the JSON file containing the instance data
    other_information : dict, optional
        Additional information for the model (e.g., 'alpha', 'beta')

    Returns:
    --------
    amplpy.AMPL
        An AMPL object with the model loaded and ready to solve
    """
    from amplpy import AMPL

    model = AMPL()
    model.read(model_structure_path)

    if problem_type in ["cflp", "CFLP", "pcflp", "PCFLP", "CFLPTC", "cflptc"]:
        # parse pcflp instance. Such instances' structure should be loaded from model_structure_path, while the instance data should be loaded from "instance".
        if isinstance(instance, str):
            # Load the instance data from JSON file
            with open(instance, "r") as f:
                instance = json.load(f)
        elif not isinstance(instance, dict):
            raise ValueError(
                f"{problem_type} instance should be a dictionary or a path to a JSON file."
            )
        # get values of alpha and beta from other_information
        alpha = other_information.get("alpha", 1)
        beta = other_information.get("beta", 4)
        # Extract instance data
        num_customers = instance["num_customers"]
        num_facilities = instance["num_facilities"]
        customer_demands = instance["customer_demands"]
        facility_capacities = instance["facility_capacities"]
        facility_fixed_costs = instance["facility_fixed_costs"]
        distances = instance["distances"]
        total_traffic = instance["total_traffic"]
        background_traffic = instance["background_traffic"]
        ### Set data in the model
        # Set for customers
        customers_set = model.getSet("CUSTOMERS")
        customers_set.setValues([i + 1 for i in range(num_customers)])

        # Set for facilities
        facilities_set = model.getSet("FACILITIES")
        facilities_set.setValues([j + 1 for j in range(num_facilities)])

        # Customer demands
        demand_param = model.getParameter("demand")
        for i in range(num_customers):
            demand_param.set(i + 1, customer_demands[i])

        # Facility capacities
        capacity_param = model.getParameter("capacity")
        for j in range(num_facilities):
            capacity_param.set(j + 1, facility_capacities[j])

        # Facility fixed costs
        fixed_cost_param = model.getParameter("fixed_cost")
        for j in range(num_facilities):
            fixed_cost_param.set(j + 1, facility_fixed_costs[j])

        # Distances between customers and facilities
        distance_param = model.getParameter("distance")
        distance_df = pd.DataFrame(
            distances,
            index=[i + 1 for i in range(num_customers)],
            columns=[j + 1 for j in range(num_facilities)],
        )
        distance_param.setValues(distance_df)

        # Total traffic
        total_traffic_param = model.getParameter("total_traffic")
        for j in range(num_facilities):
            total_traffic_param.set(j + 1, total_traffic[j])

        # Background traffic
        background_traffic_param = model.getParameter("background_traffic")
        for j in range(num_facilities):
            background_traffic_param.set(j + 1, background_traffic[j])
        # Coefficients for the objective function
        alpha_param = model.getParameter("alpha")
        alpha_param.set(alpha)
        beta_param = model.getParameter("beta")
        beta_param.set(beta)
        return model

    elif problem_type in ["minlplib", "MINLP"]:
        # parse minlplib instance. Both model sructure and instance data are stored in the model_structure_path
        return model

    else:
        raise ValueError(
            f"Unsupported problem type: {problem_type}. Supported types are 'cflp', 'pcflp', and 'minlplib'."
        )
