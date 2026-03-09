"""
Get representations of the optimization instances
"""

from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor
import os
import pickle
import numpy as np
import argparse
from scipy.sparse import coo_matrix


class ReprGenerator(ABC):
    """Abstract base class for transforming integer programming problems into representations suitable for machine learning models."""

    def __init__(self, encoder_name, extractor):
        """
        Initializes the Encoder with a ProbModelExtractor instance.

        Parameters:
            encoder_name (str): Name of the encoder.
            extractor (instance_info_get.ProbModelExtractor): An instance of ProbModelExtractor to extract information from the model.
        """
        self.encoder_name = encoder_name
        self.extractor = extractor

    @abstractmethod
    def get_repr(self, instance_info) -> dict:
        """
        Transform the problem into a representation suitable for machine learning models.

        Parameters:
            instance_info (dict): A dictionary containing the extracted instance information, including
                - variable_info
                - solution_values
                - objective_info
                - constraint_info

        Returns:
            Dict: Encoded representation of the problem.
        """
        pass

    def read_solution(self, solution_path) -> dict:
        """
        Reads a solution file and returns its content.

        Parameters:
            file_path (str): The path to the solution file.

        Returns:
            solution_dict (dict): A dictionary containing the solution data.
        """
        if not os.path.exists(solution_path):
            # print(f"Solution file {solution_path} not found")
            raise FileNotFoundError(f"Solution file {solution_path} not found")
        file_type = os.path.splitext(solution_path)[1]
        if file_type == ".sol":
            # Read solution from a .sol file, which is typically from exact solvers like Gurobi, SCIP, CPLEX, etc.
            solution_dict = {}
            with open(solution_path, "r") as f:
                for line in f:
                    if line.startswith("#") or not line.strip():
                        line = line.strip().split()
                        solution_dict[line[0]] = float(line[1])
        else:
            raise ValueError(
                f"Unsupported file type: {file_type}. Only .sol files are supported."
            )
        return solution_dict

    def extract_instance_info(
        self, prob_model, solution_dict, default_for_solution_missing
    ) -> dict:
        """
        Extracts instance information from the model using the provided extractor.

        Parameters:
            extractor (ProbModelExtractor): An instance of ProbModelExtractor to extract information.
            prob_model: The problem model from which to extract information.
            default_for_solution_missing (float): Default value to use if a variable is not found in the solution dictionary.

        Returns:
            dict: A dictionary containing the extracted instance information.
        """
        return self.extractor.get_instance_info(
            prob_model, solution_dict, default_for_solution_missing
        )

    def check_repr_exists(self, prob_model_name, output_path) -> bool:
        """
        Checks if the encoded representation already exists in the specified output path.

        Parameters:
            prob_model_name (str): The name of the problem model.
            output_path (str): The path where the encoded representation is expected to be saved.
        Returns:
            bool: True if the encoded representation file exists, False otherwise.
        """
        return os.path.exists(os.path.join(output_path, f"{prob_model_name}.pkl"))

    def save_repr(self, representation, prob_model_name, output_path):
        """
        Saves the encoded representation to a specified output path.

        Parameters:
            representation (dict): The encoded representation to save.
            prob_model_name (str): The name of the problem model.
            output_path (str): The path where the encoded representation will be saved.
        """
        if self.check_repr_exists(prob_model_name, output_path):
            print(
                f"Encoder {self.encoder_name}: Encoded representation for {prob_model_name} already exists at {output_path}, skipping saving."
            )
        pickle.dump(
            (prob_model_name, representation),
            open(os.path.join(output_path, f"{prob_model_name}.pkl"), "wb"),
        )
        print(
            f"Encoder {self.encoder_name} saved encoded representation for {prob_model_name} to {output_path}"
        )


class ReprOurs(ReprGenerator):
    """Hypergraph representation for IPHD proposed in our paper.

    Ref: to be added.
    """

    def __init__(self, extractor):
        super().__init__("NeuralQP", extractor)
        self.onehot_encoding = {
            "VTYPE": {
                "C": [1, 0, 0],
                "B": [0, 1, 0],
                "I": [0, 0, 1],
            },  # one-hot encoding for variable types
            "CTYPE": {
                "<": [1, 0, 0],
                ">": [0, 1, 0],
                "=": [0, 0, 1],
            },  # one-hot encoding for constraint senses
            "OBJ": {
                1: [1, 0],
                -1: [0, 1],
            },
        }

    def get_variable_features(self, variable_info_list, objective_terms) -> np.array:
        """
        Extract features from variable information for hypergraph representation.

        Parameters:
            variable_info_list (Dict[str, Dict[str, Any]]): Dictionary containing variable information obtained from ./utils_loaders.ProbModelExtractor.get_vars_info()
            objective_terms (List[Tuple[float, Dict[str, int]]]): List of tuples representing the objective function terms obtained by the first output of ./utils_loaders.ProbModelExtractor.get_objective_info().

        Returns:
            np.array: Array of variable features.
        """
        # get basic features from variables: type, lower bound, upper bound
        var_types, var_lbs, var_ubs = [], [], []
        for var_info in variable_info_list.values():
            var_types.append(var_info["type"])
            var_lbs.append(var_info["lb"])
            var_ubs.append(var_info["ub"])
        var_types = np.array(
            [self.onehot_encoding["VTYPE"][vtype] for vtype in var_types]
        ).T
        var_lbs = np.array(var_lbs)
        var_ubs = np.array(var_ubs)
        var_lbs_inf = np.isinf(var_lbs).astype(int)
        var_ubs_inf = np.isinf(var_ubs).astype(int)
        # obtain variable features from the objective function
        var_weight_dict = {var_name: [[], []] for var_name in variable_info_list.keys()}
        for term_info in objective_terms:
            coeff, var_deg_info = term_info
            for var_name, degree in var_deg_info.items():
                if var_name in var_weight_dict:
                    var_weight_dict[var_name][0].append(coeff)
                    var_weight_dict[var_name][1].append(degree)
        var_coeff = np.array(
            [
                np.mean(var_weight[0]) if len(var_weight[0]) > 0 else 0.0
                for var_weight in var_weight_dict.values()
            ]
        )
        var_degree = np.array(
            [
                np.mean(weight[1]) if len(weight[1]) > 0 else 0.0
                for weight in var_weight_dict.values()
            ]
        )
        # stack all features together
        var_features = np.vstack(
            (
                var_types,
                var_lbs,
                var_ubs,
                var_lbs_inf,
                var_ubs_inf,
                var_coeff,
                var_degree,
            )
        ).T
        return var_features

    def get_constraint_features(self, rhs_list, sense_list) -> np.array:
        """
        Extract features from constraints for hypergraph representation.

        Parameters:
            rhs_list (List[float]): List of right-hand side values of the constraints, obtained from the second output of ./utils_loaders.ProbModelExtractor.get_constraints_info().
            sense_list (List[str]): List of constraint senses, obtained from the third output of ./utils_loaders.ProbModelExtractor.get_constraints_info().

        Returns:
            np.array: Array of constraint features.
        """
        # get basic features from constraints: right-hand side, sense
        rhs = np.array(rhs_list)
        sense = np.array(
            [self.onehot_encoding["CTYPE"][sense] for sense in sense_list]
        ).T
        # stack all features together
        constr_features = np.vstack((sense, rhs)).T
        return constr_features

    def get_hyperedge_features(
        self, constraint_term_list, objective_terms, variable_info_list
    ) -> tuple[list[tuple[int, ...]], np.array, np.array]:
        """
        Extract features from hyperedges for hypergraph representation.

        Parameters:
            constraint_term_list (List[List[Tuple[float, Dict[str, int]]]]): List of lists containing tuples representing the terms of each constraint.
            objective_terms (List[Tuple[float, Dict[str, int]]]): List of tuples representing the objective function terms.
            variable_info_list (Dict[str, Dict[str, Any]]): Dictionary containing variable information.

        Returns:
            List[Tuple[int, ...]]: list of hyperedges. Each element is a tuple of variable indices, representing the variables involved in the hyperedge.
            v2e_weights: List[float]: list of weights for each variable-hyperedge pair.
            e2v_weights: List[float]: list of weights for each hyperedge-variable pair.
        """
        hyperedges, weight_info = [], {}
        # transverse high-degree terms in constraints and objective
        for expr_term in constraint_term_list + [objective_terms]:
            for coeff, var_deg_info in expr_term:
                # get the indices of variables involved in the hyperedge
                if len(var_deg_info) == 1 and list(var_deg_info.values())[0] == 1:
                    # if the term is linear, skip it
                    continue
                vars, orders = [], []
                for var_name, order in var_deg_info.items():
                    var_index = list(variable_info_list.keys()).index(var_name)
                    vars.append(var_index)
                    orders.append(order)
                pairs = list(zip(vars, orders))
                sorted_pairs = sorted(pairs, key=lambda x: x[0])
                sorted_vars, sorted_orders = zip(*sorted_pairs)
                hyperedge_tuple = (tuple(sorted_vars), tuple(sorted_orders))
                if hyperedge_tuple in weight_info:
                    weight_info[hyperedge_tuple]["coeff"] += coeff
                else:
                    hyperedges.append(hyperedge_tuple)
                    var_order = {
                        var_index: order
                        for var_index, order in zip(sorted_vars, sorted_orders)
                    }
                    weight_info[hyperedge_tuple] = {
                        "coeff": coeff,
                        "var_order": var_order,
                    }
        # build weights
        (
            rows1,
            cols1,
            data1,
            rows2,
            cols2,
            data2,
            rows3,
            cols3,
            data3,
            rows4,
            cols4,
            data4,
        ) = (
            [] for _ in range(12)
        )  # for v2e_matrix1, v2e_matrix2, e2v_matrix1, e2v_matrix2 separately
        for hyp_idx, hyperedge_tuple in enumerate(hyperedges):
            var_order, coeff = (
                weight_info[hyperedge_tuple]["var_order"],
                weight_info[hyperedge_tuple]["coeff"],
            )
            for var_idx, order in var_order.items():
                rows1.append(var_idx)
                cols1.append(hyp_idx)
                data1.append(order)
                rows2.append(var_idx)
                cols2.append(hyp_idx)
                data2.append(coeff)
                rows3.append(hyp_idx)
                cols3.append(var_idx)
                data3.append(order)
                rows4.append(hyp_idx)
                cols4.append(var_idx)
                data4.append(coeff)
        # Create sparse matrices
        v2e_matrix1 = coo_matrix(
            (data1, (rows1, cols1)), shape=(len(variable_info_list), len(hyperedges))
        )
        v2e_matrix2 = coo_matrix(
            (data2, (rows2, cols2)), shape=(len(variable_info_list), len(hyperedges))
        )
        e2v_matrix1 = coo_matrix(
            (data3, (rows3, cols3)), shape=(len(hyperedges), len(variable_info_list))
        )
        e2v_matrix2 = coo_matrix(
            (data4, (rows4, cols4)), shape=(len(hyperedges), len(variable_info_list))
        )
        # Extract non-zero values (these will be in matrix order)
        v2e_weights1 = v2e_matrix1.data
        v2e_weights2 = v2e_matrix2.data
        e2v_weights1 = e2v_matrix1.data
        e2v_weights2 = e2v_matrix2.data
        v2e_weights = np.vstack((v2e_weights1, v2e_weights2)).T
        e2v_weights = np.vstack((e2v_weights1, e2v_weights2)).T
        hyperedges_var_only = [hyperedge_tuple[0] for hyperedge_tuple in hyperedges]
        return hyperedges_var_only, v2e_weights, e2v_weights

    def get_pairwise_edges(
        self, variable_info_list, constraint_term_list
    ) -> tuple[list[tuple[int, int]], np.array]:
        """
        Extract pairwise edges and their features from variable and constraint information for hypergraph representation.

        Parameters:
            variable_info_list (Dict[str, Dict[str, Any]]): Dictionary containing variable information obtained from ./utils_loaders.ProbModelExtractor.get_vars_info().
            constraint_term_list (List[List[Tuple[float, Dict[str, int]]]]): List of lists containing tuples representing the terms of each constraint, obtained from the first output of ./utils_loaders.ProbModelExtractor.get_constraints_info().

        Returns:
            List[Tuple[int, int]]: List of edges, where each edge is represented by a list of vertex indices.
            List[float]: List of edge features.
        """
        weight_info = {}
        for constr_idx, single_constraint_term in enumerate(constraint_term_list):
            for coeff, var_info in single_constraint_term:
                for var_name, order in var_info.items():
                    var_index = list(variable_info_list.keys()).index(var_name)
                    pairwise_edge = (var_index, constr_idx)
                    if pairwise_edge in weight_info:
                        weight_info[pairwise_edge].append(
                            {"coeff": coeff, "order": order}
                        )
                    else:
                        weight_info[pairwise_edge] = [{"coeff": coeff, "order": order}]
        # Convert dictionary to arrays
        pairwise_edges = []
        pairwise_weights = []
        for edge_key, edge_weight_info in sorted(
            weight_info.items(), key=lambda x: x[0][0]
        ):
            pairwise_edges.append(edge_key)
            mean_coeff = np.mean([w["coeff"] for w in edge_weight_info])
            mean_order = np.mean([w["order"] for w in edge_weight_info])
            pairwise_weights.append([mean_coeff, mean_order])

        return np.array(pairwise_edges), np.array(pairwise_weights)

    def get_repr(self, instance_info, prob_model_name, output_path):
        if self.check_repr_exists(prob_model_name, output_path):
            representation = pickle.load(
                open(os.path.join(output_path, f"{prob_model_name}.pkl"), "rb")
            )[1]
            return representation
        # unpack instance info
        variable_info = instance_info["variable_info"]
        objective_info = instance_info["objective_info"]
        constraint_info = instance_info["constraint_info"]
        solution_values = instance_info["solution_values"]

        objective_terms, objective_sense = objective_info
        constraint_term_list, rhs_list, sense_list = (
            constraint_info[0],
            constraint_info[1],
            constraint_info[2],
        )
        other_constr_info = None
        if len(constraint_info) > 3:
            other_constr_info = constraint_info[3:]

        # all_var_names = list(variable_info.keys())
        # num_vars = len(variable_info)
        # num_constraints = len(rhs_list)

        var_features = self.get_variable_features(variable_info, objective_terms)
        constr_features = self.get_constraint_features(rhs_list, sense_list)
        hyperedges, v2e_weights, e2v_weights = self.get_hyperedge_features(
            constraint_term_list, objective_terms, variable_info
        )
        pairwise_edges, pairwise_weights = self.get_pairwise_edges(
            variable_info, constraint_term_list
        )
        return {
            "variable_features": var_features,
            "constraint_features": constr_features,
            "hyperedges": hyperedges,
            "v2e_weights": v2e_weights,
            "e2v_weights": e2v_weights,
            "pairwise_edges": pairwise_edges,
            "pairwise_weights": pairwise_weights,
            "solution_values": solution_values,
            "other_constr_info": other_constr_info,
        }
    

def generate_representation(
    file_idx,
    total_files,
    file,
    instance_info_dir,
    repr_dir,
    print=False,
):
    """
    Worker function for parallel representation generation.

    Args:
        file_idx (int): Index of the current file being processed.
        total_files (int): Total number of files to process.
        file (str): Filename of the current instance info file.
        instance_info_dir (str): Directory containing instance info pickle files.
        repr_dir (str): Directory to save generated representations.
        print (bool): Whether to print progress messages.
    """
    instance_info_path = os.path.join(instance_info_dir, file)
    with open(instance_info_path, "rb") as f:
        instance_info = pickle.load(f)
    prob_model_name = file.rsplit(".", 1)[0]

    repr_generator = ReprOurs(None)
    representation = repr_generator.get_repr(instance_info, prob_model_name, repr_dir)
    repr_generator.save_repr(representation, prob_model_name, repr_dir)
    if print:
        print(
            f"{file_idx + 1}/{total_files}: Saved representation for {prob_model_name} to {repr_dir}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Generate representations for optimization instances."
    )
    parser.add_argument(
        "--instance_info_dir",
        type=str,
        required=True,
        help="Directory containing instance info pickle files.",
    )
    parser.add_argument(
        "--repr_dir",
        type=str,
        required=True,
        help="Directory to save generated representations.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=16,
        help="Number of parallel workers for representation generation.",
    )
    args = parser.parse_args()

    # unpack arguments
    instance_info_dir = args.instance_info_dir
    repr_dir = args.repr_dir
    num_workers = args.num_workers

    os.makedirs(repr_dir, exist_ok=True)
    instance_info_files = [
        f for f in os.listdir(instance_info_dir) if f.endswith(".pkl")
    ]
    total_files = len(instance_info_files)
    tasks = [
        (file_idx, total_files, file, instance_info_dir, repr_dir)
        for file_idx, file in enumerate(instance_info_files)
    ]
    with ProcessPoolExecutor(max_workers=16) as executor:
        for result in executor.map(generate_representation, *zip(*tasks)):
            print(result)