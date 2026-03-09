"""
Implement the main full algorithm and run it on the dataset.
"""

import numpy as np
import lns_Neuralqp
import hydra
from omegaconf import DictConfig, OmegaConf
import os
import utils_io
import torch
import random
import pickle
import pandas as pd
from model_HNN import HybridGraphModel, hg_data, bg_data


def lns_solve(
    lns_method: str,
    problem_info: dict,
    output_prob: np.ndarray,  # neural network output.
    hard_predict: np.ndarray,  # hard predict
    confidence: np.ndarray,  # confidence score
    time_limit: float,
    solver: str,
    logfile_path: str = None,
    print_output: bool = True,
    **lns_kwargs,
):
    """
    Solve the problem using LNS with neural network predictions.

    Parameters:
    ----------
    lns_method : str
        The LNS method to use.
    problem_info : dict
        The problem information obtained from instance_info_get.py.
    output_prob : np.ndarray
        The neural network output probabilities.
    hard_predict : np.ndarray
        The hard predictions derived from the neural network output.
    confidence : np.ndarray
        The confidence scores for the hard predictions.
    time_limit : float
        The time limit for the LNS solver.
    solver : str
        The underlying solver to use.
    logfile_path : str, optional
        The path to the logfile.
    print_output : bool, optional
        Whether to print output to console.
    lns_kwargs : dict
        Additional keyword arguments for the LNS method.
    """
    if lns_method in [
        a + b for a in ["neuralqp", "NeuralQP", "Neuralqp"] for b in ["", "_lns"]
    ]:
        all_var = list(problem_info["variable_info"].keys())
        solution, objective, _ = lns_Neuralqp.optimize(
            logfile_path=logfile_path,
            solver=solver,
            problem_data=[problem_info, all_var, hard_predict, None],
            output_prob=output_prob,
            hard_predict=hard_predict,
            confidence=confidence,
            time_limit=time_limit,
            print_output=print_output,
            **lns_kwargs,
        )
    else:
        raise ValueError(f"Unsupported LNS method: {lns_method}")
    return solution, objective


def get_hard_prediction_and_confidence(predict_logits, binary_indexes):
    """
    Get the hard prediction from the model's output logits.

    Note that this function only works for mixed-binary variables. The hard prediction for a binary variable is 0 or 1, while for continuous variables it is the predicted logits.

    Parameters
    ----------
    predict_logits : np.array
        The output logits from the neural network model.
    binary_indexes : np.array
        The indexes of the binary variables in the prediction.
    """
    hard_prediction = predict_logits.copy()
    confidence = np.full(
        predict_logits.shape, -np.inf, dtype=float
    )  # initialize confidence scores. Set to -inf for non-binary variables, so that they can never be fixed in LNS.
    for idx in binary_indexes:
        hard_prediction[idx] = 0 if predict_logits[idx] < 0.5 else 1
        confidence[idx] = 1 - abs(predict_logits[idx] - hard_prediction[idx])
    return hard_prediction, confidence


@hydra.main(config_path="./configs", config_name="main_full_alg", version_base=None)
def main_full_alg(cfg: DictConfig):
    # unpack general configs: training_problem, testing_problem, seeds, device, etc.
    cfg_g = cfg.general
    training_problem: str = cfg_g.training_problem
    testing_problem: str = cfg_g.testing_problem
    seed: int = cfg_g.seed
    model_name: str = cfg_g.model_name
    device: str = cfg_g.device
    device = torch.device(device)
    nnmodel = HybridGraphModel(cfg_g.model)

    # unpack lns configs:
    cfg_lns = cfg.lns
    lns_method: str = cfg_lns.method
    lns_kwargs = OmegaConf.to_container(cfg_lns[lns_method], resolve=True)

    # unpack run configs:
    cfg_run = cfg.run
    time_limit: int = cfg_run.time_limit
    num_instances: int = cfg_run.num_instances
    repeats: int = cfg_run.repeats
    start_idx: int = cfg_run.start_idx
    solver: str = cfg_run.solver

    # create output directory
    training_name_str = model_name + "_" + training_problem.replace("/", "_")
    output_dir = f"./runs/main/{testing_problem}/{training_name_str}/{lns_method}_{solver}/{time_limit}/"
    os.makedirs(output_dir, exist_ok=True)

    # create logs
    logfile = os.path.join(output_dir, "main_logs.txt")
    if os.path.exists(logfile):
        os.remove(logfile)
    # record configurations
    utils_io.log("Configurations:", logfile)
    utils_io.log(f"{OmegaConf.to_container(cfg, resolve=True)}", logfile)

    # set random seed
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # load the model``
    if cfg_g.model.epochs == 0:
        model_path = f"./runs/train/{model_name}/" + training_problem.replace("/", "_")
    nnmodel.load_state_dict(torch.load(model_path, map_location=device))
    nnmodel.to(device)
    nnmodel.eval()

    testing_instance_info_dir = f"./data/test/instance_info/{testing_problem}/"
    testing_instance_repr_dir = f"./data/test/representation/Ours/{testing_problem}/"

    info_list = [
        info for info in os.listdir(testing_instance_info_dir) if info.endswith(".pkl")
    ]
    info_list = sorted(info_list, key=lambda x: int(x.split("_")[-1].split(".")[0]))
    if start_idx >= 0 and start_idx < len(info_list):
        info_list = info_list[start_idx:]
    if num_instances > 0 and num_instances < len(info_list):
        info_list = info_list[:num_instances]
    utils_io.log(
        f"Total {len(info_list)} testing instances to run on.",
        logfile,
    )

    # run on each testing instance
    results = []  # store results for all instances
    for idx, info_file in enumerate(info_list):
        utils_io.log(
            f"Running on instance {idx + 1}/{len(info_list)}: {info_file}",
            logfile,
            print_output=True,
        )
        instance_info_path = os.path.join(testing_instance_info_dir, info_file)
        instance_repr_path = os.path.join(testing_instance_repr_dir, info_file)
        # load instance info
        with open(instance_repr_path, "rb") as f:
            _, repr_data = pickle.load(f)
        hypergraph = hg_data(
            repr_data["variable_features"],
            repr_data["hyperedges"],
            repr_data["solution_values"],
            vertices_type="var",
        ).to(device)
        bipartite = bg_data(
            repr_data["variable_features"],
            repr_data["constraint_features"],
            repr_data["pairwise_weights"],
            repr_data["pairwise_edges"],
            repr_data["solution_values"],
        ).to(device)
        v2e_weight = torch.FloatTensor(repr_data["v2e_weights"]).to(device)
        e2v_weight = torch.FloatTensor(repr_data["e2v_weights"]).to(device)
        # get model prediction
        with torch.no_grad():
            output = nnmodel(hypergraph, bipartite, v2e_weight, e2v_weight).sigmoid()
        output = output[: len(hypergraph.opt_sol)].cpu().detach().numpy()

        # load instance info
        with open(instance_info_path, "rb") as f:
            instance_info = pickle.load(f)
        binary_indexes = []
        for var_idx, var_info in enumerate(instance_info["variable_info"].values()):
            if var_info["type"] == "B":
                binary_indexes.append(int(var_idx))
        hard_prediction, confidence = get_hard_prediction_and_confidence(
            output, binary_indexes
        )

        # repeat runs
        instance_result = []
        results_to_collect = []
        instance_name_base = info_file.replace(".pkl", "")
        for repeat_idx in range(repeats):
            utils_io.log(
                f"  Repeat {repeat_idx + 1}/{repeats}", logfile, print_output=False
            )
            solution, objective = lns_solve(
                lns_method,
                instance_info,
                output,
                hard_prediction,
                confidence,
                time_limit,
                solver,
                logfile,
                print_output=False,
                **lns_kwargs,
            )
            utils_io.log(
                f"    {instance_name_base} {repeat_idx + 1}-th run objective: {objective}",
                logfile,
            )
            result = {
                "instance_name": instance_name_base,
                "run": repeat_idx,
                "lns_method": lns_method,
                "solver": solver,
                "time": time_limit,
                "obj": objective,
                "solution": solution,
            }
            results_to_collect.append(result)
            instance_result.append({**result, "solution": solution})
        # save instance results
        instance_output_path = os.path.join(
            output_dir, f"{instance_name_base}_results.pkl"
        )
        pickle.dump(instance_result, open(instance_output_path, "wb"))
        # collect results
        results.extend(results_to_collect)

    # save all results to a dataframe
    results_df = pd.DataFrame(
        results,
        columns=[
            "instance_name",
            "repeat_idx",
            "lns_method",
            "solver",
            "time_limit",
            "objective_value",
        ],
    )
    results_csv_path = os.path.join(
        output_dir, f"results{start_idx}-{start_idx + len(info_list) - 1}.csv"
    )
    results_df.to_csv(results_csv_path, index=False)


if __name__ == "__main__":
    main_full_alg()
