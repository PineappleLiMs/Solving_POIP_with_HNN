# Supplementary Material for *Towards Solving Polynomial-Objective Integer Programming with Hypergraph Neural Networks*

This repository contains the Appendix and the code for the paper ``Towards Solving Polynomial-Objective Integer Programming with Hypergraph Neural Networks'', accepted by the 23rd International Conference on the Integration of Constraint Programming, Artificial Intelligence, and Operations Research (CPAIOR 2026).

## Environment Setup
1. Recommend using Conda with Python 3.10
```bash
conda create -n [env] python=3.10
conda activate [env]
```
2. Install Gurobi and gurobipy, and install pyscipopt (python API for SCIP) and amplpy (python API for AMPL).
3. install dependencies:
    1. Install [`dhg`](https://deephypergraph.readthedocs.io/en/latest/start/install.html) using `pip install dhg`. This might help you install the required `torch` dependencies.
    2. Install [`torch_geometric`](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) by `pip install torch_geometric`
    3. Install dependencies of `torch_geometric` based on your CUDA version.
    ```bash
    pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
    ```
    4. Install other missing dependencies using pip

## Steps for Reproducing Results

0. data: can be found at <https://drive.google.com/file/d/1h6Sf1XXc_ODAKk0IiDhpGKgddXXCh4oG/view?usp=drive_link>. Download the place it the at root of the project. The data is organized as follows:
    * ./data/{train or test}/problem/{problem_name}/{problem_size}/: instances for training/testing
    * ./data/{train or test}/instance_info/{problem_name}/{problem_size}/: extracted features that are used to build representations
    * ./data/{train or test}/representation/{representation_name}/{problem_name}/{problem_size}/: representation that models can directly handle.
    * Remarks: problem_name can be "qkp", "qis", "pcflp", corresponding to the QMKP, RandQCP and CFLPTC problem type in the paper. For qkp and qis, the original problems in .lp format are provided; for pcflp, the instance information in .pkl format are provided.

1. get representation
    ```bash
    # extract information
    python instance_info_get.py --instance_dir ./data/train/problem/{problem_name}/{problem_size} --info_dir ./data/train/instance_info/{problem_name}/{problem_size} --model_type GurobiQP --num_workers {number of CPU cores to use}
    # generate representation
    python repr_get.py --instance_info_dir ./data/train/instance_info/{problem_name}/{problem_size} --repr_dir ./data/train/representation/Ours/{problem_name}/{problem_size} --num_workers {number of CPU cores to use}
    ```

2. train
    ```bash
    python model_HNN.py -p {problem_name} -d {problem_size} --nepoch {number of epochs} --batch_size {batch size} --save_epoch {list of epochs that you wish to save the model}
    ```
    The trained models are saved at "./runs/train/Ours/{problem_name}/{problem_size}"

3. test
    * first generate representations for testing instances. This can be done by changing "./data/train/problem/" in the commands of Step 1 to "./data/test/problem/"
    * then run
    ```bash
    python main_full_alg.py general.training_problem={problem_name}/{training_problem_size} general.testing_problem={problem_name}/{testing_problem_size} general.model_name=Ours lns.neuralqp.initial_time_limit=30 lns.neuralqp.search_time_limit=30 lns.neuralqp.cross_time_limit=30 run.num_instances=10 run.time_limit={time limit for one run} run.solver={scip or gurobi}
    ```
    results will be saved at "./runs/main/{problem_name}/{problem_size}"