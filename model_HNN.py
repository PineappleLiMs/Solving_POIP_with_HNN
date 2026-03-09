# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch_geometric
import dhg
import pickle
import argparse
import numpy as np
from tqdm import tqdm
import logging
import pprint
import os
from sklearn.metrics import f1_score


# global variables
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_epoch_list = [1, 10, 50, 100]  # number of epochs to visualize


# %%
# define dataset classes
class hg_data(dhg.Hypergraph):
    """
    read instance-feature file to get instance-specific variable-monomial hypergraph (in the format of dhg.Hypergraph)
    """

    def __init__(
        self,
        var_features: np.ndarray,
        hyperedges: np.ndarray,
        sol: np.ndarray,
        con_features: np.ndarray = None,
        vertices_type: str = "var",
    ):
        self.var_features = torch.FloatTensor(var_features)
        self.opt_sol = torch.FloatTensor(sol)
        if con_features is not None:
            self.con_features = torch.FloatTensor(con_features)
        self.vertices_type = vertices_type
        if vertices_type == "var":
            super().__init__(num_v=self.var_features.shape[0], e_list=hyperedges)
        elif vertices_type == "var_con":
            if con_features is None:
                raise ValueError(
                    "con_features must be provided when vertices_type is 'var_con'"
                )
            vertice_num = self.var_features.shape[0] + self.con_features.shape[0]
            super().__init__(num_v=vertice_num, e_list=hyperedges)

    def to(self, device: torch.cuda.device):
        super().to(device)
        self.var_features = self.var_features.to(device)
        self.opt_sol = self.opt_sol.to(device)
        if hasattr(self, "con_features"):
            self.con_features = self.con_features.to(device)
        return self


class bg_data(dhg.BiGraph):
    """
    read instance-feature file to get instance-specific variable-constraint bipartite graph (in the format of dhg.BiGraph)
    """

    def __init__(
        self,
        var_features: np.ndarray,
        con_features: np.ndarray,
        edge_features: np.ndarray,
        edges: np.ndarray,
        sol: np.ndarray,
    ):
        self.edges = torch.LongTensor(edges).T
        self.var_features = torch.FloatTensor(var_features)
        self.con_features = torch.FloatTensor(con_features)
        self.edge_features = torch.FloatTensor(edge_features)
        self.opt_sol = torch.FloatTensor(sol)
        super().__init__(
            num_u=self.var_features.shape[0],
            num_v=self.con_features.shape[0],
            e_list=edges,
        )

    def to(self, device: torch.cuda.device):
        super().to(device)
        self.edges = self.edges.to(device)
        self.var_features = self.var_features.to(device)
        self.con_features = self.con_features.to(device)
        self.edge_features = self.edge_features.to(device)
        self.opt_sol = self.opt_sol.to(device)
        return self


class graph_dataset(torch.utils.data.Dataset):
    """
    customed dataset class for graph data
    """

    def __init__(self, sample_files):
        super().__init__()
        self.sample_files = sample_files

    def __len__(self):
        return len(self.sample_files)

    def __getitem__(self, idx):
        name, graph_features = pickle.load(open(self.sample_files[idx], "rb"))
        hypergraph = hg_data(
            graph_features["variable_features"],
            graph_features["hyperedges"],
            graph_features["solution_values"],
            vertices_type="var",
        )
        bipartite = bg_data(
            graph_features["variable_features"],
            graph_features["constraint_features"],
            graph_features["pairwise_weights"],
            graph_features["pairwise_edges"],
            graph_features["solution_values"],
        )
        v2e_weight = torch.FloatTensor(graph_features["v2e_weights"])
        e2v_weight = torch.FloatTensor(graph_features["e2v_weights"])
        return name, hypergraph, bipartite, v2e_weight, e2v_weight


all_datasets = {
    "combined": graph_dataset,
    "graphs": graph_dataset,
    "hybrid": graph_dataset,
}


# %%
all_activations = {
    "relu": nn.ReLU(),
    "prelu": nn.PReLU(),
    "elu": nn.ELU(),
    "leakyrelu": nn.LeakyReLU(negative_slope=0.1),  # hyperparameters
    "sigmoid": nn.Sigmoid(),
    "tanh": nn.Tanh(),
    "softmax": nn.Softmax(),
    "identity": nn.Identity(),
}

# =========================== Model =========================== #


### classes for hypergraph covolution
class UniGNN(nn.Module):
    """
    hyperedge convolution to realize:
    1. message passing from variable vertices (variables) to hyperedges (monomials)
    2. message passing from hyperedges (monomials) to variable vertices (variables)
    """

    def __init__(
        self,
        args: argparse.Namespace,  # configurations
        in_channels: int,  # number of input channels
        out_channels: int,  # number of output channels
        bias: bool,  # whether to use bias
        drop_rate: float,  # dropout rate, between 0 and 1
        activation: str,  # activation function
    ):
        super().__init__()
        self.args = args
        self.v_channels = in_channels
        self.e_channels = out_channels
        self.act = all_activations[activation]
        self.drop = nn.Dropout(drop_rate)

        # Initialize linear layers with Xavier initialization
        self.theta_vertex = nn.Linear(self.v_channels, self.e_channels, bias=bias)
        self.theta_edge = nn.Linear(self.e_channels, self.v_channels, bias=bias)
        self.edge_merge = nn.Linear(self.e_channels, self.e_channels, bias=bias)
        self.vertex_merge = nn.Linear(self.v_channels * 2, self.v_channels, bias=bias)
        self.reset_parameters()

        self.first_aggregate = args.hg_first_aggregate
        self.second_aggregate = args.hg_second_aggregate

        # Added layer normalization
        self.layer_norm_v_pre = nn.LayerNorm(self.v_channels)
        self.layer_norm_v_post = nn.LayerNorm(self.v_channels)
        self.layer_norm_e = nn.LayerNorm(self.e_channels)

    def reset_parameters(self):
        # Added proper weight initialization
        nn.init.xavier_uniform_(self.theta_vertex.weight)
        nn.init.xavier_uniform_(self.theta_edge.weight)
        nn.init.xavier_uniform_(self.edge_merge.weight)
        nn.init.xavier_uniform_(self.vertex_merge.weight)

    def forward(
        self,
        X: torch.Tensor,  # embeddings of variable vertices
        graph: dhg.Hypergraph,  # hypergraph representation of vairable-monomial
        v2e_weight: torch.Tensor = None,  # weight of variable-to-monomial edges
        e2v_weight: torch.Tensor = None,  # weight of monomial-to-variable edges
    ):
        """
        message passing to aggregate monomial information to variable embeddings
        """
        X_0 = X
        X = self.layer_norm_v_pre(X)
        X = self.theta_vertex(X)
        Y = self.edge_merge(
            graph.v2e_aggregation(X, aggr=self.first_aggregate, v2e_weight=v2e_weight)
        )
        Y = self.layer_norm_e(Y)

        Y = self.theta_edge(Y)
        X = self.vertex_merge(
            torch.cat(
                [
                    X_0,
                    graph.e2v_aggregation(
                        Y, aggr=self.second_aggregate, e2v_weight=e2v_weight
                    ),
                ],
                dim=-1,
            )
        )

        self.act(X)
        X = self.layer_norm_v_post(X)
        X = self.drop(X)
        # Y = self.drop(self.act(Y))
        # Y = self.layer_norm_e(Y)
        # return X, Y
        return X + X_0  # Residual connection


### classes for bipartite graph covolution
class BiGCNN(nn.Module):
    """
    bipartite graph convolution to realize:
    1. message passing from variable vertices to constraint vertices
    2. message passing from constraint vertices to variable vertices
    """

    def __init__(
        self,
        emb_size: int,  # number of input channels
        activation: str,  # activation function
    ):
        super().__init__()
        self.emb_size = emb_size
        self.activation = all_activations[activation]

        # Added layer normalization for all components
        self.norm_var = nn.LayerNorm(emb_size)
        self.norm_edge = nn.LayerNorm(emb_size)
        self.norm_con = nn.LayerNorm(emb_size)
        self.norm_joint = nn.LayerNorm(emb_size)

        # tranform embeddings of vertices and edges
        self.transform_left = nn.Linear(emb_size, emb_size)
        self.transform_edge = nn.Linear(emb_size, emb_size, bias=False)
        self.transform_right = nn.Linear(emb_size, emb_size, bias=False)
        self.transform_join = nn.Sequential(
            self.activation, nn.LayerNorm(emb_size), nn.Linear(emb_size, emb_size)
        )

        # message passing
        self.merge = nn.Sequential(
            nn.Linear(emb_size * 2, emb_size), self.activation, nn.LayerNorm(emb_size)
        )

        # initialize weights
        self.reset_parameters()

    def reset_parameters(self):
        # Proper weight initialization
        nn.init.xavier_uniform_(self.transform_left.weight)
        nn.init.xavier_uniform_(self.transform_edge.weight)
        nn.init.xavier_uniform_(self.transform_right.weight)

    def directional_message_passing(
        self,
        variable_emb: torch.Tensor,  # embeddings of variable vertices
        edge_emb: torch.Tensor,  # embeddings of edges from variable-constraint bipartite graph
        constraint_emb: torch.Tensor,  # embeddings of constraint vertices
        graph: dhg.BiGraph,  # bipartite graph representation of variable-constraint
        var2con: bool = True,  # whether to pass message from variable to constraint
        device: str = None,  # device to run the model
    ):
        ### transform embeddings
        variable_emb = self.norm_var(variable_emb)
        edge_emb = self.norm_edge(edge_emb)
        constraint_emb = self.norm_con(constraint_emb)
        edge_emb = self.transform_edge(edge_emb)
        variable_emb = self.transform_left(variable_emb)
        constraint_emb = self.transform_right(constraint_emb)

        ### Build adjacency matrices for left-to-edge and edge-to-right relationships
        var2edge = torch.sparse_coo_tensor(
            indices=torch.stack([graph.e_u, torch.arange(graph.num_e, device=device)]),
            values=torch.ones(graph.num_e, device=device),
            size=(graph.num_u, graph.num_e),
        )
        con2edge = torch.sparse_coo_tensor(
            indices=torch.stack([graph.e_v, torch.arange(graph.num_e, device=device)]),
            values=torch.ones(graph.num_e, device=device),
            size=(graph.num_v, graph.num_e),
        )
        ### compute joint embeddings
        var_agg = torch.sparse.mm(var2edge.t(), variable_emb)
        con_agg = torch.sparse.mm(con2edge.t(), constraint_emb)
        joint_emb = self.transform_join(var_agg + edge_emb + con_agg)
        joint_emb = self.norm_joint(joint_emb)
        ### merge and update embeddings
        if var2con:
            con_agg = torch.sparse.mm(con2edge, joint_emb)
            out = self.merge(torch.cat([constraint_emb, con_agg], dim=-1))
            return constraint_emb + out
        else:
            var_agg = torch.sparse.mm(var2edge, joint_emb)
            out = self.merge(torch.cat([variable_emb, var_agg], dim=-1))
            return variable_emb + out

    def forward(
        self,
        variable_emb: torch.Tensor,  # embeddings of variable vertices
        edge_emb: torch.Tensor,  # embeddings of edges from variable-constraint bipartite graph
        constraint_emb: torch.Tensor,  # embeddings of constraint vertices
        graph: dhg.BiGraph,  # bipartite graph representation of variable-constraint,
        device: str = None,  # device to run the model
    ):
        constraint_emb = self.directional_message_passing(
            variable_emb, edge_emb, constraint_emb, graph, var2con=True, device=device
        )
        variable_emb = self.directional_message_passing(
            variable_emb, edge_emb, constraint_emb, graph, var2con=False, device=device
        )
        return variable_emb, constraint_emb


class HybridGraphModel(nn.Module):
    """
    complete model for hypergraph and bipartite graph convolution
    """

    def __init__(
        self,
        args: argparse.Namespace,  # configurations
    ):
        super().__init__()
        self.args = args
        # set the number of convulution
        self.hyperconv = nn.ModuleList(
            [
                UniGNN(
                    args,
                    args.nhid,
                    args.nhid,
                    args.bias,
                    args.drop_rate,
                    args.hg_activation,
                )
                for _ in range(args.num_hyperconv)
            ]
        )
        self.bigconv = nn.ModuleList(
            [BiGCNN(args.nhid, args.bg_activation) for _ in range(args.num_bigconv)]
        )
        # compute raw embeddings from raw features
        self.ini_act = all_activations[args.ini_activation]
        self.var_emb = nn.Sequential(
            nn.LayerNorm(args.num_var_features),
            nn.Linear(args.num_var_features, args.nhid, bias=args.bias),
            self.ini_act,
            nn.LayerNorm(args.nhid),
            nn.Dropout(args.drop_rate),
            nn.Linear(args.nhid, args.nhid, bias=args.bias),
        )
        self.edge_emb = nn.Sequential(
            nn.LayerNorm(args.num_edge_features),
            nn.Linear(args.num_edge_features, args.nhid, bias=args.bias),
            self.ini_act,
            nn.LayerNorm(args.nhid),
            nn.Dropout(args.drop_rate),
            nn.Linear(args.nhid, args.nhid, bias=args.bias),
        )
        self.con_emb = nn.Sequential(
            nn.LayerNorm(args.num_con_features),
            nn.Linear(args.num_con_features, args.nhid, bias=args.bias),
            self.ini_act,
            nn.LayerNorm(args.nhid),
            nn.Dropout(args.drop_rate),
            nn.Linear(args.nhid, args.nhid, bias=args.bias),
        )
        self.v2hyperedge_weight_mlp = nn.Sequential(
            nn.LayerNorm(2),
            nn.Linear(2, args.nhid // 2, bias=args.bias),
            self.ini_act,
            nn.LayerNorm(args.nhid // 2),
            nn.Dropout(args.drop_rate),
            nn.Linear(args.nhid // 2, 1, bias=args.bias),
        )
        # define activation function for final output
        self.final_act = all_activations[args.final_activation]
        # output final variable embeddings
        self.var_out = nn.Sequential(
            nn.LayerNorm(args.nhid),
            nn.Linear(args.nhid, args.nhid, bias=args.bias),
            self.final_act,
            nn.LayerNorm(args.nhid),
            nn.Dropout(args.drop_rate),
            nn.Linear(args.nhid, args.nout, bias=args.bias),
        )

    def forward(
        self,
        HyG: hg_data,  # hypergraph representation of variable-monomial
        BiG: bg_data,  # bipartite graph representation of variable-constraint
        v2e_weight: torch.Tensor,  # weight of variable-to-monomial edges
        e2v_weight: torch.Tensor,  # weight of monomial-to-variable edges
    ):
        # initialize embeddings
        var_emb = self.var_emb(HyG.var_features)
        edge_emb = self.edge_emb(BiG.edge_features)
        con_emb = self.con_emb(BiG.con_features)
        v2e_weight = self.v2hyperedge_weight_mlp(v2e_weight).squeeze(-1)
        e2v_weight = self.v2hyperedge_weight_mlp(e2v_weight).squeeze(-1)

        var_emb_init = var_emb.clone()

        # hypergraph convolution
        for _, hyperconv in enumerate(self.hyperconv):
            var_emb = hyperconv(var_emb, HyG, v2e_weight, e2v_weight)
        # bipartite graph convolution
        for _, bigconv in enumerate(self.bigconv):
            var_emb, con_emb = bigconv(
                var_emb, edge_emb, con_emb, BiG, device=BiG.device
            )
        # output final variable embeddings
        var_emb = var_emb + var_emb_init
        var_emb = self.final_act(var_emb)
        var_emb = self.var_out(var_emb)
        return var_emb.squeeze()


all_models = {
    "HybridGraphModel": HybridGraphModel,
    "Ours": HybridGraphModel,
}


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = (self.alpha * (1 - pt) ** self.gamma * ce_loss).mean()

        if self.reduction == "sum":
            return focal_loss.sum()
        elif self.reduction == "mean":
            return focal_loss
        else:
            raise NotImplementedError(f"{self.reduction} hasn't been implemented.")


def collate_fn(batch):
    [batch] = batch
    return batch


def setup_logger(log_path):
    if os.path.exists(log_path):
        os.remove(log_path)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    f_handler = logging.FileHandler(log_path)
    f_handler.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(message)s")
    f_handler.setFormatter(formatter)

    logger.addHandler(f_handler)
    return logger


def train(args):
    data_names = args.problem
    difficulty = args.difficulty
    model_name = args.model
    # num_hyperconv = args.num_hyperconv
    # num_bigconv = args.num_bigconv0
    device = args.device
    nepoch = args.nepoch
    batch_size = args.batch_size
    seed = args.seed

    # for simplicity of output folder, we turn difficulty to all if all difficulties are given
    if set(difficulty) == {"tiny", "easy", "medium"}:
        difficulty = "all"

    # Handle multiple problems by creating a combined name
    combined_problems = "_".join(data_names)

    # Create the output directory path based on the new structure
    if "all" in difficulty:
        # if "all" is passed, load data under all difficulties
        out_dir = f"{_root}/runs/train/{model_name}/{combined_problems}_all"
    elif set(difficulty).issubset({"tiny", "easy", "medium"}):
        # if one or more difficulties are passed, load data under each of them
        out_dir = f"{_root}/runs/train/{model_name}/{combined_problems}"
        # Append each difficulty to the output directory name
        difficulty_suffix = "_".join(difficulty)
        out_dir = f"{out_dir}_{difficulty_suffix}"
    else:
        raise ValueError("Invalid difficulty.")

    os.makedirs(out_dir, exist_ok=True)
    model_save_path = os.path.join(out_dir, "trained_model.pkl")

    for f in os.listdir(out_dir):
        if os.path.isfile(os.path.join(out_dir, f)):
            os.remove(os.path.join(out_dir, f))

    log_file = os.path.join(out_dir, "training.log")
    logger = setup_logger(log_file)
    args_dict = vars(args)
    args_pretty = pprint.pformat(args_dict, indent=4)
    logger.info("Training arguments:\n %s", args_pretty)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    model = all_models[model_name](args).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    if args.loss == "bce":
        criterion = nn.BCEWithLogitsLoss()
    elif args.loss == "mse":
        criterion = nn.MSELoss()
    elif args.loss == "focal":
        criterion = FocalLoss(alpha=0.7, gamma=0, reduction="mean")
    else:
        raise NotImplementedError(f"{args.loss} hasn't been implemented.")

    print("Loading data...")

    # Collect sample files from all problem types and difficulties
    sample_files = []

    for data_name in data_names:
        data_dir = os.path.join(
            _root,
            "data",
            "train",
            "representation",
            "Ours",
            f"{data_name}",
        )

        if "all" in difficulty:
            # Load data from all difficulty subfolders
            for sub_folder in os.listdir(data_dir):
                sample_files += [
                    os.path.join(data_dir, sub_folder, file)
                    for file in os.listdir(os.path.join(data_dir, sub_folder))
                ]
        else:
            # Load data from specified difficulties
            for diff in difficulty:
                diff_path = os.path.join(data_dir, diff)
                if os.path.exists(diff_path):
                    sample_files += [
                        os.path.join(data_dir, diff, file)
                        for file in os.listdir(diff_path)
                    ]

    if not sample_files:
        raise ValueError(f"No data found for the specified problems and difficulties.")

    dataset = all_datasets[args.encoding](sample_files)
    if args.trail_data_size > 0 and args.trail_data_size < len(dataset):
        dataset = torch.utils.data.Subset(dataset, range(args.trail_data_size))

    train_size = int(args.split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn
    )
    data_loader = DataLoader(
        dataset, batch_size=1, shuffle=False, collate_fn=collate_fn
    )

    difficulty_str = difficulty if isinstance(difficulty, str) else "_".join(difficulty)
    print(
        f"Start training {model_name} on {combined_problems}-{difficulty_str}..."
    )

    f1s = []
    losses = []
    early_stop_count = 0
    best_val_loss = float("inf")
    best_model = None

    for epoch in range(nepoch):
        model.train()
        train_losses = []
        train_f1s = []
        accumulated_loss = torch.tensor(0.0, device=device)
        count = 0

        for one_data in tqdm(train_loader, desc=f"Epoch {epoch + 1} Training"):
            _, hypergraph, bipartite, v2e_weight, e2v_weight = one_data
            assert torch.equal(hypergraph.opt_sol, bipartite.opt_sol), (
                "the hypergraph and bipartite graph should correspond to the same instance"
            )
            hypergraph = hypergraph.to(device)
            bipartite = bipartite.to(device)
            v2e_weight = v2e_weight.to(device)
            e2v_weight = e2v_weight.to(device)
            output = model(hypergraph, bipartite, v2e_weight, e2v_weight)
            opt_sol = hypergraph.opt_sol

            output = output[
                : len(opt_sol)
            ]
            binary_output = (output > 0).long()
            loss = criterion(output, opt_sol)
            f1 = f1_score(
                opt_sol.cpu().numpy().astype(int),
                binary_output.cpu().numpy().astype(int),
                average="binary",
            )

            accumulated_loss += loss
            count += 1

            if count % batch_size == 0:
                optimizer.zero_grad()
                (accumulated_loss / batch_size).backward()
                optimizer.step()
                train_losses.append(accumulated_loss.item() / batch_size)
                train_f1s.append(f1)
                accumulated_loss = torch.tensor(0.0, device=device)

        if count % batch_size != 0:
            optimizer.zero_grad()
            (accumulated_loss / (count % batch_size)).backward()
            optimizer.step()
            train_losses.append(accumulated_loss.item() / (count % batch_size))
            train_f1s.append(f1)

        mean_train_loss = np.mean(train_losses, axis=0)
        mean_train_f1 = np.mean(train_f1s, axis=0)
        logger.info(f"Epoch {epoch + 1} training: loss: {mean_train_loss:.5f}")
        logger.info(f"Epoch {epoch + 1} training: f1: {mean_train_f1:.5f}")
        print(f"Epoch {epoch + 1} training: loss: {mean_train_loss:.5f}")
        print(f"Epoch {epoch + 1} training: f1: {mean_train_f1:.5f}")

        model.eval()
        val_losses = []
        val_f1s = []
        with torch.no_grad():
            for one_data in tqdm(val_loader, desc=f"Epoch {epoch + 1} Validation"):
                _, hypergraph, bipartite, v2e_weight, e2v_weight = one_data
                assert torch.equal(hypergraph.opt_sol, bipartite.opt_sol), (
                    "the hypergraph and bipartite graph should correspond to the same instance"
                )
                hypergraph = hypergraph.to(device)
                bipartite = bipartite.to(device)
                v2e_weight = v2e_weight.to(device)
                e2v_weight = e2v_weight.to(device)
                output = model(hypergraph, bipartite, v2e_weight, e2v_weight)
                opt_sol = hypergraph.opt_sol

                output = output[
                    : len(opt_sol)
                ]
                binary_output = (output > 0).long()
                loss = criterion(output, opt_sol)
                f1 = f1_score(
                    opt_sol.cpu().numpy().astype(int),
                    binary_output.cpu().numpy(),
                    average="binary",
                )
                val_losses.append(loss.item())
                val_f1s.append(f1)

            mean_val_loss = np.mean(val_losses, axis=0)
            mean_val_f1 = np.mean(val_f1s, axis=0)
            logger.info(f"Epoch {epoch + 1} validation: loss: {mean_val_loss:.5f}")
            logger.info(f"Epoch {epoch + 1} validation: f1: {mean_val_f1:.5f}")
            print(f"Epoch {epoch + 1} validation: loss: {mean_val_loss:.5f}")
            print(f"Epoch {epoch + 1} validation: f1: {mean_val_f1:.5f}")

            if mean_val_loss < best_val_loss:
                best_val_loss = mean_val_loss
                best_model = model
                early_stop_count = 0
                torch.save(best_model.state_dict(), model_save_path)
                print(
                    f"Model {model_name} saved to {model_save_path}."
                )
                logger.info(
                    f"Model {model_name} saved to {model_save_path}."
                )
            else:
                early_stop_count += 1
                if early_stop_count >= args.early_stop:
                    print(f"Early stop at epoch {epoch + 1}.")
                    logger.info(f"Early stop at epoch {epoch + 1}.")
                    break

        f1s.append(mean_val_f1)
        losses.append(mean_val_loss)

        if epoch + 1 in _epoch_list and args.output:
            with torch.no_grad():
                outputs = []
                for name, hypergraph, bipartite, v2e_weight, e2v_weight in tqdm(
                    data_loader, desc=f"Epoch {epoch + 1} Output"
                ):
                    hypergraph = hypergraph.to(device)

                    output = model(hypergraph)
                    output = output[: len(hypergraph.opt_sol)]

                    outputs.append((name, output.cpu().numpy()))

                pickle.dump(
                    outputs, open(f"{out_dir}/trained_model_epoch{epoch + 1}.pkl", "wb")
                )
                print(f"Epoch {epoch + 1} outputs saved.")
                logger.info(f"Epoch {epoch + 1} outputs saved.")

        if epoch + 1 in args.save_epoch:  # save model at specified epochs
            torch.save(
                best_model.state_dict(),
                os.path.join(out_dir, f"epoch_{epoch + 1}_model.pkl"),
            )
            print(f"Model saved at epoch {epoch + 1}.")
            logger.info(f"Model saved at epoch {epoch + 1}.")

    f1s = np.array(f1s)
    losses = np.array(losses)
    logger.info(f"Best valid f1: {max(f1s):.5f}")
    pickle.dump([f1s, losses], open(os.path.join(out_dir, "training_data.pkl"), "wb"))
    print(f"Training finished.\nBest validation f1: {max(f1s):.5f}")


### add parse_args function
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", "-p", nargs="+", required=True)
    parser.add_argument("--difficulty", "-d", nargs="+", required=True)
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="Ours",
    )
    parser.add_argument(
        "--encoding",
        "-e",
        type=str,
        default="hybrid",
        choices=["hybrid", "hyper", "bipartite"],
    )
    parser.add_argument("--num_hyperconv", type=int, default=6)
    parser.add_argument("--num_bigconv", type=int, default=1)
    parser.add_argument("--nout", type=int, default=1, choices=[1])
    parser.add_argument("--nhid", type=int, default=64)
    parser.add_argument("--drop_rate", type=float, default=0.1)
    parser.add_argument(
        "--hg_first_aggregate",
        type=str,
        default="sum",
        choices=["sum", "mean", "softmax_then_sum"],
    )
    parser.add_argument(
        "--hg_second_aggregate",
        type=str,
        default="mean",
        choices=["sum", "mean", "softmax_then_sum"],
    )
    parser.add_argument("--nobias", action="store_true", default=False)
    parser.add_argument("--ini_activation", type=str, default="leakyrelu")
    parser.add_argument("--hg_activation", type=str, default="leakyrelu")
    parser.add_argument("--bg_activation", type=str, default="leakyrelu")
    parser.add_argument("--final_activation", type=str, default="leakyrelu")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--nepoch", type=int, default=100)
    parser.add_argument("--early_stop", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", type=str, default="cuda:0"
    )
    parser.add_argument(
        "--loss", type=str, default="bce", choices=["mse", "bce", "focal"]
    )
    parser.add_argument(
        "--visualize",
        "-v",
        default=None,
        type=str,
        help="visualize the neural outputs of the given problem",
    )
    parser.add_argument(
        "--vepoch",
        type=int,
        default=0,
        help="0 for default epochs in one figure, -1 for default epochs in separate figures, integer for a specific epoch",
    )
    parser.add_argument(
        "--no_train", "-n", action="store_true", default=False, help="train the model"
    )
    parser.add_argument(
        "--output",
        "-o",
        action="store_true",
        default=False,
        help="output the neural outputs",
    )
    parser.add_argument("--split", type=float, default=0.8)
    parser.add_argument(
        "--trail_data_size",
        type=int,
        default=0,
        help="number of data samples for trail run",
    )
    parser.add_argument(
        "--save_epoch",
        type=int,
        nargs="*",
        default=[],
        help="save model at specified epochs",
    )
    parser.add_argument("--fix_edge", "-f", action="store_true", default=False)
    args = parser.parse_args()
    args.bias = not args.nobias
    channels_dict = {
        "hybrid": (9, 2, 4),
    }  ### 注意: 这里应该是一个3元tuple, 分别表示variable, edge和constraint的feature数量. 根据实际情报补全
    args.num_var_features, args.num_edge_features, args.num_con_features = (
        channels_dict[args.encoding]
    )

    return args


if __name__ == "__main__":
    args = parse_args()
    if not args.no_train:
        train(args=args)