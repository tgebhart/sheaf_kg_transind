from distutils.util import strtobool
import argparse


def str2bool(x):
    # Not used currently. Can be used to turn the string 'True' into True
    if type(x) == bool:
        return x
    elif type(x) == str:
        return bool(strtobool(x))
    else:
        raise ValueError(f"Unrecognised type {type(x)}")


def get_parser():
    parser = argparse.ArgumentParser("harmonic-extension")
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="Name of dataset",
        default="Cora",
        choices=[
            "Cora",
            "CiteSeer",
            "PubMed",
            "MixHopSynthetic",
        ],
    )
    parser.add_argument(
        "--mask_type",
        type=str,
        help="Type of missing feature mask",
        default="uniform",
        choices=["uniform", "structural"],
    )
    parser.add_argument(
        "--filling_method",
        type=str,
        help="Method to solve the missing feature problem",
        default="feature_propagation",
        choices=[
            "random",
            "zero",
            "mean",
            "neighborhood_mean",
            "constant_propagation",
            "sheaf_propagation",
        ],
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Type of model to make a prediction on the downstream task",
        default="gcn",
        choices=["mlp", "sgc", "sage", "gcn", "gat", "gcnmf", "pagnn", "lp"],
    )
    parser.add_argument(
        "--missing_rate", type=float, help="Rate of node features missing", default=0.99
    )
    parser.add_argument(
        "--patience", type=int, help="Patience for early stopping", default=200
    )
    parser.add_argument("--lr", type=float, help="Learning Rate", default=0.005)
    parser.add_argument(
        "--epochs", type=int, help="Max number of epochs", default=10000
    )
    parser.add_argument("--n_runs", type=int, help="Max number of runs", default=5)
    parser.add_argument(
        "--hidden_dim", type=int, help="Hidden dimension of model", default=64
    )
    parser.add_argument(
        "--num_layers", type=int, help="Number of GNN layers", default=2
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        help="Number of diffusion iterations for feature reconstruction",
        default=40,
    )
    parser.add_argument(
        "--lp_alpha",
        type=float,
        help="Alpha parameter of label propagation",
        default=0.9,
    )
    parser.add_argument("--dropout", type=float, help="Feature dropout", default=0.5)
    parser.add_argument(
        "--jk", action="store_true", help="Whether to use the jumping knowledge scheme"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size for models trained with neighborhood sampling",
        default=1024,
    )
    parser.add_argument(
        "--graph_sampling",
        help="Set if you want to use graph sampling (always true for large graphs)",
        action="store_true",
    )
    parser.add_argument(
        "--homophily",
        type=float,
        help="Level of homophily for synthetic datasets",
        default=None,
    )
    parser.add_argument(
        "--gpu_idx", type=int, help="Indexes of gpu to run program on", default=0
    )
    parser.add_argument(
        "--log",
        type=str,
        help="Log Level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING"],
    )
    return parser
