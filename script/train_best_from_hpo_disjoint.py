import argparse
import json
import os

from data_tools import (get_disjoint_dataset, get_eval_graph,
                        get_model_name_from_config, load_best_config)
from pykeen.pipeline import pipeline_from_config

DATASET = "InductiveFB15k237"
HPO_CONFIG_NAME = "se_hpo_extension_disjoint"
VERSION = "v1"
TRAIN_GRAPH = "train"
EVAL_GRAPH = "valid"


def train_model(training_set, testing_set, best_hpo_loc, savedir):
    config = load_best_config(best_hpo_loc)
    config["pipeline"]["training"] = training_set
    config["pipeline"]["validation"] = testing_set
    config["pipeline"]["testing"] = testing_set

    result = pipeline_from_config(config)

    mr = result.metric_results.to_df()
    print(mr[mr["Metric"] == "hits_at_10"])

    # save out model
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    result.save_to_directory(savedir)


def run(
    hpo_config_name,
    dataset_name,
    version,
    train_graph=TRAIN_GRAPH,
    eval_graph=EVAL_GRAPH,
):
    model, hpo_config_name = get_model_name_from_config(hpo_config_name)

    best_hpo_loc = f"data/{dataset_name}/{version}/models/train/{model}/ablation/{hpo_config_name}/best_pipeline/pipeline_config.json"

    dataset = get_disjoint_dataset(dataset_name, version)

    transductive = dataset.transductive_training
    inductive = get_eval_graph(dataset, eval_graph)

    # model on training dataset (graph)
    print(f"TRAINING MODEL ON {train_graph} GRAPH...")
    savedir = f"data/{dataset_name}/{version}/models/{train_graph}/{model}/{hpo_config_name}/hpo_best"
    train_model(transductive, transductive, best_hpo_loc, savedir)

    # model on eval dataset (eval_graph)
    print(f"TRAINING MODEL ON {eval_graph} GRAPH...")
    savedir = f"data/{dataset_name}/{version}/models/{eval_graph}/{model}/{hpo_config_name}/hpo_best"
    train_model(inductive, inductive, best_hpo_loc, savedir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="simple PyKeen training pipeline")
    # Training Hyperparameters
    training_args = parser.add_argument_group("training")
    training_args.add_argument(
        "--dataset", type=str, default=DATASET, help="dataset to run"
    )
    training_args.add_argument(
        "--version", type=str, default=VERSION, help="dataset version"
    )
    training_args.add_argument(
        "--hpo-config-name",
        type=str,
        default=HPO_CONFIG_NAME,
        help="name of hyperparameter search configuration file",
    )
    training_args.add_argument(
        "--train-graph",
        type=str,
        required=False,
        default=TRAIN_GRAPH,
        help="graph to train on",
    )
    training_args.add_argument(
        "--eval-graph",
        type=str,
        required=False,
        default=EVAL_GRAPH,
        help="inductive graph to train on for transductive comparison",
    )

    args = parser.parse_args()

    run(
        args.hpo_config_name,
        args.dataset,
        args.version,
        train_graph=args.train_graph,
        eval_graph=args.eval_graph,
    )
