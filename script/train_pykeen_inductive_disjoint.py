import argparse
import os

import pandas as pd
import torch
from data_tools import get_disjoint_dataset, get_eval_graph, split_mapped_triples
from extension import diffuse_interior, get_extender
from pykeen.evaluation import RankBasedEvaluator
from pykeen.pipeline import pipeline
from utils import expand_model_to_inductive_graph

DATASET = "InductiveWN18RR"
VERSION = "v3"
MODEL = "transr"
NUM_EPOCHS = 500
C0_DIM = 16
C1_DIM = 16
RANDOM_SEED = 134
EVAL_GRAPH = "valid"

CONVERGENCE_TOL = 1e-6
DIFFUSION_ITERATIONS = 1000
EVAL_EVERY = 50
ALPHA = 1e-1


def run(
    model,
    dataset_name,
    version,
    num_epochs,
    random_seed,
    embedding_dim,
    c1_dimension=None,
    eval_graph=EVAL_GRAPH,
    diffusion_iterations=DIFFUSION_ITERATIONS,
    alpha=ALPHA,
    eval_every=EVAL_EVERY,
    convergence_tol=CONVERGENCE_TOL,
):
    train_device = "cuda"

    saveloc = f"data/{dataset_name}/{version}/models/{eval_graph}/{model}/{random_seed}seed_{embedding_dim}C0_{c1_dimension}C1_{num_epochs}epochs"

    dataset = get_disjoint_dataset(dataset_name, version)

    model_kwargs = {"embedding_dim": embedding_dim, "scoring_fct_norm": 2}
    if model == "rotate":
        model_kwargs = {"embedding_dim": embedding_dim}
    if model == "transr":
        model_kwargs["relation_dim"] = c1_dimension
    training_kwargs = {"batch_size": 512, "num_epochs": num_epochs}
    negative_sampler = "basic"
    negative_sampler_kwargs = {"num_negs_per_pos": 66}
    loss = "marginrankingloss"
    optimizer = "adam"
    optimizer_kwargs = {"lr": 0.0048731266, "weight_decay": 0}

    training_triples, eval_triples = split_mapped_triples(dataset.transductive_training)

    result = pipeline(
        model=model,
        training=training_triples,
        testing=eval_triples,
        device=train_device,
        random_seed=random_seed,
        model_kwargs=model_kwargs,
        training_kwargs=training_kwargs,
        loss=loss,
        optimizer=optimizer,
        optimizer_kwargs=optimizer_kwargs,
        negative_sampler=negative_sampler,
        negative_sampler_kwargs=negative_sampler_kwargs,
    )

    mr = result.metric_results.to_df()
    print(mr[mr["Metric"] == "hits_at_10"])
    mr["query_structure"] = "test"

    # save out
    savedir = f"data/{dataset_name}/{version}/models/train/{model}/{random_seed}seed_{embedding_dim}C0_{c1_dimension}C1_{num_epochs}epochs"
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    result.save_to_directory(savedir)

    training_model = result.model
    inference_graph = dataset.inductive_inference
    eval_triples = get_eval_graph(dataset, eval_graph)

    result_eval = pipeline(
        model=model,
        training=dataset.inductive_inference,
        testing=eval_triples,
        device=train_device,
        random_seed=random_seed,
        model_kwargs=model_kwargs,
        training_kwargs=training_kwargs,
        loss=loss,
        optimizer=optimizer,
        optimizer_kwargs=optimizer_kwargs,
        negative_sampler=negative_sampler,
        negative_sampler_kwargs=negative_sampler_kwargs,
    )

    eval_mr = result_eval.metric_results.to_df()
    print(mr[mr["Metric"] == "hits_at_10"])
    eval_mr.rename({"Value": "Value_eval"}, axis=1, inplace=True)
    print(f"eval result:")
    print(eval_mr[eval_mr["Metric"] == "hits_at_10"])

    # save out
    savedir_eval = f"data/{dataset_name}/{version}/models/inference/{model}/{random_seed}seed_{embedding_dim}C0_{c1_dimension}C1_{num_epochs}epochs"
    if not os.path.exists(savedir_eval):
        os.makedirs(savedir_eval)
    result_eval.save_to_directory(savedir_eval)

    ############## EVALUATION #################

    # Define evaluator
    evaluator = RankBasedEvaluator()

    orig_model, interior_mask = expand_model_to_inductive_graph(
        training_model, {}, inference_graph
    )

    with torch.no_grad():
        iteration = 0
        orig_result = evaluator.evaluate(
            model=orig_model,
            mapped_triples=eval_triples.mapped_triples,
            additional_filter_triples=[inference_graph.mapped_triples],
        )
        orig_mr = orig_result.to_df()
        prev_it_mr = orig_mr.copy()
        orig_mr.rename({"Value": "Value_original"}, axis=1, inplace=True)
        print(f"original model, iteration {iteration}:")
        print(orig_mr[orig_mr["Metric"] == "hits_at_10"])

        extender = get_extender(model)(model=orig_model, alpha=alpha)

        res_df = []
        for iteration in range(diffusion_iterations):
            xU = diffuse_interior(
                extender, inference_graph.mapped_triples, interior_mask
            )

            if iteration % eval_every == 0:
                print(xU.sum())
                if torch.isnan(xU.sum()):
                    raise ValueError("interior vertices contain nans")

                orig_result = evaluator.evaluate(
                    model=orig_model,
                    mapped_triples=eval_triples.mapped_triples,
                    additional_filter_triples=[inference_graph.mapped_triples],
                )
                it_mr = orig_result.to_df()
                diff_mr = (
                    it_mr.merge(
                        prev_it_mr,
                        on=["Side", "Type", "Metric"],
                        suffixes=("_diffused", "_iteration"),
                    )
                    .merge(orig_mr, on=["Side", "Type", "Metric"])
                    .merge(eval_mr, on=["Side", "Type", "Metric"])
                )

                diff_mr["iteration_difference"] = (
                    diff_mr["Value_diffused"] - diff_mr["Value_iteration"]
                )
                diff_mr["orig_difference"] = (
                    diff_mr["Value_diffused"] - diff_mr["Value_original"]
                )
                diff_mr["eval_difference"] = (
                    diff_mr["Value_diffused"] - diff_mr["Value_eval"]
                )
                diff_mr["iteration"] = iteration
                print(f"difference from orig model, iteration {iteration}:")
                print(diff_mr[diff_mr["Metric"] == "hits_at_10"])

                prev_it_mr = it_mr
                res_df.append(diff_mr)

                it_diff = diff_mr[
                    (diff_mr["Side"] == "both")
                    & (diff_mr["Type"] == "realistic")
                    & (diff_mr["Metric"] == "hits_at_10")
                ]
                # if it_diff['iteration_difference'].abs().values[0] < convergence_tol and iteration > eval_every:
                # break

        # save out iteration results
        if not os.path.exists(saveloc):
            os.makedirs(saveloc)
        res_df = pd.concat(res_df, axis=0, ignore_index=True)
        res_df.to_csv(
            os.path.join(
                saveloc, f"metrics_{diffusion_iterations}iterations_{alpha}alpha.csv"
            ),
            index=False,
        )

        # save out extended model
        torch.save(
            orig_model,
            os.path.join(
                saveloc,
                f"extended_model_{diffusion_iterations}iterations_{alpha}alpha.pkl",
            ),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="simple PyKeen training pipeline")
    # Training Hyperparameters
    training_args = parser.add_argument_group("training")
    training_args.add_argument(
        "--dataset", type=str, default=DATASET, help="dataset to run"
    )
    training_args.add_argument(
        "--version", type=str, default=VERSION, help="dataset version to run"
    )
    training_args.add_argument(
        "--num-epochs", type=int, default=NUM_EPOCHS, help="number of training epochs"
    )
    training_args.add_argument(
        "--embedding-dim", type=int, default=C0_DIM, help="entity embedding dimension"
    )
    training_args.add_argument(
        "--c1-dimension", type=int, default=C1_DIM, help="entity embedding dimension"
    )
    training_args.add_argument(
        "--random-seed", type=int, default=RANDOM_SEED, help="random seed"
    )
    training_args.add_argument(
        "--model",
        type=str,
        required=False,
        default=MODEL,
        help="name of model to train",
    )
    training_args.add_argument(
        "--eval-graph",
        type=str,
        required=False,
        default=EVAL_GRAPH,
        help="inductive graph to train on",
    )

    args = parser.parse_args()

    run(
        args.model,
        args.dataset,
        args.version,
        args.num_epochs,
        args.random_seed,
        args.embedding_dim,
        c1_dimension=args.c1_dimension,
        eval_graph=EVAL_GRAPH,
    )
