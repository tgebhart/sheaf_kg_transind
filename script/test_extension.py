import os
import argparse

import pandas as pd
import torch
from pykeen.evaluation import RankBasedEvaluator

from data_tools import get_train_eval_inclusion_data
from utils import expand_model_to_inductive_graph
from extension import get_extender, diffuse_interior

DATASET = "fb15k-237"
BASE_DATA_PATH = "data"
MODEL = "se"
NUM_EPOCHS = 25
C0_DIM = 32
C1_DIM = 32
RANDOM_SEED = 134
TRAINING_BATCH_SIZE = 64
EVALUATION_BATCH_SIZE = 512
DATASET_PCT = 175
ORIG_GRAPH = "train"
EVAL_GRAPH = "valid"
FROM_SAVE = True

CONVERGENCE_TOL = 1e-4
DIFFUSION_ITERATIONS = 5000
EVAL_EVERY = 500
ALPHA = 1e-1


def run(
    model,
    dataset,
    num_epochs,
    random_seed,
    embedding_dim,
    c1_dimension=None,
    evaluate_device="cuda",
    dataset_pct=DATASET_PCT,
    orig_graph_type=ORIG_GRAPH,
    eval_graph_type=EVAL_GRAPH,
    diffusion_iterations=DIFFUSION_ITERATIONS,
    evaluation_batch_size=EVALUATION_BATCH_SIZE,
    from_save=FROM_SAVE,
    alpha=ALPHA,
    eval_every=EVAL_EVERY,
    convergence_tol=CONVERGENCE_TOL,
):
    orig_savedir = f"data/{dataset}/{dataset_pct}/models/{orig_graph_type}/{model}/{random_seed}seed_{embedding_dim}C0_{c1_dimension}C1_{num_epochs}epochs"
    eval_savedir = f"data/{dataset}/{dataset_pct}/models/{eval_graph_type}/{model}/{random_seed}seed_{embedding_dim}C0_{c1_dimension}C1_{num_epochs}epochs"

    saveloc = f"data/{dataset}/{dataset_pct}/models/development/{orig_graph_type}/{model}/{random_seed}seed_{embedding_dim}C0_{c1_dimension}C1_{num_epochs}epochs"

    rdata = get_train_eval_inclusion_data(
        dataset, dataset_pct, orig_graph_type, eval_graph_type
    )
    orig_graph = rdata["orig"]["graph"]
    orig_triples = rdata["orig"]["triples"]
    eval_graph = rdata["eval"]["graph"]
    eval_triples = rdata["eval"]["triples"]
    orig_eval_entity_inclusion = rdata["inclusion"]["entities"]
    orig_eval_relation_inclusion = rdata["inclusion"]["relations"]

    # Define evaluator
    evaluator = RankBasedEvaluator()

    print("loading eval model...")
    eval_model = torch.load(os.path.join(eval_savedir, "trained_model.pkl")).to(
        evaluate_device
    )

    with torch.no_grad():
        # evaluate model trained on evaluation data first, to get upper bound on expected performance
        print("evaluating eval model...")
        eval_result = evaluator.evaluate(
            batch_size=evaluation_batch_size,
            model=eval_model,
            mapped_triples=eval_triples.mapped_triples,
            additional_filter_triples=[
                orig_triples.mapped_triples,
                eval_graph.mapped_triples,
            ],
        )
        eval_mr = eval_result.to_df()
        eval_mr.rename({"Value": "Value_eval"}, axis=1, inplace=True)
        print(f"eval result:")
        print(eval_mr[eval_mr["Metric"] == "hits_at_10"])

        print("loading original model...")
        orig_model = torch.load(os.path.join(orig_savedir, "trained_model.pkl")).to(
            evaluate_device
        )
        if from_save:
            orig_model = torch.load(os.path.join(saveloc, "trained_model.pkl")).to(
                evaluate_device
            )
            interior_mask = torch.load(os.path.join(saveloc, "interior_mask.pkl"))
        else:
            print("expanding original model to size of validation graph...")
            orig_model, interior_mask = expand_model_to_inductive_graph(
                orig_model, orig_eval_entity_inclusion, eval_graph
            )

        if not os.path.exists(saveloc):
            os.makedirs(saveloc)

        torch.save(orig_model, os.path.join(saveloc, "trained_model.pkl"))
        torch.save(interior_mask, os.path.join(saveloc, "interior_mask.pkl"))

        iteration = 0
        orig_result = evaluator.evaluate(
            batch_size=evaluation_batch_size,
            model=orig_model,
            mapped_triples=eval_triples.mapped_triples,
            additional_filter_triples=[
                orig_triples.mapped_triples,
                eval_graph.mapped_triples,
            ],
        )
        orig_mr = orig_result.to_df()
        prev_it_mr = orig_mr.copy()
        orig_mr.rename({"Value": "Value_original"}, axis=1, inplace=True)
        print(f"original model, iteration {iteration}:")
        print(orig_mr[orig_mr["Metric"] == "hits_at_10"])

        extender = get_extender(model)(model=orig_model, alpha=alpha)

        res_df = []
        for iteration in range(diffusion_iterations):
            xU = diffuse_interior(extender, eval_graph.mapped_triples, interior_mask)

            if iteration % eval_every == 0:
                print(xU.sum())

                orig_result = evaluator.evaluate(
                    batch_size=evaluation_batch_size,
                    model=orig_model,
                    device=evaluate_device,
                    mapped_triples=eval_triples.mapped_triples,
                    additional_filter_triples=[
                        orig_triples.mapped_triples,
                        eval_graph.mapped_triples,
                    ],
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
                if (
                    it_diff["iteration_difference"].values[0] < convergence_tol
                    and iteration > 10
                ):
                    break

        # save out iteration results
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
        "--num-epochs", type=int, default=NUM_EPOCHS, help="number of training epochs"
    )
    training_args.add_argument(
        "--embedding-dim", type=int, default=C0_DIM, help="entity embedding dimension"
    )
    training_args.add_argument(
        "--c1-dimension", type=int, default=C1_DIM, help="entity embedding dimension"
    )
    training_args.add_argument(
        "--dataset-pct",
        type=int,
        default=DATASET_PCT,
        help="inductive graph unknown entity relative percentage",
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
        "--orig-graph",
        type=str,
        required=False,
        default=ORIG_GRAPH,
        help="inductive graph to train on",
    )
    training_args.add_argument(
        "--eval-graph",
        type=str,
        required=False,
        default=EVAL_GRAPH,
        help="inductive graph to train on",
    )
    training_args.add_argument(
        "--batch-size",
        type=int,
        default=EVALUATION_BATCH_SIZE,
        help="evaluation batch size",
    )
    training_args.add_argument(
        "--alpha", type=float, default=ALPHA, help="diffusion learning rate (h)"
    )
    training_args.add_argument(
        "--diffusion-iterations",
        type=int,
        default=DIFFUSION_ITERATIONS,
        help="number of diffusion steps",
    )
    training_args.add_argument(
        "--eval-every",
        type=int,
        default=EVAL_EVERY,
        help="number of diffusion steps to take between each evaluation",
    )
    training_args.add_argument(
        "--convergence-tolerance",
        type=float,
        default=CONVERGENCE_TOL,
        help="diffusion convergence tolerance within which to stop diffusing",
    )

    args = parser.parse_args()

    run(
        args.model,
        args.dataset,
        args.num_epochs,
        args.random_seed,
        args.embedding_dim,
        c1_dimension=args.c1_dimension,
        dataset_pct=args.dataset_pct,
        orig_graph_type=args.orig_graph,
        eval_graph_type=args.eval_graph,
        evaluation_batch_size=args.batch_size,
        alpha=args.alpha,
        diffusion_iterations=args.diffusion_iterations,
        eval_every=args.eval_every,
        convergence_tol=args.convergence_tolerance,
    )
