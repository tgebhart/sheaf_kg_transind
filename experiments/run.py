"""
Copyright 2020 Twitter, Inc.
SPDX-License-Identifier: Apache-2.0
"""
import numpy as np
from tqdm import tqdm
import logging
import time

import torch
from torch_geometric.data import NeighborSampler

from experiments.parser import get_parser
from utilities.data_loading import get_dataset, set_train_val_test_split, get_missing_feature_mask
from models.GCN_models import get_model
from utilities.seeds import seeds
from utilities.extension_strategies import filling
from experiments.evaluation import test

def train(model, x, data, optimizer, critereon, train_loader=None, device="cuda"):
    model.train()

    return (
        train_sampled(model, train_loader, x, data, optimizer, critereon, device)
        if train_loader
        else train_full_batch(model, x, data, optimizer, critereon)
    )


def train_full_batch(model, x, data, optimizer, critereon):
    model.train()

    optimizer.zero_grad()
    y_pred = model(x, data.edge_index)[data.train_mask]
    y_true = data.y[data.train_mask].squeeze()

    loss = critereon(y_pred, y_true)
    loss.backward()
    optimizer.step()

    return loss


def train_sampled(model, train_loader, x, data, optimizer, critereon, device):
    model.train()

    total_loss = 0
    for batch_size, n_id, adjs in train_loader:
        # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
        adjs = [adj.to(device) for adj in adjs]
        x_batch = x[n_id]

        optimizer.zero_grad()
        y_pred = model(x_batch, adjs=adjs, full_batch=False)
        y_true = data.y[n_id[:batch_size]].squeeze()
        loss = critereon(y_pred, y_true)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        logger.debug(f"Batch loss: {loss.item():.2f}")

    return total_loss / len(train_loader)


def run(args):
    logger.info(args)

    assert not (
        args.graph_sampling and args.model != "sage"
    ), f"{args.model} model does not support training with neighborhood sampling"
    assert not (args.graph_sampling and args.jk), "Jumping Knowledge is not supported with neighborhood sampling"

    device = torch.device(
        f"cuda:{args.gpu_idx}"
        if torch.cuda.is_available() and not (args.dataset_name == "OGBN-Products" and args.model == "lp")
        else "cpu"
    )
    dataset, evaluator = get_dataset(name=args.dataset_name, homophily=args.homophily)
    data = dataset.data

    split_idx = dataset.get_idx_split() if hasattr(dataset, "get_idx_split") else None
    n_nodes, n_features = dataset.data.x.shape
    test_accs, best_val_accs, train_times = [], [], []

    train_loader = (
        NeighborSampler(
            dataset.data.edge_index,
            node_idx=split_idx["train"],
            sizes=[15, 10, 5][: args.num_layers],
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=12,
        )
        if args.graph_sampling
        else None
    )
    # Setting `sizes` to -1 simply loads all the neighbors for each node. We can do this while evaluating
    # as we first compute the representation of all nodes after the first layer (in batches), then for the second layer, and so on
    inference_loader = (
        NeighborSampler(
            dataset.data.edge_index, node_idx=None, sizes=[-1], batch_size=4096, shuffle=False, num_workers=12,
        )
        if args.graph_sampling
        else None
    )

    for seed in tqdm(seeds[: args.n_runs]):
        num_classes = dataset.num_classes
        data = set_train_val_test_split(
            seed=seed, data=dataset.data, split_idx=split_idx, dataset_name=args.dataset_name,
        ).to(device)
        train_start = time.time()
        if args.model == "lp":
            model = get_model(
                model_name=args.model,
                num_features=data.num_features,
                num_classes=num_classes,
                edge_index=data.edge_index,
                x=None,
                args=args,
            ).to(device)
            logger.info("Starting Label Propagation")
            logits = model(y=data.y, edge_index=data.edge_index, mask=data.train_mask)
            (_, val_acc, test_acc), _ = test(model=None, x=None, data=data, logits=logits, evaluator=evaluator)
        else:
            missing_feature_mask = get_missing_feature_mask(
                rate=args.missing_rate, n_nodes=n_nodes, n_features=n_features, type=args.mask_type,
            ).to(device)
            x = data.x.clone()
            # x[~missing_feature_mask] = float("nan") This makes absolutely sure that I don't use the hidden of values of x to extend! However, we make sure to apply the mask at every opportunity OTHER than to compute the laplacian. I should implement a way to make sure that works with missing values also.....
            y = data.y.clone()

            logger.debug("Starting feature filling")
            start = time.time()
            filled_features = (
                filling(args.filling_method, data.edge_index, x, y, missing_feature_mask, args.num_iterations,)
                if args.model not in ["gcnmf", "pagnn"]
                else torch.full_like(x, float("nan"))
            )
            logger.debug(f"Feature filling completed. It took: {time.time() - start:.2f}s")

            x[~missing_feature_mask] = float("nan") #I moved it down here!

            model = get_model(
                model_name=args.model,
                num_features=data.num_features,
                num_classes=num_classes,
                edge_index=data.edge_index,
                x=x,
                mask=missing_feature_mask,
                args=args,
            ).to(device)
            params = list(model.parameters())

            optimizer = torch.optim.Adam(params, lr=args.lr)
            critereon = torch.nn.NLLLoss()

            test_acc = 0
            val_accs = []
            for epoch in range(0, args.epochs):
                start = time.time()
                x = torch.where(missing_feature_mask, data.x, filled_features)

                train(
                    model, x, data, optimizer, critereon, train_loader=train_loader, device=device,
                )
                (train_acc, val_acc, tmp_test_acc), out = test(
                    model, x=x, data=data, evaluator=evaluator, inference_loader=inference_loader, device=device,
                )
                if epoch == 0 or val_acc > max(val_accs):
                    test_acc = tmp_test_acc
                    y_soft = out.softmax(dim=-1)

                val_accs.append(val_acc)
                if epoch > args.patience and max(val_accs[-args.patience :]) <= max(val_accs[: -args.patience]):
                    break
                logger.debug(
                    f"Epoch {epoch + 1} - Train acc: {train_acc:.3f}, Val acc: {val_acc:.3f}, Test acc: {tmp_test_acc:.3f}. It took {time.time() - start:.2f}s"
                )

            (_, val_acc, test_acc), _ = test(model, x=x, data=data, logits=y_soft, evaluator=evaluator)
        best_val_accs.append(val_acc)
        test_accs.append(test_acc)
        train_times.append(time.time() - train_start)

    test_acc_mean, test_acc_std = np.mean(test_accs), np.std(test_accs)
    print(f"Test Accuracy: {test_acc_mean * 100:.2f}% +- {test_acc_std * 100:.2f}")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    logger.setLevel(level=getattr(logging, args.log.upper(), None))

    run(args)