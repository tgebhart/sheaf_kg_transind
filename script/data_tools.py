import os
import pickle

import numpy as np
import pandas as pd
import torch
from pykeen.triples import TriplesFactory

BASE_DATA_PATH = "data"


def find_dataset_betae(dataset, pct):
    return {
        "train": {
            "graph": f"{BASE_DATA_PATH}/{dataset}/{pct}/train_graph.txt",
        },
        "validate": {
            "graph": f"{BASE_DATA_PATH}/{dataset}/{pct}/val_inference.txt",
            "queries": f"{BASE_DATA_PATH}/{dataset}/{pct}/valid_queries.pkl",
            "answers": {
                "easy": f"{BASE_DATA_PATH}/{dataset}/{pct}/valid_answers_easy.pkl",
                "hard": f"{BASE_DATA_PATH}/{dataset}/{pct}/valid_answers_hard.pkl",
            },
            "triplets": f"{BASE_DATA_PATH}/{dataset}/{pct}/val_predict.txt",
        },
        "test": {
            "graph": f"{BASE_DATA_PATH}/{dataset}/{pct}/test_inference.txt",
            "queries": f"{BASE_DATA_PATH}/{dataset}/{pct}/test_queries.pkl",
            "answers": {
                "easy": f"{BASE_DATA_PATH}/{dataset}/{pct}/test_answers_easy.pkl",
                "hard": f"{BASE_DATA_PATH}/{dataset}/{pct}/test_answers_hard.pkl",
            },
            "triplets": f"{BASE_DATA_PATH}/{dataset}/{pct}/test_predict.txt",
        },
        "id_mappings": f"{BASE_DATA_PATH}/{dataset}/{pct}/id_mappings.pkl",
    }


def create_relation_id_mapping(dstf, relation_col=1, delimiter="\t"):
    print("creating id maps...")
    train = pd.read_csv(dstf["train"]["graph"], delimiter=delimiter, header=None)
    valid = pd.read_csv(dstf["validate"]["graph"], delimiter=delimiter, header=None)
    test = pd.read_csv(dstf["test"]["graph"], delimiter=delimiter, header=None)

    relations = np.unique(
        np.concatenate(
            [
                train[relation_col].unique(),
                valid[relation_col].unique(),
                test[relation_col].unique(),
            ]
        )
    )

    return {str(r): ix for ix, r in enumerate(relations)}


def get_graphs(dataset, pct, delimiter="\t"):
    triples_factory = TriplesFactory

    dstf = find_dataset_betae(dataset, pct)
    r2id = create_relation_id_mapping(dstf)

    train_df = pd.read_csv(dstf["train"]["graph"], delimiter=delimiter, header=None)
    valid_df = pd.read_csv(dstf["validate"]["graph"], delimiter=delimiter, header=None)
    test_df = pd.read_csv(dstf["test"]["graph"], delimiter=delimiter, header=None)

    train = triples_factory.from_labeled_triples(
        train_df.astype(str).values, create_inverse_triples=False, relation_to_id=r2id
    )
    # now we need to combine the train and validate graphs
    validate = triples_factory.from_labeled_triples(
        pd.concat([train_df, valid_df]).astype(str).values,
        create_inverse_triples=False,
        relation_to_id=r2id,
    )
    test = triples_factory.from_labeled_triples(
        pd.concat([train_df, test_df]).astype(str).values,
        create_inverse_triples=False,
        relation_to_id=r2id,
    )

    return train, validate, test


def get_factories(dataset, pct):
    triples_factory = TriplesFactory

    dstf = find_dataset_betae(dataset, pct)
    train_graph, validate_graph, test_graph = get_graphs(dataset, pct)
    train = triples_factory.from_path(
        dstf["train"]["graph"],
        create_inverse_triples=False,
        entity_to_id=train_graph.entity_to_id,
        relation_to_id=train_graph.relation_to_id,
    )
    validate = triples_factory.from_path(
        dstf["validate"]["triplets"],
        create_inverse_triples=False,
        entity_to_id=validate_graph.entity_to_id,
        relation_to_id=validate_graph.relation_to_id,
    )
    # pykeen is going to remap the indices for the training dataset, so pass this new map to test dataset
    test = triples_factory.from_path(
        dstf["test"]["triplets"],
        create_inverse_triples=False,
        entity_to_id=test_graph.entity_to_id,
        relation_to_id=test_graph.relation_to_id,
    )

    return train, validate, test


def graph_entity_inclusion_map(subgraph_tf, graph_tf):
    subgraph_entity_id_to_label = subgraph_tf.entity_id_to_label
    graph_label_to_entity_id = graph_tf.entity_to_id
    return {
        k: graph_label_to_entity_id[v]
        for k, v in subgraph_entity_id_to_label.items()
        if v in graph_label_to_entity_id
    }


def graph_relation_inclusion_map(subgraph_tf, graph_tf):
    subgraph_relation_id_to_label = subgraph_tf.relation_id_to_label
    graph_label_to_relation_id = graph_tf.relation_to_id
    # relations should always be the same
    return {
        k: graph_label_to_relation_id[v]
        for k, v in subgraph_relation_id_to_label.items()
        if v in graph_label_to_relation_id
    }


def get_train_eval_inclusion_data(
    dataset, dataset_pct, orig_graph_type, eval_graph_type
):
    print("loading factories and graphs...")
    train_graph, valid_graph, test_graph = get_graphs(dataset, dataset_pct)
    train_tf, valid_tf, test_tf = get_factories(dataset, dataset_pct)

    def get_train_eval_sets(graph_type):
        if graph_type == "train":
            training_set = train_graph
            eval_set = train_tf
        elif graph_type == "valid":
            training_set = valid_graph
            eval_set = valid_tf
        elif graph_type == "test":
            training_set = test_graph
            eval_set = test_tf
        else:
            raise ValueError(f"unknown graph type {graph_type}")
        return training_set, eval_set

    orig_graph, orig_triples = get_train_eval_sets(orig_graph_type)
    eval_graph, eval_triples = get_train_eval_sets(eval_graph_type)

    print("computing orig-->eval inclusion maps...")
    orig_eval_entity_inclusion = graph_entity_inclusion_map(orig_graph, eval_graph)
    orig_eval_relation_inclusion = graph_relation_inclusion_map(orig_graph, eval_graph)

    r = {
        "orig": {"graph": orig_graph, "triples": orig_triples},
        "eval": {"graph": eval_graph, "triples": eval_triples},
        "inclusion": {
            "entities": orig_eval_entity_inclusion,
            "relations": orig_eval_relation_inclusion,
        },
    }
    return r


def split_mapped_triples(triples_factory, train_pct=0.95):
    triples = triples_factory.mapped_triples
    ntrip = triples.size(0)
    perm = torch.randperm(ntrip)
    k = int(train_pct * ntrip)
    msk = torch.zeros(ntrip, dtype=torch.bool)
    idx = perm[:k]
    msk[idx] = True
    train_triples = triples[msk]
    eval_triples = triples[~msk]

    train = triples_factory.clone_and_exchange_triples(train_triples)
    eval = triples_factory.clone_and_exchange_triples(eval_triples)
    return train, eval
