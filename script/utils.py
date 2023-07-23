from typing import Tuple

import numpy as np
import torch
from pykeen.models.nbase import ERModel
from pykeen.nn import Embedding
from typing import Dict, Any
from data_tools import TriplesFactory


def batch_chunk(seq, size):
    return (seq[pos : pos + size] for pos in range(0, len(seq), size))


def expand_entity_embeddings(
    model: ERModel,
    boundary_vertices_original: torch.Tensor,
    boundary_vertices_extended: torch.Tensor,
    num_embeddings_new: int,
    dtype=None,
):
    """
       This function creates a new embedding of the knowledge graph
       by merging the embeddings for the old entities with new embeddings (untrained)
       for the unseen entities.

    Args:
        model (ERModel): _description_
        boundary_vertices_original (torch.Tensor): _description_
        boundary_vertices_extended (torch.Tensor): _description_
        num_embeddings_new (_type_): Number of embeddings to generate (old + new)
        dtype (_type_, optional): Datatype of the embeddings. Defaults to None.

    Returns:
        _type_: _description_
    """
    oe = model.entity_representations[0]
    if model._get_name() == "RotatE":
        dtype = torch.cfloat
    new_embeddings = Embedding(
        num_embeddings=num_embeddings_new,
        shape=oe.shape,
        initializer=oe.initializer,
        constrainer=oe.constrainer,
        trainable=False,
        dtype=dtype,
    ).to(model.device)
    new_embeddings._embeddings.weight[boundary_vertices_extended] = torch.clone(
        oe._embeddings.weight[boundary_vertices_original]
    )
    model.entity_representations[0] = new_embeddings
    return model


def expand_model_to_inductive_graph(
    model: ERModel, entity_inclusion: Dict[int, int], extended_graph: TriplesFactory
) -> Tuple[ERModel, torch.Tensor]:
    """
    Expands the model to account for unseen entities by initializing embeddings of 
    unseen (interior) entities, and maintaining embeddings of known entities.

    Args:
        model (ERModel): The pykeen model to expand.
        entity_inclusion (Dict[int, int]): mapping of original graph entity ids to extended graph entity ids
        extended_graph (TripleFactory): the extended graph.

    Returns:
        Tuple[ERModel, torch.Tensor]: A pair, consisting of the model with the updated embeddings
        for the extended graph, and a tensor with the (boolean) indices of all "interior" vertices. 
    """
    triples = extended_graph.mapped_triples
    edge_index = triples[:, [0, 2]].T
    all_ents = edge_index.flatten().unique()

    # determine which vertices are in boundary in both train and extended (inductive) graphs
    boundary_vertices_extended = list(entity_inclusion.values())
    boundary_vertices_original = list(entity_inclusion.keys())
    # interior vertices are the set difference of all nodes and the boundary
    # They are the entities which we have not seen; i.e. they are not shared between the 
    # original + extended graphs.
    interior_ent_msk = torch.as_tensor(
        [e not in boundary_vertices_extended for e in all_ents]
    )
    interior_vertices = all_ents[interior_ent_msk]

    boundary_vertices_extended = torch.LongTensor(boundary_vertices_extended)
    boundary_vertices_original = torch.LongTensor(boundary_vertices_original)
    assert (
        interior_vertices.shape[0] + boundary_vertices_extended.shape[0]
        == all_ents.shape[0]
    )

    # expand pykeen model to accomodate new entities introduced by inductive graph
    num_embeddings_new = all_ents.shape[0]
    print("reinitializing unknown entities according to model embedding")
    model = expand_entity_embeddings(
        model,
        boundary_vertices_original,
        boundary_vertices_extended,
        num_embeddings_new,
    )
    return model, interior_ent_msk


def generate_eval_logspace(iterations, num) -> np.uint64:
    return np.around(np.logspace(0, np.log10(int(iterations)), int(num))).astype(
        np.uint64
    )


def suggest_value(trial, value_config, name):
    value_type = value_config["type"]
    low = value_config["low"]
    high = value_config["high"]
    q = value_config.get("q", 1)
    scale = value_config.get("scale", "linear")

    logscale = True if scale == "log" else False
    q = None if logscale else q
    if value_type == "float":
        return trial.suggest_float(name, low, high, step=q, log=logscale)
    elif value_type == "int":
        if scale == "power_two":
            return 2 ** trial.suggest_int(name, low, high, log=False)
        return trial.suggest_int(name, low, high, log=logscale)
    else:
        raise ValueError(f"Unknown value type: {value_type}")
