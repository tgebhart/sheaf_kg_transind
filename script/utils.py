
import torch
import numpy as np
from pykeen.nn import Embedding

def batch_chunk(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

def expand_entity_embeddings(model, boundary_vertices_original, boundary_vertices_extended, num_embeddings_new, dtype=None):
    oe = model.entity_representations[0]
    if model._get_name() == 'RotatE':
        dtype = torch.cfloat
    new_embeddings = Embedding(num_embeddings=num_embeddings_new, 
                                        shape=oe.shape,  
                                        initializer=oe.initializer,
                                        constrainer=oe.constrainer,
                                        trainable=False,
                                        dtype=dtype).to(model.device)
    new_embeddings._embeddings.weight[boundary_vertices_extended] = torch.clone(oe._embeddings.weight[boundary_vertices_original])
    model.entity_representations[0] = new_embeddings
    return model

def expand_model_to_inductive_graph(model, entity_inclusion, extended_graph):
    triples = extended_graph.mapped_triples
    edge_index = triples[:,[0,2]].T
    all_ents = edge_index.flatten().unique()

    # determine which vertices are in boundary in both train and extended (inductive) graphs
    boundary_vertices_extended = list(entity_inclusion.values())
    boundary_vertices_original = list(entity_inclusion.keys())
    # interior vertices are the set difference of all nodes and the boundary
    interior_ent_msk = torch.as_tensor([e not in boundary_vertices_extended for e in all_ents])
    interior_vertices = all_ents[interior_ent_msk]

    boundary_vertices_extended = torch.LongTensor(boundary_vertices_extended)
    boundary_vertices_original = torch.LongTensor(boundary_vertices_original)
    assert interior_vertices.shape[0] + boundary_vertices_extended.shape[0] == all_ents.shape[0]

    # expand pykeen model to accomodate new entities introduced by inductive graph
    num_embeddings_new = all_ents.shape[0]
    print('reinitializing unknown entities according to model embedding')
    model = expand_entity_embeddings(model, boundary_vertices_original, boundary_vertices_extended, num_embeddings_new)
    return model, interior_ent_msk

def generate_eval_logspace(iterations, num):
    return np.around(np.logspace(0,np.log10(int(iterations)),int(num))).astype(np.uint64)

def suggest_value(trial, value_config, name):
    value_type = value_config['type']
    low = value_config['low']
    high = value_config['high']
    q = value_config.get('q', 1)
    scale = value_config.get('scale', 'linear')
    
    logscale = True if scale == 'log' else False
    q = None if logscale else q
    if value_type == 'float':
        return trial.suggest_float(name, low, high, step=q, log=logscale)
    elif value_type == 'int':
        if scale == 'power_two':
            return 2**trial.suggest_int(name, low, high, log=False)
        return trial.suggest_int(name, low, high, log=logscale)
    else:
        raise ValueError(f"Unknown value type: {value_type}")