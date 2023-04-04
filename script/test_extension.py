import os
import argparse

from tqdm import tqdm
import torch
from torch_geometric.utils import remove_self_loops
from pykeen.evaluation import RankBasedEvaluator
from pykeen.nn import Embedding

from data_tools import get_graphs, get_factories
from extension import SEExtender, TransEExtender, RotatEExtender, TransRExtender

DATASET = 'fb15k-237'
BASE_DATA_PATH = 'data'
MODEL = 'se'
NUM_EPOCHS = 25
C0_DIM = 32
C1_DIM = 32
RANDOM_SEED = 134
TRAINING_BATCH_SIZE = 64
EVALUATION_BATCH_SIZE = 512
DATASET_PCT = 175
ORIG_GRAPH = 'train'
EVAL_GRAPH = 'valid'
FROM_SAVE = False

CONVERGENCE_TOL = 1e-2
DIFFUSION_ITERATIONS = 5000
EVAL_EVERY = 500
ALPHA = 1e-1

def graph_entity_inclusion_map(subgraph_tf, graph_tf):
    subgraph_entity_id_to_label = subgraph_tf.entity_id_to_label
    graph_label_to_entity_id = graph_tf.entity_to_id
    return {k:graph_label_to_entity_id[v] for k,v in subgraph_entity_id_to_label.items() if v in graph_label_to_entity_id}

def graph_relation_inclusion_map(subgraph_tf, graph_tf):
    subgraph_relation_id_to_label = subgraph_tf.relation_id_to_label
    graph_label_to_relation_id = graph_tf.relation_to_id
    # relations should always be the same
    return {k:graph_label_to_relation_id[v] for k,v in subgraph_relation_id_to_label.items() if v in graph_label_to_relation_id}

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

def diffuse_interior(diffuser, triples, interior_ent_msk):
    edge_index = triples[:,[0,2]].T
    relations = triples[:,1]
    all_ents = edge_index.flatten().unique()
    num_nodes = all_ents.size(0)
    interior_vertices = all_ents[interior_ent_msk]

    # edge_index, relations = remove_self_loops(edge_index, relations)

    xU = diffuser.diffuse_interior(edge_index, relations, interior_vertices, nv=num_nodes)
    return xU
    
def expand_model(model, entity_inclusion, relation_inclusion, extended_graph, model_type):
    assert list(relation_inclusion.keys()) == list(relation_inclusion.values())
    expanded_model, interior_mask = expand_model_to_inductive_graph(model, entity_inclusion, extended_graph)
    return expanded_model, interior_mask
    
def get_extender(model_type):
    if model_type == 'se':
        return SEExtender
    if model_type == 'transe':
        return TransEExtender
    if model_type == 'rotate':
        return RotatEExtender
    if model_type == 'transr':
        return TransRExtender

def run(model, dataset, num_epochs, random_seed, 
        embedding_dim, c1_dimension=None, evaluate_device = 'cuda', 
        dataset_pct=DATASET_PCT, orig_graph_type=ORIG_GRAPH, eval_graph_type=EVAL_GRAPH,
        diffusion_iterations=DIFFUSION_ITERATIONS, evaluation_batch_size=EVALUATION_BATCH_SIZE,
        from_save=FROM_SAVE, alpha=ALPHA, eval_every=EVAL_EVERY):

    orig_savedir = f'data/{dataset}/{dataset_pct}/models/{orig_graph_type}/{model}/{random_seed}seed_{embedding_dim}C0_{c1_dimension}C1_{num_epochs}epochs'
    eval_savedir = f'data/{dataset}/{dataset_pct}/models/{eval_graph_type}/{model}/{random_seed}seed_{embedding_dim}C0_{c1_dimension}C1_{num_epochs}epochs'

    saveloc = f'data/{dataset}/{dataset_pct}/models/development/{orig_graph_type}/{model}/{random_seed}seed_{embedding_dim}C0_{c1_dimension}C1_{num_epochs}epochs'

    print('loading getting factories and graphs...')
    train_graph, valid_graph, test_graph = get_graphs(dataset, dataset_pct)
    train_tf, valid_tf, test_tf = get_factories(dataset, dataset_pct)
    
    def get_train_eval_sets(graph_type):
        if graph_type == 'train':
            training_set = train_graph
            eval_set = train_tf
        elif graph_type == 'valid':
            training_set = valid_graph
            eval_set = valid_tf
        elif graph_type == 'test':
            training_set = test_graph
            eval_set = test_tf
        else:
            raise ValueError(f'unknown graph type {graph_type}')
        return training_set, eval_set

    orig_graph, orig_triples = get_train_eval_sets(orig_graph_type)
    eval_graph, eval_triples = get_train_eval_sets(eval_graph_type)

    print('computing orig-->eval inclusion maps...')
    orig_eval_entity_inclusion = graph_entity_inclusion_map(orig_graph, eval_graph)
    orig_eval_relation_inclusion = graph_relation_inclusion_map(orig_graph, eval_graph)

    # Define evaluator
    evaluator = RankBasedEvaluator()

    print('loading eval model...')
    eval_model = torch.load(os.path.join(eval_savedir, 'trained_model.pkl')).to(evaluate_device)
    
    with torch.no_grad():

        # evaluate model trained on evaluation data first, to get upper bound on expected performance
        print('evaluating eval model...')
        eval_result  = evaluator.evaluate(
            batch_size=evaluation_batch_size,
            model=eval_model,
            mapped_triples=eval_triples.mapped_triples,
            additional_filter_triples=[orig_triples.mapped_triples,
                                        eval_graph.mapped_triples]
        )
        eval_mr = eval_result.to_df()
        eval_mr.rename({'Value':'Value_eval'}, axis=1, inplace=True)
        print(f'eval result:')
        print(eval_mr[eval_mr['Metric'] == 'hits_at_10'])

        print('loading original model...')
        orig_model = torch.load(os.path.join(orig_savedir, 'trained_model.pkl')).to(evaluate_device)
        if from_save:
            orig_model = torch.load(os.path.join(saveloc, 'trained_model.pkl')).to(evaluate_device)
            interior_mask = torch.load(os.path.join(saveloc, 'interior_mask.pkl'))
        else:
            print('expanding original model to size of validation graph...')
            orig_model, interior_mask = expand_model(orig_model, orig_eval_entity_inclusion, orig_eval_relation_inclusion, eval_graph, model)
        
        if not os.path.exists(saveloc):
            os.makedirs(saveloc)
        
        torch.save(orig_model, os.path.join(saveloc, 'trained_model.pkl'))
        torch.save(interior_mask, os.path.join(saveloc, 'interior_mask.pkl'))

        iteration = 0
        orig_result = evaluator.evaluate(
                batch_size=evaluation_batch_size,
                model=orig_model,
                mapped_triples=eval_triples.mapped_triples,
                additional_filter_triples=[orig_triples.mapped_triples,
                                        eval_graph.mapped_triples]
        )
        orig_mr = orig_result.to_df()
        prev_it_mr = orig_mr.copy()
        orig_mr.rename({'Value':'Value_original'}, axis=1, inplace=True)
        print(f'original model, iteration {iteration}:')
        print(orig_mr[orig_mr['Metric'] == 'hits_at_10'])

        lrs = torch.linspace(1e-1, 1e-2, steps=diffusion_iterations)

        # print('extending...')
        # triples = eval_graph.mapped_triples
        # xU = extend_interior_se(orig_model.to('cpu'), triples, interior_mask)
        # print()

        extender = get_extender(model)(model=orig_model, alpha=alpha)
        
        for iteration in range(diffusion_iterations):
            # print('diffusing...')
            xU = diffuse_interior(extender, eval_graph.mapped_triples, interior_mask)
            # print(xU.sum())

            if iteration % eval_every == 0:

                print(xU.sum())

                orig_result = evaluator.evaluate(
                    batch_size=evaluation_batch_size,
                    model=orig_model.to(evaluate_device),
                    mapped_triples=eval_triples.mapped_triples,
                    additional_filter_triples=[orig_triples.mapped_triples,
                                            eval_graph.mapped_triples]
                )
                it_mr = orig_result.to_df()
                diff_mr = (it_mr.merge(prev_it_mr, on=['Side','Type','Metric'], suffixes=('_diffused', '_iteration'))
                            .merge(orig_mr, on=['Side','Type','Metric'])
                            .merge(eval_mr, on=['Side','Type','Metric']))

                diff_mr['iteration_difference'] = diff_mr['Value_diffused'] - diff_mr['Value_iteration']
                diff_mr['orig_difference'] = diff_mr['Value_diffused'] - diff_mr['Value_original']
                diff_mr['eval_difference'] = diff_mr['Value_diffused'] - diff_mr['Value_eval']
                print(f'difference from orig model, iteration {iteration}:')
                print(diff_mr[diff_mr['Metric'] == 'hits_at_10'])

                prev_it_mr = it_mr

                print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='simple PyKeen training pipeline')
    # Training Hyperparameters
    training_args = parser.add_argument_group('training')
    training_args.add_argument('--dataset', type=str, default=DATASET,
                        help='dataset to run')
    training_args.add_argument('--num-epochs', type=int, default=NUM_EPOCHS,
                        help='number of training epochs')
    training_args.add_argument('--embedding-dim', type=int, default=C0_DIM,
                        help='entity embedding dimension')
    training_args.add_argument('--c1-dimension', type=int, default=C1_DIM,
                        help='entity embedding dimension')
    training_args.add_argument('--dataset-pct', type=int, default=DATASET_PCT,
                        help='inductive graph unknown entity relative percentage')                        
    training_args.add_argument('--random-seed', type=int, default=RANDOM_SEED,
                        help='random seed')
    training_args.add_argument('--model', type=str, required=False, default=MODEL,
                        help='name of model to train')
    training_args.add_argument('--orig-graph', type=str, required=False, default=ORIG_GRAPH,
                        help='inductive graph to train on')
    training_args.add_argument('--eval-graph', type=str, required=False, default=EVAL_GRAPH,
                        help='inductive graph to train on')
    training_args.add_argument('--batch-size', type=int, default=EVALUATION_BATCH_SIZE,
                        help='evaluation batch size')

    args = parser.parse_args()

    run(args.model, args.dataset, args.num_epochs, args.random_seed,
        args.embedding_dim, c1_dimension=args.c1_dimension, dataset_pct=args.dataset_pct, 
        orig_graph_type=args.orig_graph, eval_graph_type=args.eval_graph, evaluation_batch_size=args.batch_size)
