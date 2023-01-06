import os
import argparse

from tqdm import tqdm
import torch
from torch_geometric.utils import k_hop_subgraph
from pykeen.evaluation import RankBasedEvaluator
from pykeen.nn import Embedding

from batch_harmonic_extension import step_matrix
from data_tools import get_graphs, get_factories

DATASET = 'fb15k-237'
BASE_DATA_PATH = 'data'
MODEL = 'se'
NUM_EPOCHS = 5
C0_DIM = 32
C1_DIM = 32
RANDOM_SEED = 134
TRAINING_BATCH_SIZE = 64
DATASET_PCT = 106
ORIG_GRAPH = 'train'
EVAL_GRAPH = 'valid'

CONVERGENCE_TOL = 1e-2
DIFFUSION_ITERATIONS = 50

def graph_entity_inclusion_map(subgraph_tf, graph_tf):
    subgraph_entity_id_to_label = subgraph_tf.entity_id_to_label
    graph_label_to_entity_id = graph_tf.entity_to_id
    return {k:graph_label_to_entity_id[v] for k,v in subgraph_entity_id_to_label.items() if v in graph_label_to_entity_id}

def graph_relation_inclusion_map(subgraph_tf, graph_tf):
    subgraph_relation_id_to_label = subgraph_tf.relation_id_to_label
    graph_label_to_relation_id = graph_tf.relation_to_id
    # relations should always be the same
    return {k:graph_label_to_relation_id[v] for k,v in subgraph_relation_id_to_label.items() if v in graph_label_to_relation_id}

def expand_entity_embeddings(model, boundary_vertices_original, boundary_vertices_extended, num_embeddings_new):
    oe = model.entity_representations[0]
    new_embeddings = Embedding(num_embeddings=num_embeddings_new, 
                                        shape=oe.shape,  
                                        initializer=oe.initializer,
                                        constrainer=oe.constrainer, 
                                        trainable=False).to(model.device)
    new_embeddings._embeddings.weight[boundary_vertices_extended] = torch.clone(oe._embeddings.weight[boundary_vertices_original])
    model.entity_representations[0] = new_embeddings
    return model

def check_convergence(old, new, tol=CONVERGENCE_TOL): 
    if torch.linalg.norm(old-new) < tol:
        return True
    return False

def get_model_entities(model, entities):
    nu = torch.LongTensor([0]) # null value, we just want all entities
    xt, _, _ = model._get_representations(h=entities, r=nu, t=nu, mode=None)
    return xt

def get_model_restriction_maps(model, triples):
    nu = torch.LongTensor([0]) # null value, we just want all restriction maps
    _, r, _ = model._get_representations(h=nu, r=triples[:,1], t=nu, mode=None)
    restriction_maps = torch.cat([tr.unsqueeze(1) for tr in r], dim=1)
    return restriction_maps

def expand_model_se(model, entity_inclusion, extended_graph):
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
    model = expand_entity_embeddings(model, boundary_vertices_original, boundary_vertices_extended, num_embeddings_new)
    return model, interior_ent_msk

def diffuse_interior_se(model, triples, interior_ent_msk,
                    k=1, h=1, max_iterations=1, max_nodes=100, convergence_tol=1e-2, normalized=True):

    edge_index = triples[:,[0,2]].T
    all_ents = edge_index.flatten().unique()
    num_nodes = all_ents.size(0)
    interior_vertices = all_ents[interior_ent_msk]
    boundary_ent_msk = ~interior_ent_msk

    converged = False
    iterations = 0

    while not converged and iterations < max_iterations:
        # int_orig = xt[interior_ent_msk]
        for ivix in tqdm(torch.randperm(interior_vertices.size(0)), desc='iterating over interior vertex subgraphs'):
            center_vertex = interior_vertices[ivix]
            sg_nodes, sg_edge_index, sg_node_map, sg_msk = k_hop_subgraph(center_vertex.item(), k, edge_index, 
                                                                relabel_nodes=False, num_nodes=num_nodes)

            # if the subgraph is larger than desired, randomly sample nodes and remap masks
            if sg_nodes.size(0) > max_nodes:
                rand_indices = torch.randperm(sg_edge_index.shape[1])
                keep_indices = rand_indices[:max_nodes]
                drop_indices = rand_indices[max_nodes:]
                
                sg_edge_index = sg_edge_index[:,keep_indices]
                sg_msk[sg_msk][drop_indices] = False
                sg_nodes = sg_edge_index.flatten().unique()

            # get entity representations
            xt = get_model_entities(model, sg_nodes)
            restriction_maps = get_model_restriction_maps(model, triples[sg_msk])
                
            # determine which of the subgraph nodes are boundary vertices and which are interior
            this_boundary_vertices = sg_nodes[boundary_ent_msk[sg_nodes]]
            this_interior_vertices = sg_nodes[interior_ent_msk[sg_nodes]]

            # create map for translating boundary and interior vertices into their [0,sg_nodes.size(0)] reindexed representations
            row = sg_edge_index[1]
            node_idx = row.new_full((num_nodes, ), -1)
            node_idx[sg_nodes] = torch.arange(sg_nodes.size(0))

            # create diffusion matrix from the sheaf Laplacian for the subgraph
            L_step = step_matrix(node_idx[sg_edge_index], restriction_maps.unsqueeze(0), node_idx[this_boundary_vertices], node_idx[this_interior_vertices], h=h, normalized=normalized)

            # take a diffusion step over the subgraph
            model.entity_representations[0]._embeddings.weight[this_interior_vertices] = (L_step @ xt.flatten()).reshape((sg_nodes.shape[0], xt.shape[1]))[node_idx[this_interior_vertices]]
            # check for numerical issues
            if torch.isnan(xt).any():
                print(xt)

        iterations += 1
        # converged = check_convergence(int_orig, xt[interior_ent_msk], tol=convergence_tol)

    return model
    
def expand_model(model, entity_inclusion, relation_inclusion, extended_graph, model_type):
    assert list(relation_inclusion.keys()) == list(relation_inclusion.values())
    if model_type == 'se':
        return expand_model_se(model, entity_inclusion, extended_graph)

def run(model, dataset, num_epochs, random_seed, 
        embedding_dim, c1_dimension=None, evaluate_device = 'cuda', 
        dataset_pct=DATASET_PCT, orig_graph_type=ORIG_GRAPH, eval_graph_type=EVAL_GRAPH,
        diffusion_iterations=DIFFUSION_ITERATIONS):

    orig_savedir = f'data/{dataset}/{dataset_pct}/models/{orig_graph_type}/{model}/{random_seed}seed_{embedding_dim}C0_{c1_dimension}C1_{num_epochs}epochs'
    eval_savedir = f'data/{dataset}/{dataset_pct}/models/{eval_graph_type}/{model}/{random_seed}seed_{embedding_dim}C0_{c1_dimension}C1_{num_epochs}epochs'

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

        print('evaluating eval model...')
        eval_result  = evaluator.evaluate(
            model=eval_model,
            mapped_triples=eval_triples.mapped_triples,
            additional_filter_triples=[orig_triples.mapped_triples,
                                        eval_graph.mapped_triples]
            # mapped_triples=eval_graph.mapped_triples,
            # additional_filter_triples=[orig_triples.mapped_triples]
        )
        eval_mr = eval_result.to_df()
        eval_mr.rename({'Value':'Value_eval'}, axis=1, inplace=True)
        print(f'eval result:')
        print(eval_mr[eval_mr['Metric'] == 'hits_at_10'])

        print('loading original model...')
        orig_model = torch.load(os.path.join(orig_savedir, 'trained_model.pkl')).to(evaluate_device)
        print('expanding original model to size of validation graph...')
        orig_model, interior_mask = expand_model(orig_model, orig_eval_entity_inclusion, orig_eval_relation_inclusion, eval_graph, model)

        iteration = 0
        orig_result = evaluator.evaluate(
                model=orig_model,
                mapped_triples=eval_triples.mapped_triples,
                additional_filter_triples=[orig_triples.mapped_triples,
                                        eval_graph.mapped_triples]
                # mapped_triples=eval_graph.mapped_triples,
                # additional_filter_triples=[orig_triples.mapped_triples]
        )
        orig_mr = orig_result.to_df()
        prev_it_mr = orig_mr.copy()
        orig_mr.rename({'Value':'Value_original'}, axis=1, inplace=True)
        print(f'original model, iteration {iteration}:')
        print(orig_mr[orig_mr['Metric'] == 'hits_at_10'])

        for iteration in range(1,diffusion_iterations+1):
            torch.cuda.empty_cache()
            print('diffusing model...')
            orig_model = diffuse_interior_se(orig_model, eval_graph.mapped_triples, interior_mask, 
                                            max_nodes=150, normalized=True, h=1e-1, k=1)

            print('evaluating extended model...')
            # evaluate original model
            orig_result = evaluator.evaluate(
                model=orig_model,
                mapped_triples=eval_triples.mapped_triples,
                additional_filter_triples=[orig_triples.mapped_triples,
                                        eval_graph.mapped_triples]
                # mapped_triples=eval_graph.mapped_triples,
                # additional_filter_triples=[orig_triples.mapped_triples]
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

    args = parser.parse_args()

    run(args.model, args.dataset, args.num_epochs, args.random_seed,
        args.embedding_dim, c1_dimension=args.c1_dimension, dataset_pct=args.dataset_pct, 
        orig_graph_type=args.orig_graph, eval_graph_type=args.eval_graph)
