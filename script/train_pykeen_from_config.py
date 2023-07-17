import os
import argparse
import json
from tqdm import tqdm

import torch
from torch.optim.adam import Adam
import numpy as np
from pykeen.pipeline import pipeline_from_config
from pykeen.sampling import BasicNegativeSampler
from pykeen.losses import MarginRankingLoss

from data_tools import get_train_eval_inclusion_data, get_model_name_from_config, load_best_config
from complex_extension import get_complex_extender
from complex_data_info import TRAINING_QUERY_STRUCTURES


DATASET = 'fb15k-237'
HPO_CONFIG_NAME = 'se'
DATASET_PCT = 175
GRAPH = 'train'
EVAL_GRAPH = 'valid'
TRAIN_COMPLEX = False
COMPLEX_EPOCHS = 10
COMPLEX_BATCH_SIZE = 128

def train_model(training_set, testing_set, best_hpo_loc, savedir):

    config = load_best_config(best_hpo_loc)
    config['pipeline']['training'] = training_set
    config['pipeline']['validation'] = testing_set
    config['pipeline']['testing'] = testing_set

    result = pipeline_from_config(config)

    mr = result.metric_results.to_df()
    print(mr[mr['Metric'] == 'hits_at_10'])

    # save out model
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    result.save_to_directory(savedir)
    return result.model

def create_batch_list(queries):
    return [q['hard'] for q in queries]

def create_neg_batch_list(queries, num_neg_per_pos, entities):
    ent_range = set(entities)
    return [torch.LongTensor(np.random.choice(list(ent_range.difference(set(q['hard']))), int(num_neg_per_pos*len(q['hard'])), replace=True)) for q in queries]

def create_sampler_batch(queries):
    return torch.LongTensor([[q['sources'][0], q['relations'][0], a] for q in queries for a in q['hard']])

def create_negative_queries(queries, sampler_batch, neg_sampler):
    neg_batch = neg_sampler.corrupt_batch(sampler_batch)
    neg_batch = queries.copy()
    j = 0
    for i in range(len(queries)):
        l = len(queries[i]['hard'])
        neg_batch[i]['hard'] = sampler_batch[j:j+l,2]
        j += l
    return neg_batch

def attempt_infer_best_hyperparams(best_hpo_loc):
    with open(best_hpo_loc, 'r') as f:
        config = json.load(f)
    r = {'margin':None,
         'num_negs_per_pos':None,
         'lr':None}
    if 'pipeline' in config:
        pc = config['pipeline']
        if 'loss_kwargs' in pc and 'marginranking' in pc['loss']:
            r['margin'] = pc['loss_kwargs'].get('margin', None)
        if 'negative_sampler_kwargs' in pc and pc['negative_sampler'] == 'basic':
            r['num_negs_per_pos'] = pc['negative_sampler_kwargs'].get('num_negs_per_pos', None)
        if 'optimizer_kwargs' in pc and pc['optimizer'] == 'adam':
            r['lr'] = pc['optimizer_kwargs'].get('lr', None)
    return r

def train_complex_loop(model, train_model, rdata, best_hpo_loc, savedir, query_structures=TRAINING_QUERY_STRUCTURES,
                  complex_epochs=COMPLEX_EPOCHS, complex_batch_size=COMPLEX_BATCH_SIZE):
    
    extender = get_complex_extender(model)(model=train_model)
    config = attempt_infer_best_hyperparams(best_hpo_loc)

    loss = MarginRankingLoss(margin=config['margin'])
    optimizer = Adam(train_model.get_grad_params(), lr=config['lr'])

    epochs = list(range(1,complex_epochs))

    train_model.train()
    for epoch in epochs:
        for query_structure in query_structures:

            queries = rdata['complex'][query_structure]
            sampler_batch = create_sampler_batch(queries)
            batch_list = create_batch_list(queries)
            neg_sampler = BasicNegativeSampler(mapped_triples=sampler_batch, num_negs_per_pos=config['num_negs_per_pos'])
            neg_queries = create_negative_queries(queries, sampler_batch, neg_sampler)
            neg_batch_list = create_neg_batch_list(queries, config['num_negs_per_pos'], list(range(train_model.num_entities)))

            print(f'Training query structure {query_structure}, epoch: {epoch}')

            for bix in tqdm(range(0,len(queries), complex_batch_size), desc=f'training epoch: {epoch}'):
                optimizer.zero_grad()

                batch_queries = queries[bix:bix+complex_batch_size]
                batch_neg_queries = neg_queries[bix:bix+complex_batch_size]
                this_batch_list = batch_list[bix:bix+complex_batch_size]
                this_neg_batch_list = neg_batch_list[bix:bix+complex_batch_size]

                scores = extender.slice_and_score_complex(query_structure, batch_queries, complex_batch_size, progress=False)
                neg_scores = extender.slice_and_score_complex(query_structure, batch_neg_queries, complex_batch_size, progress=False)

                col_indices = torch.cat(this_batch_list)
                row_indices = torch.cat([torch.LongTensor([i]*len(c)) for i,c in enumerate(this_batch_list)])
                hard_scores = scores[row_indices, col_indices].unsqueeze(-1)

                neg_col_indices = torch.cat(this_neg_batch_list)
                neg_row_indices = torch.cat([torch.LongTensor([i]*len(c)) for i,c in enumerate(this_neg_batch_list)])
                hard_neg_scores = neg_scores[neg_row_indices, neg_col_indices].reshape(hard_scores.shape[0], -1)
                l = loss.process_slcwa_scores(hard_scores,hard_neg_scores)

                l.backward()
                optimizer.step()
                train_model.post_parameter_update()

    torch.save(train_model, os.path.join(savedir, 'complex_trained_model.pkl'))

def run(config_name, dataset, dataset_pct=DATASET_PCT, graph=GRAPH, eval_graph=EVAL_GRAPH,
        train_complex=TRAIN_COMPLEX, query_structures=TRAINING_QUERY_STRUCTURES,
        complex_epochs=COMPLEX_EPOCHS, complex_batch_size=COMPLEX_BATCH_SIZE):

    model, config_name = get_model_name_from_config(config_name)

    config_loc = f'config/ablation/{config_name}.json'
        
    # model on training dataset (graph)
    print(f'TRAINING MODEL ON {graph} GRAPH...')
    tdata = get_train_eval_inclusion_data(dataset, dataset_pct, graph, graph, include_complex=train_complex, skip_ea=True)
    print('complete.')
    training_set = tdata['orig']['triples']
    testing_set = tdata['eval']['triples']
    savedir = f'data/{dataset}/{dataset_pct}/models/development/{graph}/{model}/{config_name}'
    trained_model = train_model(training_set, testing_set, config_loc, savedir)

    if train_complex:
        train_complex_loop(model, trained_model, tdata, config_loc, savedir, 
                    query_structures=query_structures, complex_epochs=complex_epochs, 
                    complex_batch_size=complex_batch_size)

    # model on eval dataset (eval_graph)
    print(f'TRAINING MODEL ON {eval_graph} GRAPH...')
    edata = get_train_eval_inclusion_data(dataset, dataset_pct, eval_graph, eval_graph)
    training_set = edata['orig']['triples']
    testing_set = edata['eval']['triples']
    savedir = f'data/{dataset}/{dataset_pct}/models/development/{eval_graph}/{model}/{config_name}'
    train_model(training_set, testing_set, config_loc, savedir)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='simple PyKeen training pipeline')
    # Training Hyperparameters
    training_args = parser.add_argument_group('training')
    training_args.add_argument('--dataset', type=str, default=DATASET,
                        help='dataset to run')
    training_args.add_argument('--dataset-pct', type=int, default=DATASET_PCT,
                        help='inductive graph unknown entity relative percentage')
    training_args.add_argument('--hpo-config-name', type=str, default=HPO_CONFIG_NAME,
                        help='name of hyperparameter search configuration file')
    training_args.add_argument('--graph', type=str, required=False, default=GRAPH,
                        help='graph to train on')
    training_args.add_argument('--eval-graph', type=str, required=False, default=EVAL_GRAPH,
                        help='inductive graph to train on for transductive comparison')
    training_args.add_argument('--query_structures', type=str, nargs='+', 
                        help='Query structures')
    training_args.add_argument('--complex_epochs', type=int, default=COMPLEX_EPOCHS, 
                        help='training epochs for complex')
    training_args.add_argument('--complex_batch_size', type=int, default=COMPLEX_BATCH_SIZE, 
                        help='Complex training batch size')
    training_args.add_argument('--train_complex', action='store_true', 
                        help='whether to run complex training')


    args = parser.parse_args()

    run(args.hpo_config_name, args.dataset, dataset_pct=args.dataset_pct, graph=args.graph, eval_graph=args.eval_graph,
        train_complex=args.train_complex, query_structures=args.query_structures,
        complex_epochs=args.complex_epochs, complex_batch_size=args.complex_batch_size)

