import os
import argparse
from tqdm import tqdm

import torch
from torch.optim.adam import Adam
from pykeen.pipeline import pipeline
from pykeen.sampling import BasicNegativeSampler
from pykeen.losses import MarginRankingLoss
from pykeen.evaluation import RankBasedEvaluator
import numpy as np

from data_tools import get_train_eval_inclusion_data
from complex_extension import get_complex_extender
from complex_data_info import QUERY_STRUCTURES

DATASET = 'fb15k-237'
MODEL = 'se'
NUM_EPOCHS = 10
C0_DIM = 8
C1_DIM = 8
RANDOM_SEED = 134
DATASET_PCT = 175
GRAPH = 'train'

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

def measure_evaluation(extender, evaluator, query_structure, queries, evaluation_batch_size):
    with torch.no_grad():
        
        scores = extender.slice_and_score_complex(query_structure, queries, evaluation_batch_size)
        
        for qix in tqdm(range(len(queries)), desc='evaluation'):
            q = queries[qix]
            easy_answers = q['easy']
            hard_answers = q['hard']
            nhard = hard_answers.shape[0]
            
            scores_q = scores[qix]
            true_scores = scores_q[hard_answers]
            scores_q[easy_answers] = float('nan')
            scores_q[hard_answers] = float('nan')                
            
            scores_q = scores_q.unsqueeze(0).repeat((nhard,1))
            scores_q[torch.arange(nhard), hard_answers] = true_scores
        
            evaluator.process_scores_(None, 'tail', scores_q, true_scores=true_scores.unsqueeze(dim=-1))
        result = evaluator.finalize()
        result_df = result.to_df()
        result_df['query_structure'] = query_structure
        return result_df

def run(model, dataset, num_epochs, random_seed,
        embedding_dim, c1_dimension=None, dataset_pct=DATASET_PCT,
        graph=GRAPH, query_structures=QUERY_STRUCTURES):
    
    train_device = 'cuda'

    rdata = get_train_eval_inclusion_data(dataset, dataset_pct, graph, graph, include_complex=True, skip_ea=True)

    training_set = rdata['orig']['triples']
    testing_set = rdata['eval']['triples']
    
    model_kwargs = {'embedding_dim': embedding_dim, 'scoring_fct_norm': 2}
    if model == 'rotate':
        model_kwargs = {'embedding_dim': embedding_dim}    
    if model == 'transr':
        model_kwargs['relation_dim'] = c1_dimension
    training_kwargs = {'batch_size': 512, 'num_epochs':num_epochs}
    negative_sampler = 'basic'
    negative_sampler_kwargs = {'num_negs_per_pos': 66}
    loss = 'marginrankingloss'
    optimizer = 'adam'
    optimizer_kwargs = {'lr': 0.0048731266, 'weight_decay': 0}

    result = pipeline(
        model=model,
        training=training_set,
        testing=testing_set,
        device=train_device,
        random_seed=random_seed,

        model_kwargs=model_kwargs,
        training_kwargs=training_kwargs,
        
        loss=loss,
        optimizer=optimizer,
        optimizer_kwargs=optimizer_kwargs,
        
        negative_sampler=negative_sampler, 
        negative_sampler_kwargs=negative_sampler_kwargs,
        evaluation_kwargs={'batch_size': 512}
        
    )

    mr = result.metric_results.to_df()
    print(mr[mr['Metric'] == 'hits_at_10'])
    mr['query_structure'] = 'test'

    training_batch_size = 512
    evaluation_batch_size = 2048
    complex_training_device = 'cuda'
    num_epochs = 20
    evaluate_every = 5
    learning_rate = 1e-1
    epochs = list(range(1,num_epochs+1))

    train_model = result.model.to(complex_training_device)
    extender = get_complex_extender(model)(model=train_model)
    loss = MarginRankingLoss()
    optimizer = Adam(train_model.get_grad_params(), lr=learning_rate)
    evaluator = RankBasedEvaluator()

    for query_structure in query_structures:

        queries = rdata['complex'][query_structure]
        sampler_batch = create_sampler_batch(queries)
        batch_list = create_batch_list(queries)
        neg_sampler = BasicNegativeSampler(mapped_triples=sampler_batch, num_negs_per_pos=negative_sampler_kwargs['num_negs_per_pos'])
        neg_queries = create_negative_queries(queries, sampler_batch, neg_sampler)
        neg_batch_list = create_neg_batch_list(queries, negative_sampler_kwargs['num_negs_per_pos'], list(range(train_model.num_entities)))

        results = []

        print(f'evaluating query structure: {query_structure}, before training')
        result_df = measure_evaluation(extender, evaluator, query_structure, queries, evaluation_batch_size)
        result_df['epoch'] = 0
        print(result_df.set_index(['Side','Type','Metric','epoch']).loc['tail','realistic','hits_at_10',0])
        results.append(result_df)

        for epoch in epochs:
            train_model.train()
            print(f'Training query structure {query_structure}, epoch: {epoch}')

            for bix in tqdm(range(0,len(queries), training_batch_size), desc=f'training epoch: {epoch}'):
                optimizer.zero_grad()

                batch_queries = queries[bix:bix+training_batch_size]
                batch_neg_queries = neg_queries[bix:bix+training_batch_size]
                this_batch_list = batch_list[bix:bix+training_batch_size]
                this_neg_batch_list = neg_batch_list[bix:bix+training_batch_size]

                scores = extender.slice_and_score_complex(query_structure, batch_queries, training_batch_size, progress=False)
                neg_scores = extender.slice_and_score_complex(query_structure, batch_neg_queries, training_batch_size, progress=False)

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

            if epoch % evaluate_every == 0:
                print(f'evaluating query structure: {query_structure}, epoch: {epoch}')
                result_df = measure_evaluation(extender, evaluator, query_structure, queries, evaluation_batch_size)
                result_df['epoch'] = epoch
                print(result_df.set_index(['Side','Type','Metric','epoch']).loc['tail','realistic','hits_at_10',epoch])
                results.append(result_df)

                orig_result = evaluator.evaluate(
                    batch_size=evaluation_batch_size,
                    model=train_model,
                    device=complex_training_device,
                    mapped_triples=testing_set.mapped_triples,
                    additional_filter_triples=[training_set.mapped_triples]
                )
                orig_mr = orig_result.to_df()
                orig_mr.rename({'Value':'Value_original'}, axis=1, inplace=True)
                print(f'original model, epoch {epoch}:')
                print(orig_mr[orig_mr['Metric'] == 'hits_at_10'])


    # save out
    # savedir = f'data/{dataset}/{dataset_pct}/models/development/{graph}/{model}/{random_seed}seed_{embedding_dim}C0_{c1_dimension}C1_{num_epochs}epochs'
    # if not os.path.exists(savedir):
    #     os.makedirs(savedir)
    
    # result.save_to_directory(savedir)

    
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
    training_args.add_argument('--graph', type=str, required=False, default=GRAPH,
                        help='inductive graph to train on')

    args = parser.parse_args()

    run(args.model, args.dataset, args.num_epochs, args.random_seed,
        args.embedding_dim, c1_dimension=args.c1_dimension, dataset_pct=args.dataset_pct, graph=args.graph)
