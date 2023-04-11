import os
import argparse

from tqdm import tqdm
import pandas as pd
import torch
from pykeen.evaluation import RankBasedEvaluator

from data_tools import get_train_eval_inclusion_data
from utils import expand_model_to_inductive_graph
from complex_extension import get_complex_extender
from complex_data_info import QUERY_STRUCTURES

DATASET = 'fb15k-237'
BASE_DATA_PATH = 'data'
MODEL = 'transr'
NUM_EPOCHS = 25
C0_DIM = 32
C1_DIM = 16
RANDOM_SEED = 134
TRAINING_BATCH_SIZE = 64
EVALUATION_BATCH_SIZE = 32
DATASET_PCT = 175
ORIG_GRAPH = 'train'
EVAL_GRAPH = 'valid'
FROM_SAVE = True

CONVERGENCE_TOL = 1e-4
DIFFUSION_ITERATIONS = 5000
ALPHA = 1e-1
    
def run(model, dataset, num_epochs, random_seed, 
        embedding_dim, c1_dimension=None, evaluate_device = 'cuda', 
        dataset_pct=DATASET_PCT, orig_graph_type=ORIG_GRAPH, eval_graph_type=EVAL_GRAPH,
        diffusion_iterations=DIFFUSION_ITERATIONS, evaluation_batch_size=EVALUATION_BATCH_SIZE,
        from_save=FROM_SAVE, alpha=ALPHA, convergence_tol=CONVERGENCE_TOL, query_structures=QUERY_STRUCTURES):

    saveloc = f'data/{dataset}/{dataset_pct}/models/development/{orig_graph_type}/{model}/{random_seed}seed_{embedding_dim}C0_{c1_dimension}C1_{num_epochs}epochs'

    rdata = get_train_eval_inclusion_data(dataset, dataset_pct, orig_graph_type, eval_graph_type, include_complex=True)
    orig_graph = rdata['orig']['graph']
    orig_triples = rdata['orig']['triples']
    eval_graph = rdata['eval']['graph']
    eval_triples = rdata['eval']['triples']
    orig_eval_entity_inclusion = rdata['inclusion']['entities']
    orig_eval_relation_inclusion = rdata['inclusion']['relations']

    # Define evaluator
    evaluator = RankBasedEvaluator()

    print('loading eval model...')
    eval_model = torch.load(os.path.join(saveloc, f'extended_model_{diffusion_iterations}iterations_{alpha}alpha.pkl')).to(evaluate_device)
    with torch.no_grad():

        extender = get_complex_extender(model)(model=eval_model)

        for query_structure in query_structures:
            print(f'scoring query structure {query_structure}')

            queries = rdata['complex'][query_structure]
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
            print(result_df.set_index(['Side','Type','Metric']).loc['tail','realistic','hits_at_10'])
            evaluator.clear()
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
    training_args.add_argument('--alpha', type=float, default=ALPHA,
                        help='diffusion learning rate (h)')
    training_args.add_argument('--diffusion-iterations', type=int, default=DIFFUSION_ITERATIONS,
                        help='number of diffusion steps')
    training_args.add_argument('--convergence-tolerance', type=float, default=CONVERGENCE_TOL,
                        help='diffusion convergence tolerance within which to stop diffusing')

    args = parser.parse_args()

    run(args.model, args.dataset, args.num_epochs, args.random_seed,
        args.embedding_dim, c1_dimension=args.c1_dimension, dataset_pct=args.dataset_pct, 
        orig_graph_type=args.orig_graph, eval_graph_type=args.eval_graph, evaluation_batch_size=args.batch_size,
         alpha=args.alpha, diffusion_iterations=args.diffusion_iterations, convergence_tol=args.convergence_tolerance)
