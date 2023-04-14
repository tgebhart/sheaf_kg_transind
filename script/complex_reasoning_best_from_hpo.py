import os
import argparse

from tqdm import tqdm
import pandas as pd
import torch
from pykeen.evaluation import RankBasedEvaluator

from data_tools import get_train_eval_inclusion_data, get_model_name_from_config
from complex_extension import get_complex_extender
from complex_data_info import QUERY_STRUCTURES

DATASET = 'fb15k-237'
BASE_DATA_PATH = 'data'
HPO_CONFIG_NAME = 'transe_hpo_config'
EVALUATION_BATCH_SIZE = 32
DATASET_PCT = 175
ORIG_GRAPH = 'train'
EVAL_GRAPH = 'valid'
EVALUATE_DEVICE = 'cuda'

DIFFUSION_ITERATIONS = 5000
ALPHA = 1e-1
    
def run(hpo_config_name, dataset=DATASET, evaluate_device=EVALUATE_DEVICE, 
        dataset_pct=DATASET_PCT, orig_graph_type=ORIG_GRAPH, eval_graph_type=EVAL_GRAPH,
        diffusion_iterations=DIFFUSION_ITERATIONS, evaluation_batch_size=EVALUATION_BATCH_SIZE,
        alpha=ALPHA, query_structures=QUERY_STRUCTURES):
    
    model, hpo_config_name = get_model_name_from_config(hpo_config_name)

    savedir_model = f'data/{dataset}/{dataset_pct}/models/{orig_graph_type}-{eval_graph_type}_extended/{model}/{hpo_config_name}/hpo_best'
    savedir_results = f'data/{dataset}/{dataset_pct}/complex_results/{eval_graph_type}/{model}/{hpo_config_name}/hpo_best'

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
    eval_model = torch.load(os.path.join(savedir_model, f'extended_model_{diffusion_iterations}iterations_{alpha}alpha.pkl')).to(evaluate_device)
    with torch.no_grad():

        extender = get_complex_extender(model)(model=eval_model)
        results = []
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
            result_df['query_structure'] = query_structure
            print(result_df.set_index(['Side','Type','Metric']).loc['tail','realistic','hits_at_10'])
            results.append(result_df)
            evaluator.clear()

    res_df = pd.concat(results, ignore_index=True)
    if not os.path.exists(savedir_results):
            os.makedirs(savedir_results)
    res_df.to_csv(os.path.join(savedir_results, 'complex_extension_results.csv'), index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='complex reasoning task')
    # Training Hyperparameters
    training_args = parser.add_argument_group('training')
    training_args.add_argument('--dataset', type=str, default=DATASET,
                        help='dataset to run')
    training_args.add_argument('--dataset-pct', type=int, default=DATASET_PCT,
                        help='inductive graph unknown entity relative percentage')                        
    training_args.add_argument('--hpo-config-name', type=str, default=HPO_CONFIG_NAME,
                        help='name of hyperparameter search configuration file')
    training_args.add_argument('--orig-graph', type=str, required=False, default=ORIG_GRAPH,
                        help='inductive graph to train on')
    training_args.add_argument('--eval-graph', type=str, required=False, default=EVAL_GRAPH,
                        help='inductive graph to train on')
    training_args.add_argument('--batch-size', type=int, default=EVALUATION_BATCH_SIZE,
                        help='evaluation batch size')
    training_args.add_argument('--alpha', type=float, default=ALPHA,
                        help='diffusion learning rate')
    training_args.add_argument('--diffusion-iterations', type=int, default=DIFFUSION_ITERATIONS,
                        help='number of diffusion steps')

    args = parser.parse_args()

    run(args.hpo_config_name, dataset=args.dataset, dataset_pct=args.dataset_pct, 
        orig_graph_type=args.orig_graph, eval_graph_type=args.eval_graph, evaluation_batch_size=args.batch_size,
         alpha=args.alpha, diffusion_iterations=args.diffusion_iterations)