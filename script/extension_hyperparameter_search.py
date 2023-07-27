import os
import argparse
import json
from tqdm import tqdm
from functools import partial

import optuna
import torch
import numpy as np
from pykeen.pipeline import pipeline_from_config
from pykeen.evaluation import RankBasedEvaluator

from utils import expand_model_to_inductive_graph, suggest_value
from data_tools import get_train_eval_inclusion_data, split_mapped_triples, load_hpo_config, get_model_name_from_config
from extension import get_extender, diffuse_interior
from complex_extension import get_complex_extender

DATASET = 'fb15k-237'
HPO_CONFIG_NAME = 'se_hpo_extension'
DATASET_PCT = 175
TRAIN_GRAPH = 'train'
EVAL_GRAPH = 'valid'
COMPLEX_QUERY_TARGETS = ['1p', 'ip', 'pi']
    
def evaluate_complex(model, model_name, rdata, query_structures=COMPLEX_QUERY_TARGETS, evaluation_batch_size=5):
    extender = get_complex_extender(model_name)(model=model)
    evaluator = RankBasedEvaluator()

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
        res = result_df.set_index(['Side','Type','Metric']).loc['tail','realistic','hits_at_10'].values[0]
        results.append(res)
        evaluator.clear()
    return np.mean(results)

def objective(orig_config, rdata, model_name, complex_query_targets, diffusion_batch_size, trial):
    config = orig_config.copy()
    model_kwargs_ranges = config['pipeline'].get('model_kwargs_ranges', {})
    loss_kwargs_ranges = config['pipeline'].get('loss_kwargs_ranges', {})
    optimizer_kwargs_ranges = config['pipeline'].get('optimizer_kwargs_ranges', {})
    negative_sampler_kwargs_ranges = config['pipeline'].get('negative_sampler_kwargs_ranges', {})
    diffusion_kwargs_ranges = config['extension'].get('diffusion_ranges', {})
    
    model_kwargs = {
        key: suggest_value(trial, model_kwargs_ranges[key], key) for key in model_kwargs_ranges
    }
    loss_kwargs = {
        key: suggest_value(trial, loss_kwargs_ranges[key], key) for key in loss_kwargs_ranges
    }
    optimizer_kwargs = {
        key: suggest_value(trial, optimizer_kwargs_ranges[key], key) for key in optimizer_kwargs_ranges
    }
    negative_sampler_kwargs = {
        key: suggest_value(trial, negative_sampler_kwargs_ranges[key], key) for key in negative_sampler_kwargs_ranges
    }
    diffusion_kwargs = {
        key: suggest_value(trial, diffusion_kwargs_ranges[key], key) for key in diffusion_kwargs_ranges
    }
    
    # Set non-suggested configuration values
    config['pipeline']['model_kwargs'] = config['pipeline'].get('model_kwargs', {})
    config['pipeline']['model_kwargs'].update(model_kwargs)

    config['pipeline']['loss_kwargs'] = config['pipeline'].get('loss_kwargs', {}) 
    config['pipeline']['loss_kwargs'].update(loss_kwargs)
    
    config['pipeline']['optimizer_kwargs'] = config['pipeline'].get('optimizer_kwargs', {})
    config['pipeline']['optimizer_kwargs'].update(optimizer_kwargs)
    
    config['pipeline']['negative_sampler_kwargs'] = config['pipeline'].get('negative_sampler_kwargs', {})
    config['pipeline']['negative_sampler_kwargs'].update(negative_sampler_kwargs)

    pipeline_config = config.copy()
    pipeline_config['pipeline'] = {k:v for k,v in pipeline_config['pipeline'].items() if '_ranges' not in k}

    result = pipeline_from_config(pipeline_config)
    print('training complete...')

    model = result.model
    eval_graph = rdata['eval']['graph']
    orig_eval_entity_inclusion = rdata['inclusion']['entities']
    
    alpha = diffusion_kwargs['alpha']
    diffusion_iterations = diffusion_kwargs['diffusion_iterations']

    config['extension']['alpha'] = alpha
    config['extension']['diffusion_iterations'] = diffusion_iterations

    with torch.no_grad():

        print('expanding original model to size of validation graph...')
        model, interior_mask = expand_model_to_inductive_graph(model, orig_eval_entity_inclusion, eval_graph)

        extender = get_extender(model_name)(model=model, alpha=alpha)
        print(f'diffusing for {diffusion_iterations} iterations with alpha {alpha}...')
        for dit in tqdm(range(diffusion_iterations), desc='diffusion'):
            xU = diffuse_interior(extender, eval_graph.mapped_triples, interior_mask, batch_size=diffusion_batch_size)
            norm = torch.linalg.norm(xU)    
            if torch.isnan(norm).any() or torch.isinf(norm).any():
                print('INTERIOR VERTICES CONTAIN NANs or INFs, stopping diffusion')
                return 0

        print('evaluating complex query performance...')
        evaluation_metric = evaluate_complex(extender.model, model_name, rdata, query_structures=complex_query_targets)

    trial.set_user_attr('trial_config', config)
    return evaluation_metric
    

def run(hpo_config_name, dataset, dataset_pct=DATASET_PCT, train_graph=TRAIN_GRAPH, eval_graph=EVAL_GRAPH,
        complex_query_targets=COMPLEX_QUERY_TARGETS, diffusion_batch_size=None):

    rdata = get_train_eval_inclusion_data(dataset, dataset_pct, train_graph, eval_graph, include_complex=True)
    training_set = rdata['orig']['triples']

    pipeline_training_set, pipeline_testing_set = split_mapped_triples(training_set)

    config = load_hpo_config(hpo_config_name)
    model_name, hpo_config_name = get_model_name_from_config(hpo_config_name)
    
    config['pipeline']['training'] = pipeline_training_set
    config['pipeline']['validation'] = pipeline_testing_set
    config['pipeline']['testing'] = pipeline_testing_set

    study = optuna.create_study(direction=config['optuna']['direction'])
    obj = partial(objective, config.copy(), rdata, model_name, complex_query_targets, diffusion_batch_size)
    
    study.optimize(obj, n_trials=config['optuna']['n_trials'])
    best_config = study.best_trial.user_attrs['trial_config']
    # remove training triples
    best_config['pipeline'] = {k:v for k,v in best_config['pipeline'].items() if (k not in ('training','validation','testing')) and ('_ranges' not in k)}
    best_config['pipeline']['model'] = model_name # in case custom model
    
    savedir = f'data/{dataset}/{dataset_pct}/models/{train_graph}/{model_name}/ablation/{hpo_config_name}/best_pipeline'
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    with open(os.path.join(savedir, 'pipeline_config.json'), "w") as outfile:
        json.dump(best_config, outfile)

    
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
    training_args.add_argument('--train-graph', type=str, required=False, default=TRAIN_GRAPH,
                        help='graph to train on')
    training_args.add_argument('--eval-graph', type=str, required=False, default=EVAL_GRAPH,
                        help='inductive graph to validate on')
    training_args.add_argument('--diffusion-batch-size', type=int, required=False, default=None,
                        help='diffusion batch size')

    args = parser.parse_args()

    run(args.hpo_config_name, args.dataset, dataset_pct=args.dataset_pct, 
        train_graph=args.train_graph, eval_graph=args.eval_graph, diffusion_batch_size=args.diffusion_batch_size)

