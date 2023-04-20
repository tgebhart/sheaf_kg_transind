import os
import argparse
import json
from tqdm import tqdm
from functools import partial

import optuna
import torch
from pykeen.pipeline import pipeline_from_config
from pykeen.evaluation import RankBasedEvaluator

from utils import expand_model_to_inductive_graph, suggest_value
from data_tools import split_mapped_triples, load_hpo_config, get_model_name_from_config, get_disjoint_dataset, get_eval_graph
from extension import get_extender, diffuse_interior

DATASET = 'InductiveFB15k237'
VERSION = 'v1'
HPO_CONFIG_NAME = 'se_hpo_extension_disjoint'

def evaluate(model, eval_triples, inference_graph):
    evaluator = RankBasedEvaluator()
    result = evaluator.evaluate(
                    model=model,
                    mapped_triples=eval_triples.mapped_triples,
                    additional_filter_triples=[inference_graph.mapped_triples]
    )
    result_df = result.to_df()
    res = result_df.set_index(['Side','Type','Metric']).loc['both','realistic','hits_at_10'].values[0]
    return res

def objective(orig_config, dataset, model_name, trial):
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
    inference_graph = dataset.inductive_inference
    eval_triples = get_eval_graph(dataset, 'valid')
    
    alpha = diffusion_kwargs['alpha']
    diffusion_iterations = diffusion_kwargs['diffusion_iterations']

    config['extension']['alpha'] = alpha
    config['extension']['diffusion_iterations'] = diffusion_iterations

    with torch.no_grad():

        print('expanding original model to size of validation graph...')
        model, interior_mask = expand_model_to_inductive_graph(model, {}, inference_graph)

        extender = get_extender(model_name)(model=model, alpha=alpha)
        print(f'diffusing for {diffusion_iterations} iterations with alpha {alpha}...')
        for dit in tqdm(range(diffusion_iterations), desc='diffusion'):
            xU = diffuse_interior(extender, inference_graph.mapped_triples, interior_mask)
            norm = torch.linalg.norm(xU)    
            if torch.isnan(norm).any() or torch.isinf(norm).any():
                print('INTERIOR VERTICES CONTAIN NANs or INFs, stopping diffusion')
                return 0
                
        print('evaluating complex query performance...')
        
        evaluation_metric = evaluate(extender.model, eval_triples, inference_graph)

    trial.set_user_attr('trial_config', config)
    return evaluation_metric
    

def run(hpo_config_name, dataset_name, version):

    config = load_hpo_config(hpo_config_name)
    model_name, hpo_config_name = get_model_name_from_config(hpo_config_name)
    
    dataset = get_disjoint_dataset(dataset_name, version)
    training_triples, eval_triples = split_mapped_triples(dataset.transductive_training)
    
    config['pipeline']['training'] = training_triples
    config['pipeline']['validation'] = eval_triples
    config['pipeline']['testing'] = eval_triples

    study = optuna.create_study(direction=config['optuna']['direction'])
    obj = partial(objective, config.copy(), dataset, model_name)
    
    study.optimize(obj, n_trials=config['optuna']['n_trials'])
    best_config = study.best_trial.user_attrs['trial_config']
    # remove training triples
    best_config['pipeline'] = {k:v for k,v in best_config['pipeline'].items() if k not in ('training','validation','testing')}
    
    savedir = f'data/{dataset_name}/{version}/models/train/{model_name}/ablation/{hpo_config_name}/best_pipeline'
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
    training_args.add_argument('--version', type=str, default=VERSION,
                        help='dataset version to run')
    training_args.add_argument('--hpo-config-name', type=str, default=HPO_CONFIG_NAME,
                        help='name of hyperparameter search configuration file')

    args = parser.parse_args()

    run(args.hpo_config_name, args.dataset, args.version)

