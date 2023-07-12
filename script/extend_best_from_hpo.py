import os
import argparse

import pandas as pd
import torch
from pykeen.evaluation import RankBasedEvaluator

from data_tools import get_train_eval_inclusion_data, get_model_name_from_config
from utils import expand_model_to_inductive_graph, generate_eval_logspace
from extension import get_extender, diffuse_interior

DATASET = 'fb15k-237'
BASE_DATA_PATH = 'data'
HPO_CONFIG_NAME = 'se_hpo'
EVALUATION_BATCH_SIZE = 512
DATASET_PCT = 175
ORIG_GRAPH = 'train'
EVAL_GRAPH = 'valid'
EVALUATION_DEVICE = 'cuda'
DIFFUSION_DEVICE = 'cuda'

CONVERGENCE_TOL = 1e-4
DIFFUSION_ITERATIONS = 5000
EVAL_EVERY = 50
ALPHA = 1e-1
TRAIN_COMPLEX = False

def run(hpo_config_name, dataset=DATASET, evaluate_device=EVALUATION_DEVICE, diffusion_device=DIFFUSION_DEVICE, 
        alpha=ALPHA, dataset_pct=DATASET_PCT, 
        orig_graph_type=ORIG_GRAPH, eval_graph_type=EVAL_GRAPH, diffusion_iterations=DIFFUSION_ITERATIONS,
        diffusion_batch_size=None, evaluation_batch_size=EVALUATION_BATCH_SIZE, eval_every=EVAL_EVERY, convergence_tol=CONVERGENCE_TOL,
        train_complex=TRAIN_COMPLEX):

    model, hpo_config_name = get_model_name_from_config(hpo_config_name)

    orig_savedir = f'data/{dataset}/{dataset_pct}/models/{orig_graph_type}/{model}/{hpo_config_name}/hpo_best'
    eval_savedir = f'data/{dataset}/{dataset_pct}/models/{eval_graph_type}/{model}/{hpo_config_name}/hpo_best'

    if train_complex:
        savedir_results = f'data/{dataset}/{dataset_pct}/extension_results/hpo_best/{eval_graph_type}/{model}/{hpo_config_name}/train_complex'
        savedir_model = f'data/{dataset}/{dataset_pct}/models/{orig_graph_type}-{eval_graph_type}_extended/{model}/{hpo_config_name}/hpo_best/train_complex'
    else:
        savedir_results = f'data/{dataset}/{dataset_pct}/extension_results/hpo_best/{eval_graph_type}/{model}/{hpo_config_name}'
        savedir_model = f'data/{dataset}/{dataset_pct}/models/{orig_graph_type}-{eval_graph_type}_extended/{model}/{hpo_config_name}/hpo_best'

    savename_model = f'complex_trained_model.pkl' if train_complex else 'trained_model.pkl'

    rdata = get_train_eval_inclusion_data(dataset, dataset_pct, orig_graph_type, eval_graph_type)
    orig_graph = rdata['orig']['graph']
    orig_triples = rdata['orig']['triples']
    eval_graph = rdata['eval']['graph']
    eval_triples = rdata['eval']['triples']
    orig_eval_entity_inclusion = rdata['inclusion']['entities']
    orig_eval_relation_inclusion = rdata['inclusion']['relations']

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
            device=evaluate_device,
            mapped_triples=eval_triples.mapped_triples,
            additional_filter_triples=[orig_triples.mapped_triples,
                                        eval_graph.mapped_triples]
        )
        eval_mr = eval_result.to_df()
        eval_mr.rename({'Value':'Value_eval'}, axis=1, inplace=True)
        print(f'eval result:')
        print(eval_mr[eval_mr['Metric'] == 'hits_at_10'])

        print('loading original model...')
        orig_model = torch.load(os.path.join(orig_savedir, savename_model)).to(evaluate_device)
        print('expanding original model to size of validation graph...')
        orig_model, interior_mask = expand_model_to_inductive_graph(orig_model, orig_eval_entity_inclusion, eval_graph)

        print('orig model on cuda', next(orig_model.parameters()).is_cuda)
        iteration = 0
        orig_result = evaluator.evaluate(
                batch_size=evaluation_batch_size,
                model=orig_model,
                device=evaluate_device,
                mapped_triples=eval_triples.mapped_triples,
                additional_filter_triples=[orig_triples.mapped_triples,
                                        eval_graph.mapped_triples]
        )
        orig_mr = orig_result.to_df()
        prev_it_mr = orig_mr.copy()
        orig_mr.rename({'Value':'Value_original'}, axis=1, inplace=True)
        print(f'original model, iteration {iteration}:')
        print(orig_mr[orig_mr['Metric'] == 'hits_at_10'])

        extender = get_extender(model)(model=orig_model, alpha=alpha)

        res_df = []
        eval_iterations = generate_eval_logspace(diffusion_iterations, diffusion_iterations//eval_every) 
        for iteration in range(1, diffusion_iterations+1):

            try:
                xU = diffuse_interior(extender, eval_graph.mapped_triples, interior_mask, batch_size=diffusion_batch_size)
            except torch.cuda.OutOfMemoryError as e:
                diffusion_batch_size = diffusion_batch_size // 10
                print(f'setting batch size to {diffusion_batch_size}')
                xU = diffuse_interior(extender, eval_graph.mapped_triples, interior_mask, batch_size=diffusion_batch_size)

            if iteration in eval_iterations:

                norm = torch.linalg.norm(xU)
                print(norm)
                if torch.isnan(norm) or torch.isinf(norm):
                    print('INTERIOR VERTICES CONTAIN NANs or INFs, stopping diffusion')
                    break

                orig_result = evaluator.evaluate(
                    batch_size=evaluation_batch_size,
                    model=orig_model,
                    device=evaluate_device,
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
                diff_mr['iteration'] = iteration
                print(f'difference from orig model, iteration {iteration}:')
                print(diff_mr[diff_mr['Metric'] == 'hits_at_10'])

                prev_it_mr = it_mr
                res_df.append(diff_mr)

                it_diff = diff_mr[(diff_mr['Side'] == 'both') & (diff_mr['Type'] == 'realistic') & (diff_mr['Metric'] == 'hits_at_10')]
                # if it_diff['iteration_difference'].values[0] < convergence_tol and iteration > 10:
                    # break

        # save out iteration results
        print('saving iteration results...')
        res_df = pd.concat(res_df, axis=0, ignore_index=True)
        if not os.path.exists(savedir_results):
            os.makedirs(savedir_results)
        res_df.to_csv(os.path.join(savedir_results, f'metrics_{diffusion_iterations}iterations_{alpha}alpha.csv'), index=False)

        # save out extended model
        print('saving model...')
        if not os.path.exists(savedir_model):
            os.makedirs(savedir_model)
        torch.save(orig_model, os.path.join(savedir_model, f'extended_model_{diffusion_iterations}iterations_{alpha}alpha.pkl'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='harmonic extension task')
    # Training Hyperparameters
    training_args = parser.add_argument_group('training')
    training_args.add_argument('--dataset', type=str, default=DATASET,
                        help='dataset to run')
    training_args.add_argument('--dataset-pct', type=int, default=DATASET_PCT,
                        help='inductive graph unknown entity relative percentage')                        
    training_args.add_argument('--hpo-config-name', type=str, default=HPO_CONFIG_NAME,
                        help='name of hyperparameter search configuration file')
    training_args.add_argument('--evaluation-device', type=str, required=False, default=EVALUATION_DEVICE,
                        help='device to perform evaluation on (cpu/cuda)')
    training_args.add_argument('--diffusion-device', type=str, required=False, default=DIFFUSION_DEVICE,
                        help='device to perform diffusion on (cpu/cuda)')
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
    training_args.add_argument('--diffusion-batch-size', type=int, default=None,
                        help='batch size for diffusion iteration')
    training_args.add_argument('--eval-every', type=int, default=EVAL_EVERY,
                        help='number of diffusion steps to take between each evaluation')
    training_args.add_argument('--convergence-tolerance', type=float, default=CONVERGENCE_TOL,
                        help='diffusion convergence tolerance within which to stop diffusing')
    training_args.add_argument('--train_complex', action='store_true', 
                        help='whether to run complex training')


    args = parser.parse_args()

    run(args.hpo_config_name, dataset=args.dataset, dataset_pct=args.dataset_pct, evaluate_device=args.evaluation_device, diffusion_device=args.diffusion_device,
        orig_graph_type=args.orig_graph, eval_graph_type=args.eval_graph, evaluation_batch_size=args.batch_size, diffusion_batch_size=args.diffusion_batch_size,
        alpha=args.alpha, diffusion_iterations=args.diffusion_iterations, eval_every=args.eval_every, convergence_tol=args.convergence_tolerance,
        train_complex=args.train_complex)
