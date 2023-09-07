import argparse
import json

DATASET = 'InductiveFB15k237'
BASE_DATA_PATH = 'data'
HPO_CONFIG_NAME = 'se_hpo_disjoint'
EVALUATION_BATCH_SIZE = 32
EVALUATION_SLICE_SIZE = None
VERSION = 'v1'
ORIG_GRAPH = 'train'
EVAL_GRAPH = 'valid'
EVALUATION_DEVICE = 'cuda'
DIFFUSION_DEVICE = 'cuda'

CONVERGENCE_TOL = 1e-4
DIFFUSION_ITERATIONS = 10000
EVAL_EVERY = 100
ALPHA = 1e-1

from extension_disjoint_hyperparameter_search import run as hpo
from train_best_from_hpo_disjoint import run as train
from extend_best_from_hpo_disjoint import run as extend
from data_tools import get_model_name_from_config

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='simple PyKeen training pipeline')
    # Training Hyperparameters
    training_args = parser.add_argument_group('training')
    training_args.add_argument('--dataset', type=str, default=DATASET,
                        help='dataset to run')
    training_args.add_argument('--version', type=str, default=VERSION,
                        help='dataset version')                        
    training_args.add_argument('--hpo-config-name', type=str, default=HPO_CONFIG_NAME,
                        help='name of hyperparameter search configuration file')
    training_args.add_argument('--evaluation-device', type=str, required=False, default=EVALUATION_DEVICE,
                        help='device to perform evaluation on (cpu/cuda)')
    training_args.add_argument('--diffusion-device', type=str, required=False, default=DIFFUSION_DEVICE,
                        help='device to perform diffusion on (cpu/cuda)')
    training_args.add_argument('--orig-graph', type=str, required=False, default=ORIG_GRAPH,
                        help='inductive graph to train on')
    training_args.add_argument('--eval-graph', type=str, required=False, default=EVAL_GRAPH,
                        help='inductive graph to test on')
    training_args.add_argument('--batch-size', type=int, default=EVALUATION_BATCH_SIZE,
                        help='evaluation batch size')
    training_args.add_argument('--slice-size', type=int, default=EVALUATION_SLICE_SIZE,
                        help='evaluation slice size')
    training_args.add_argument('--diffusion-iterations', type=int, default=DIFFUSION_ITERATIONS,
                        help='number of diffusion steps')
    training_args.add_argument('--diffusion-batch-size', type=int, default=None,
                        help='diffusion batch size')
    training_args.add_argument('--eval-every', type=int, default=EVAL_EVERY,
                        help='number of diffusion steps to take between each evaluation')
    training_args.add_argument('--convergence-tolerance', type=float, default=CONVERGENCE_TOL,
                        help='diffusion convergence tolerance within which to stop diffusing')
    args = parser.parse_args()

    strblock = '='*25

    print(f'{strblock} RUNNING {args.dataset} {args.version} {args.hpo_config_name} {strblock}')

    print(f'{strblock} HPO {strblock}')
    hpo(args.hpo_config_name, args.dataset, args.version, diffusion_batch_size=args.diffusion_batch_size)
    
    print(f'{strblock} Training Best {strblock}')
    train(args.hpo_config_name, args.dataset, args.version, train_graph=args.orig_graph, eval_graph=args.eval_graph)

    model_name, hpo_config_name = get_model_name_from_config(args.hpo_config_name)
    best_hpo_loc = f'data/{args.dataset}/{args.version}/models/train/{model_name}/ablation/{args.hpo_config_name}/best_pipeline/pipeline_config.json'
    with open(best_hpo_loc, 'r') as f:
        config = json.load(f)
        alpha = config['extension']['alpha']
    
    print(f'{strblock} Extending Best {strblock}')
    extend(args.hpo_config_name, dataset=args.dataset, version=args.version, evaluate_device=args.evaluation_device, diffusion_device=args.diffusion_device,
        orig_graph_type=args.orig_graph, eval_graph_type=args.eval_graph, eval_data_type=args.eval_graph, evaluation_batch_size=args.batch_size, diffusion_batch_size=args.diffusion_batch_size,
        alpha=alpha, diffusion_iterations=args.diffusion_iterations, eval_every=args.eval_every, convergence_tol=args.convergence_tolerance)