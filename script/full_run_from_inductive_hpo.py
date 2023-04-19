import argparse

DATASET = 'fb15k-237'
BASE_DATA_PATH = 'data'
HPO_CONFIG_NAME = 'transe_hpo_config'
EVALUATION_BATCH_SIZE = 32
DATASET_PCT = 175
ORIG_GRAPH = 'train'
EVAL_GRAPH = 'valid'
EVALUATION_DEVICE = 'cuda'
DIFFUSION_DEVICE = 'cuda'

CONVERGENCE_TOL = 1e-4
DIFFUSION_ITERATIONS = 5000
EVAL_EVERY = 50
ALPHA = 1e-1

from extension_hyperparameter_search import run as hpo
from train_best_from_hpo import run as train
from extend_best_from_hpo import run as extend
from complex_reasoning_best_from_hpo import run as reason


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
    training_args.add_argument('--eval-every', type=int, default=EVAL_EVERY,
                        help='number of diffusion steps to take between each evaluation')
    training_args.add_argument('--convergence-tolerance', type=float, default=CONVERGENCE_TOL,
                        help='diffusion convergence tolerance within which to stop diffusing')
    args = parser.parse_args()

    strblock = '='*25

    print(f'{strblock} HPO {strblock}')
    hpo(args.hpo_config_name, args.dataset, args.dataset_pct, args.orig_graph, args.eval_graph)
    
    print(f'{strblock} Training Best {strblock}')
    train(args.hpo_config_name, args.dataset, dataset_pct=args.dataset_pct, graph=args.orig_graph, eval_graph=args.eval_graph)
    
    print(f'{strblock} Extending Best {strblock}')
    extend(args.hpo_config_name, dataset=args.dataset, dataset_pct=args.dataset_pct, evaluate_device=args.evaluation_device, diffusion_device=args.diffusion_device,
        orig_graph_type=args.orig_graph, eval_graph_type=args.eval_graph, evaluation_batch_size=args.batch_size,
        alpha=args.alpha, diffusion_iterations=args.diffusion_iterations, eval_every=args.eval_every, convergence_tol=args.convergence_tolerance)
    
    print(f'{strblock} Complex Queries Best {strblock}')
    reason(args.hpo_config_name, dataset=args.dataset, dataset_pct=args.dataset_pct, 
        orig_graph_type=args.orig_graph, eval_graph_type=args.eval_graph, evaluation_batch_size=args.batch_size,
         alpha=args.alpha, diffusion_iterations=args.diffusion_iterations)