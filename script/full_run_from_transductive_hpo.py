import argparse

DATASET = 'fb15k-237'
BASE_DATA_PATH = 'data'
HPO_CONFIG_NAME = 'transe_hpo_config'
EVALUATION_BATCH_SIZE = 32
EVALUATION_SLICE_SIZE = None
DATASET_PCT = 175
ORIG_GRAPH = 'train'
EVAL_GRAPH = 'valid'
EVALUATION_DEVICE = 'cuda'
DIFFUSION_DEVICE = 'cuda'

CONVERGENCE_TOL = 1e-4
DIFFUSION_ITERATIONS = 10000
EVAL_EVERY = 100
ALPHA = 1e-1
COMPLEX_EPOCHS = 10
COMPLEX_BATCH_SIZE = 128

from validation_hyperparam_search import run as hpo
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
    training_args.add_argument('--slice-size', type=int, default=EVALUATION_SLICE_SIZE,
                        help='evaluation slice size')
    training_args.add_argument('--alpha', type=float, default=ALPHA,
                        help='diffusion learning rate (h)')
    training_args.add_argument('--diffusion-iterations', type=int, default=DIFFUSION_ITERATIONS,
                        help='number of diffusion steps')
    training_args.add_argument('--diffusion-batch-size', type=int, default=None,
                        help='batch size for diffusion')
    training_args.add_argument('--eval-every', type=int, default=EVAL_EVERY,
                        help='number of diffusion steps to take between each evaluation')
    training_args.add_argument('--convergence-tolerance', type=float, default=CONVERGENCE_TOL,
                        help='diffusion convergence tolerance within which to stop diffusing')
    training_args.add_argument('--complex_epochs', type=int, default=COMPLEX_EPOCHS, 
                        help='training epochs for complex')
    training_args.add_argument('--complex_batch_size', type=int, default=COMPLEX_BATCH_SIZE, 
                        help='Complex training batch size')
    training_args.add_argument('--train_complex', action='store_true', 
                        help='whether to run complex training')
    args = parser.parse_args()

    strblock = '='*25

    print(f'{strblock} RUNNING {args.dataset} {args.dataset_pct} {args.hpo_config_name} {strblock}')

    print(f'{strblock} HPO {strblock}')
    hpo(args.hpo_config_name, args.dataset, args.dataset_pct, args.orig_graph)
    
    print(f'{strblock} Training Best {strblock}')
    train(args.hpo_config_name, args.dataset, dataset_pct=args.dataset_pct, graph=args.orig_graph, eval_graph=args.eval_graph,
          train_complex=args.train_complex, complex_epochs=args.complex_epochs, complex_batch_size=args.complex_batch_size)
    
    print(f'{strblock} Extending Best {strblock}')
    best_iteration = extend(args.hpo_config_name, dataset=args.dataset, dataset_pct=args.dataset_pct, evaluate_device=args.evaluation_device, diffusion_device=args.diffusion_device,
        orig_graph_type=args.orig_graph, eval_graph_type='valid', evaluation_batch_size=args.batch_size, diffusion_batch_size=args.diffusion_batch_size,
        alpha=args.alpha, diffusion_iterations=args.diffusion_iterations, eval_every=args.eval_every, convergence_tol=args.convergence_tolerance,
        train_complex=False)
    
    extend(args.hpo_config_name, dataset=args.dataset, dataset_pct=args.dataset_pct, evaluate_device=args.evaluation_device, diffusion_device=args.diffusion_device,
        orig_graph_type=args.orig_graph, eval_graph_type=args.eval_graph, evaluation_batch_size=args.batch_size, diffusion_batch_size=args.diffusion_batch_size,
        alpha=args.alpha, diffusion_iterations=best_iteration, eval_every=args.eval_every, convergence_tol=args.convergence_tolerance,
        train_complex=False)
    
    print(f'{strblock} Complex Queries Best {strblock}')
    reason(args.hpo_config_name, dataset=args.dataset, dataset_pct=args.dataset_pct, 
        orig_graph_type=args.orig_graph, eval_graph_type=args.eval_graph,
        evaluation_batch_size=args.batch_size, evaluation_slice_size=args.slice_size,
        alpha=args.alpha, diffusion_iterations=args.diffusion_iterations,
        train_complex=False)
    
    if args.train_complex:
        print(f'{strblock} Extending Best {strblock}')

        best_iteration = extend(args.hpo_config_name, dataset=args.dataset, dataset_pct=args.dataset_pct, evaluate_device=args.evaluation_device, diffusion_device=args.diffusion_device,
            orig_graph_type=args.orig_graph, eval_graph_type='valid', evaluation_batch_size=args.batch_size, diffusion_batch_size=args.diffusion_batch_size,
            alpha=args.alpha, diffusion_iterations=args.diffusion_iterations, eval_every=args.eval_every, convergence_tol=args.convergence_tolerance,
            train_complex=True)

        extend(args.hpo_config_name, dataset=args.dataset, dataset_pct=args.dataset_pct, evaluate_device=args.evaluation_device, diffusion_device=args.diffusion_device,
            orig_graph_type=args.orig_graph, eval_graph_type=args.eval_graph, evaluation_batch_size=args.batch_size, diffusion_batch_size=args.diffusion_batch_size,
            alpha=args.alpha, diffusion_iterations=best_iteration, eval_every=args.eval_every, convergence_tol=args.convergence_tolerance,
            train_complex=True)
    
        print(f'{strblock} Complex Queries Best {strblock}')
        reason(args.hpo_config_name, dataset=args.dataset, dataset_pct=args.dataset_pct, 
            orig_graph_type=args.orig_graph, eval_graph_type=args.eval_graph,
            evaluation_batch_size=args.batch_size, evaluation_slice_size=args.slice_size,
            alpha=args.alpha, diffusion_iterations=args.diffusion_iterations,
            train_complex=True)