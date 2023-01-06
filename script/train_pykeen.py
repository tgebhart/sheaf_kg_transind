import os
import argparse

from pykeen.pipeline import pipeline
from data_tools import get_graphs, get_factories

DATASET = 'fb15k-237'
BASE_DATA_PATH = 'data'
MODEL = 'se'
NUM_EPOCHS = 151
C0_DIM = 32
C1_DIM = 32
RANDOM_SEED = 134
DATASET_PCT = 106
GRAPH = 'train'

def run(model, dataset, num_epochs, random_seed,
        embedding_dim, c1_dimension=None, dataset_pct=DATASET_PCT,
        graph=GRAPH):

    train_graph, valid_graph, test_graph = get_graphs(dataset, dataset_pct)
    train_tf, valid_tf, test_tf = get_factories(dataset, dataset_pct)
    
    train_device = 'cuda'

    def get_train_eval_sets(graph_type):
        if graph_type == 'train':
            training_set = train_graph
            eval_set = train_tf
        elif graph_type == 'valid':
            training_set = valid_graph
            eval_set = valid_tf
        elif graph_type == 'test':
            training_set = test_graph
            eval_set = test_tf
        else:
            raise ValueError(f'unknown graph type {graph_type}')
        return training_set, eval_set
    
    training_set, testing_set = get_train_eval_sets(graph)
    
    model_kwargs = {'embedding_dim': embedding_dim, 'scoring_fct_norm': 1}
    training_kwargs = {'batch_size': 512, 'num_epochs':num_epochs}
    negative_sampler = 'basic'
    negative_sampler_kwargs = {'num_negs_per_pos': 66}
    loss = 'bceaftersigmoid'
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
        evaluation_kwargs={'batch_size': 1024}
        
    )

    mr = result.metric_results.to_df()
    print(mr[mr['Metric'] == 'hits_at_10'])
    mr['query_structure'] = 'test'

    # save out
    savedir = f'data/{dataset}/{dataset_pct}/models/{graph}/{model}/{random_seed}seed_{embedding_dim}C0_{c1_dimension}C1_{num_epochs}epochs'
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    
    result.save_to_directory(savedir)

    
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
