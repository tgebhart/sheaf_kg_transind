import os
import argparse

from pykeen.pipeline import pipeline
from data_tools import get_train_eval_inclusion_data
from models import SE

DATASET = 'fb15k-237'
MODEL = 'se'
NUM_EPOCHS = 25
C0_DIM = 128
C1_DIM = 128
RANDOM_SEED = 134
DATASET_PCT = 175
GRAPH = 'train'

def run(model, dataset, num_epochs, random_seed,
        embedding_dim, c1_dimension=None, dataset_pct=DATASET_PCT,
        graph=GRAPH):
    
    train_device = 'cuda'

    rdata = get_train_eval_inclusion_data(dataset, dataset_pct, graph, graph)

    training_set = rdata['orig']['triples']
    testing_set = rdata['eval']['triples']
    
    model_input = model
    model_kwargs = {'embedding_dim': embedding_dim, 'scoring_fct_norm': 2}
    if model == 'rotate':
        model_kwargs = {'embedding_dim': embedding_dim}    
    if model == 'transr':
        model_kwargs['relation_dim'] = c1_dimension
    if model == 'se':
        model_input = SE
    training_kwargs = {'batch_size': 512, 'num_epochs':num_epochs}
    negative_sampler = 'basic'
    negative_sampler_kwargs = {'num_negs_per_pos': 12}
    loss = 'marginrankingloss'
    optimizer = 'adam'
    optimizer_kwargs = {'lr': 0.0048731266, 'weight_decay': 0}
    evaluation_kwargs = {'batch_size': 128}

    result = pipeline(
        model=model_input,
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
        evaluation_kwargs=evaluation_kwargs
        
    )

    mr = result.metric_results.to_df()
    print(mr[mr['Metric'] == 'hits_at_10'])
    mr['query_structure'] = 'test'

    # save out
    savedir = f'data/{dataset}/{dataset_pct}/models/development/{graph}/{model}/{random_seed}seed_{embedding_dim}C0_{c1_dimension}C1_{num_epochs}epochs'
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
