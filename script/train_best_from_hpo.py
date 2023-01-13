import os
import argparse
import json

from pykeen.pipeline import pipeline_from_config
from data_tools import get_graphs, get_factories

DATASET = 'fb15k-237'
MODEL = 'transe'
DATASET_PCT = 175
GRAPH = 'valid'

def run(model, dataset, dataset_pct=DATASET_PCT, graph=GRAPH):

    train_graph, valid_graph, test_graph = get_graphs(dataset, dataset_pct)
    train_tf, valid_tf, test_tf = get_factories(dataset, dataset_pct)

    best_hpo_loc = f'data/{dataset}/{dataset_pct}/models/{graph}/{model}/ablation/best_pipeline/pipeline_config.json'
    
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

    with open(best_hpo_loc, 'r') as f:
        config = json.load(f)
    config['pipeline']['training'] = training_set
    config['pipeline']['validation'] = testing_set
    config['pipeline']['testing'] = testing_set

    result = pipeline_from_config(config)

    mr = result.metric_results.to_df()
    print(mr[mr['Metric'] == 'hits_at_10'])
    mr['query_structure'] = 'test'

    # save out
    savedir = f'data/{dataset}/{dataset_pct}/models/{graph}/{model}/hpo_best'
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    
    result.save_to_directory(savedir)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='simple PyKeen training pipeline')
    # Training Hyperparameters
    training_args = parser.add_argument_group('training')
    training_args.add_argument('--dataset', type=str, default=DATASET,
                        help='dataset to run')
    training_args.add_argument('--dataset-pct', type=int, default=DATASET_PCT,
                        help='inductive graph unknown entity relative percentage')
    training_args.add_argument('--model', type=str, required=False, default=MODEL,
                        help='name of model to train')
    training_args.add_argument('--graph', type=str, required=False, default=GRAPH,
                        help='inductive graph to train on')

    args = parser.parse_args()

    run(args.model, args.dataset, dataset_pct=args.dataset_pct, graph=args.graph)
