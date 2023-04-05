import os
import argparse
import json

from pykeen.hpo import hpo_pipeline_from_config
from data_tools import get_train_eval_inclusion_data

DATASET = 'fb15k-237'
MODEL = 'transe'
DATASET_PCT = 175
GRAPH = 'train'

def run(model, dataset, dataset_pct=DATASET_PCT, graph=GRAPH):

    rdata = get_train_eval_inclusion_data(dataset, dataset_pct, graph, graph)
    training_set = rdata['orig']['triples']
    testing_set = rdata['eval']['triples']
    
    hpo_config_loc = os.path.join(f'config/ablation/{model}_hpo_config.json')
    with open(hpo_config_loc, 'r') as f:
        config = json.load(f)
    config['pipeline']['training'] = training_set
    config['pipeline']['validation'] = testing_set
    config['pipeline']['testing'] = testing_set
    hpo_pipeline_result = hpo_pipeline_from_config(config)

    # save out
    savedir = f'data/{dataset}/{dataset_pct}/models/{graph}/{model}/ablation'
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    
    hpo_pipeline_result.save_to_directory(savedir)

    
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

