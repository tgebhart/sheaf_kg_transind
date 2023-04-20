import os
import argparse
import json

from pykeen.pipeline import pipeline_from_config
from data_tools import get_train_eval_inclusion_data, get_model_name_from_config

DATASET = 'fb15k-237'
HPO_CONFIG_NAME = 'transe_hpo_config'
DATASET_PCT = 175
GRAPH = 'train'
EVAL_GRAPH = 'valid'

def train_model(training_set, testing_set, best_hpo_loc, savedir):

    with open(best_hpo_loc, 'r') as f:
        config = json.load(f)
    config['pipeline']['training'] = training_set
    config['pipeline']['validation'] = testing_set
    config['pipeline']['testing'] = testing_set

    result = pipeline_from_config(config)

    mr = result.metric_results.to_df()
    print(mr[mr['Metric'] == 'hits_at_10'])

    # save out model
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    result.save_to_directory(savedir)

def run(hpo_config_name, dataset, dataset_pct=DATASET_PCT, graph=GRAPH, eval_graph=EVAL_GRAPH):

    model, hpo_config_name = get_model_name_from_config(hpo_config_name)

    best_hpo_loc = f'data/{dataset}/{dataset_pct}/models/train/{model}/ablation/{hpo_config_name}/best_pipeline/pipeline_config.json'
        
    # model on training dataset (graph)
    print(f'TRAINING MODEL ON {graph} GRAPH...')
    tdata = get_train_eval_inclusion_data(dataset, dataset_pct, graph, graph)
    training_set = tdata['orig']['triples']
    testing_set = tdata['eval']['triples']
    savedir = f'data/{dataset}/{dataset_pct}/models/{graph}/{model}/{hpo_config_name}/hpo_best'
    train_model(training_set, testing_set, best_hpo_loc, savedir)

    # model on eval dataset (eval_graph)
    print(f'TRAINING MODEL ON {eval_graph} GRAPH...')
    edata = get_train_eval_inclusion_data(dataset, dataset_pct, eval_graph, eval_graph)
    training_set = edata['orig']['triples']
    testing_set = edata['eval']['triples']
    savedir = f'data/{dataset}/{dataset_pct}/models/{eval_graph}/{model}/{hpo_config_name}/hpo_best'
    train_model(training_set, testing_set, best_hpo_loc, savedir)
    
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
    training_args.add_argument('--graph', type=str, required=False, default=GRAPH,
                        help='graph to train on')
    training_args.add_argument('--eval-graph', type=str, required=False, default=EVAL_GRAPH,
                        help='inductive graph to train on for transductive comparison')

    args = parser.parse_args()

    run(args.hpo_config_name, args.dataset, dataset_pct=args.dataset_pct, graph=args.graph, eval_graph=args.eval_graph)

