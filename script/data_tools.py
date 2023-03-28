import os
import pickle

import numpy as np
import pandas as pd
from pykeen.triples import TriplesFactory

BASE_DATA_PATH = 'data'

def find_dataset_betae(dataset, pct):
    return {
            'train': {
                'graph': f'{BASE_DATA_PATH}/{dataset}/{pct}/train_graph.txt', 
            },
            'validate': { 
                'graph': f'{BASE_DATA_PATH}/{dataset}/{pct}/val_inference.txt', 
                'queries': f'{BASE_DATA_PATH}/{dataset}/{pct}/valid_queries.pkl',
                'answers': {
                    'easy': f'{BASE_DATA_PATH}/{dataset}/{pct}/valid_answers_easy.pkl',
                    'hard': f'{BASE_DATA_PATH}/{dataset}/{pct}/valid_answers_hard.pkl',
                },
                'triplets': f'{BASE_DATA_PATH}/{dataset}/{pct}/val_predict.txt',
            },
            'test': {
                'graph': f'{BASE_DATA_PATH}/{dataset}/{pct}/test_inference.txt',
                'queries': f'{BASE_DATA_PATH}/{dataset}/{pct}/test_queries.pkl',
                'answers': {
                    'easy': f'{BASE_DATA_PATH}/{dataset}/{pct}/test_answers_easy.pkl',
                    'hard': f'{BASE_DATA_PATH}/{dataset}/{pct}/test_answers_hard.pkl',
                },
                'triplets': f'{BASE_DATA_PATH}/{dataset}/{pct}/test_predict.txt',
            },
            'id_mappings': f'{BASE_DATA_PATH}/{dataset}/{pct}/id_mappings.pkl'
    }

def create_relation_id_mapping(dstf, relation_col=1, delimiter='\t'):

    print('creating id maps...')
    train = pd.read_csv(dstf['train']['graph'], delimiter=delimiter, header=None)
    valid = pd.read_csv(dstf['validate']['graph'], delimiter=delimiter, header=None)
    test = pd.read_csv(dstf['test']['graph'], delimiter=delimiter, header=None)

    relations = np.unique(np.concatenate([train[relation_col].unique(), valid[relation_col].unique(), test[relation_col].unique()]))
    
    return {str(r):ix for ix, r in enumerate(relations)}
    

def get_graphs(dataset, pct, delimiter='\t'):

    triples_factory = TriplesFactory

    dstf = find_dataset_betae(dataset, pct)
    r2id = create_relation_id_mapping(dstf)

    train_df = pd.read_csv(dstf['train']['graph'], delimiter=delimiter, header=None)
    valid_df = pd.read_csv(dstf['validate']['graph'], delimiter=delimiter, header=None)
    test_df = pd.read_csv(dstf['test']['graph'], delimiter=delimiter, header=None)

    train = triples_factory.from_labeled_triples(train_df.astype(str).values, create_inverse_triples=False, relation_to_id=r2id)
    # now we need to combine the train and validate graphs
    validate = triples_factory.from_labeled_triples(pd.concat([train_df, valid_df]).astype(str).values, create_inverse_triples=False, relation_to_id=r2id)
    test = triples_factory.from_labeled_triples(pd.concat([train_df, test_df]).astype(str).values, create_inverse_triples=False, relation_to_id=r2id)

    return train, validate, test

def get_factories(dataset, pct):

    triples_factory = TriplesFactory

    dstf = find_dataset_betae(dataset, pct)
    train_graph, validate_graph, test_graph = get_graphs(dataset, pct)
    train = triples_factory.from_path(dstf['train']['graph'], create_inverse_triples=False, 
                                        entity_to_id=train_graph.entity_to_id, relation_to_id=train_graph.relation_to_id)
    validate = triples_factory.from_path(dstf['validate']['triplets'], create_inverse_triples=False,
                                        entity_to_id=validate_graph.entity_to_id, relation_to_id=validate_graph.relation_to_id)
    # pykeen is going to remap the indices for the training dataset, so pass this new map to test dataset
    test = triples_factory.from_path(dstf['test']['triplets'], create_inverse_triples=False,
                                        entity_to_id=test_graph.entity_to_id, relation_to_id=test_graph.relation_to_id)

    return train, validate, test