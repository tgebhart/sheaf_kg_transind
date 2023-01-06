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

# def create_or_load_id_mappings(dstf, entity_cols=[0,2], relation_col=1, delimiter='\t'):
#     if os.path.exists(dstf['id_mappings']):
#         print('found pre-computed id mappings, returning these')
#         with open(dstf['id_mappings'], 'rb') as f:
#             return pickle.load(f)

#     print('creating id maps...')
#     train = pd.read_csv(dstf['train']['graph'], delimiter=delimiter, header=None)
#     valid = pd.read_csv(dstf['validate']['graph'], delimiter=delimiter, header=None)
#     test = pd.read_csv(dstf['test']['graph'], delimiter=delimiter, header=None)

#     relations = np.unique(np.concatenate([train[relation_col].unique(), valid[relation_col].unique(), test[relation_col].unique()]))
#     entities = []
#     entities.append(np.unique(train[entity_cols].values.flatten()))
#     entities.append(np.unique(valid[entity_cols].values.flatten()))
#     entities.append(np.unique(test[entity_cols].values.flatten()))
#     entities = np.unique(np.concatenate(entities))

#     ret = {}
#     ret['all'] = {}
#     ret['train'] = {}
#     ret['validate'] = {}
#     ret['test'] = {}

#     # full mapping
#     ret['all']['e2id'] = {str(e):ix for ix, e in enumerate(entities)}
#     ret['all']['r2id'] = {str(r):ix for ix, r in enumerate(relations)}
#     # training mapping
#     ret['train']['e2id'] = {str(e):ret['all']['e2id'][str(e)] for e in np.unique(train[entity_cols].values.flatten())}
#     ret['train']['r2id'] = {str(r):ret['all']['r2id'][str(r)] for r in np.unique(train[relation_col].values.flatten())}
#     # validation mapping
#     ret['validate']['e2id'] = {str(e):ret['all']['e2id'][str(e)] for e in np.unique(valid[entity_cols].values.flatten())}
#     ret['validate']['r2id'] = {str(r):ret['all']['r2id'][str(r)] for r in np.unique(valid[relation_col].values.flatten())}
#     # test mapping
#     ret['test']['e2id'] = {str(e):ret['all']['e2id'][str(e)] for e in np.unique(test[entity_cols].values.flatten())}
#     ret['test']['r2id'] = {str(r):ret['all']['r2id'][str(r)] for r in np.unique(test[relation_col].values.flatten())}

#     with open(dstf['id_mappings'], 'wb') as f:
#         print('saving created mappings...')
#         pickle.dump(ret, f)

#     return ret

# def create_or_load_id_mappings(dstf, entity_cols=[0,2], relation_col=1, delimiter='\t'):
#     if os.path.exists(dstf['id_mappings']):
#         print('found pre-computed id mappings, returning these')
#         with open(dstf['id_mappings'], 'rb') as f:
#             return pickle.load(f)

#     print('creating id maps...')
#     train = pd.read_csv(dstf['train']['graph'], delimiter=delimiter, header=None)
#     valid = pd.read_csv(dstf['validate']['graph'], delimiter=delimiter, header=None)
#     test = pd.read_csv(dstf['test']['graph'], delimiter=delimiter, header=None)

#     relations = np.unique(np.concatenate([train[relation_col].unique(), valid[relation_col].unique(), test[relation_col].unique()]))
#     entities = []
#     entities.append(np.unique(train[entity_cols].values.flatten()))
#     entities.append(np.unique(valid[entity_cols].values.flatten()))
#     entities.append(np.unique(test[entity_cols].values.flatten()))
#     entities = np.unique(np.concatenate(entities))

#     ret = {}
#     ret['all'] = {}
#     ret['train'] = {}
#     ret['validate'] = {}
#     ret['test'] = {}

#     # full mapping
#     ret['all']['e2id'] = {str(e):ix for ix, e in enumerate(entities)}
#     ret['all']['r2id'] = {str(r):ix for ix, r in enumerate(relations)}
#     # training mapping
#     ret['train']['e2id'] = ret['all']['e2id']
#     ret['train']['r2id'] = ret['all']['r2id']
#     # validation mapping
#     ret['validate']['e2id'] = ret['all']['e2id']
#     ret['validate']['r2id'] = ret['all']['r2id']
#     # test mapping
#     ret['test']['e2id'] = ret['all']['e2id']
#     ret['test']['r2id'] = ret['all']['r2id']

#     with open(dstf['id_mappings'], 'wb') as f:
#         print('saving created mappings...')
#         pickle.dump(ret, f)

#     return ret


def create_relation_id_mapping(dstf, relation_col=1, delimiter='\t'):

    print('creating id maps...')
    train = pd.read_csv(dstf['train']['graph'], delimiter=delimiter, header=None)
    valid = pd.read_csv(dstf['validate']['graph'], delimiter=delimiter, header=None)
    test = pd.read_csv(dstf['test']['graph'], delimiter=delimiter, header=None)

    relations = np.unique(np.concatenate([train[relation_col].unique(), valid[relation_col].unique(), test[relation_col].unique()]))
    
    return {str(r):ix for ix, r in enumerate(relations)}
    

# def get_graphs(dataset, pct, id_mappings=None):

#     triples_factory = TriplesFactory

#     dstf = find_dataset_betae(dataset, pct)
#     if id_mappings is None:
#         id_mappings = create_or_load_id_mappings(dstf)
#     train = triples_factory.from_path(dstf['train']['graph'], create_inverse_triples=False, 
#                                         entity_to_id=id_mappings['train']['e2id'], relation_to_id=id_mappings['train']['r2id'])
#     validate = triples_factory.from_path(dstf['validate']['graph'], create_inverse_triples=False,
#                                         entity_to_id=id_mappings['validate']['e2id'], relation_to_id=id_mappings['validate']['r2id'])
#     # pykeen is going to remap the indices for the training dataset, so pass this new map to test dataset
#     test = triples_factory.from_path(dstf['test']['graph'], create_inverse_triples=False,
#                                         entity_to_id=id_mappings['test']['e2id'], relation_to_id=id_mappings['test']['r2id'])

#     return train, validate, test

# def get_factories(dataset, pct, id_mappings=None):

#     triples_factory = TriplesFactory

#     dstf = find_dataset_betae(dataset, pct)
#     if id_mappings is None:
#         id_mappings = create_or_load_id_mappings(dstf)
#     train = triples_factory.from_path(dstf['train']['graph'], create_inverse_triples=False, 
#                                         entity_to_id=id_mappings['train']['e2id'], relation_to_id=id_mappings['train']['r2id'])
#     validate = triples_factory.from_path(dstf['validate']['graph'], create_inverse_triples=False,
#                                         entity_to_id=id_mappings['validate']['e2id'], relation_to_id=id_mappings['validate']['r2id'])
#     # pykeen is going to remap the indices for the training dataset, so pass this new map to test dataset
#     test = triples_factory.from_path(dstf['test']['graph'], create_inverse_triples=False,
#                                         entity_to_id=id_mappings['test']['e2id'], relation_to_id=id_mappings['test']['r2id'])

#     return train, validate, test

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