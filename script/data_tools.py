import os
import pickle
import json

import numpy as np
import pandas as pd
import torch
torch.manual_seed(0)
from pykeen.triples import TriplesFactory
from pykeen.datasets.inductive import InductiveFB15k237, InductiveWN18RR, InductiveNELL

from complex_data_info import QUERY_STRUCTURES, name_query_dict

BASE_DATA_PATH = 'data'
BASE_CONFIG_PATH = 'config/ablation'

def find_dataset_betae(dataset, pct):
    basepath = f'{BASE_DATA_PATH}/{dataset}/{pct}'
    join = os.path.join
    return {
            'train': {
                'graph': join(basepath, 'train_graph.txt'), 
                'queries': join(basepath, 'train_queries.pkl'),
                'answers': {
                    'easy': join(basepath, 'train_answers_valid.pkl'),
                    'hard': join(basepath, 'train_answers_hard.pkl'),
                },
            },
            'valid': { 
                'graph': join(basepath,'val_inference.txt'), 
                'queries': join(basepath,'valid_queries.pkl'),
                'answers': {
                    'easy': join(basepath,'valid_answers_easy.pkl'),
                    'hard': join(basepath,'valid_answers_hard.pkl'),
                },
                'triplets': join(basepath,'val_predict.txt'),
            },
            'test': {
                'graph': join(basepath,'test_inference.txt'),
                'queries': join(basepath,'test_queries.pkl'),
                'answers': {
                    'easy': join(basepath,'test_answers_easy.pkl'),
                    'hard': join(basepath,'test_answers_hard.pkl'),
                },
                'triplets': join(basepath,'test_predict.txt'),
            },
            'id_mappings': join(basepath,'id_mappings.pkl')
    }

def create_relation_id_mapping(dstf, relation_col=1, delimiter='\t'):

    print('creating id maps...')
    train = pd.read_csv(dstf['train']['graph'], delimiter=delimiter, header=None)
    valid = pd.read_csv(dstf['valid']['graph'], delimiter=delimiter, header=None)
    test = pd.read_csv(dstf['test']['graph'], delimiter=delimiter, header=None)

    relations = np.unique(np.concatenate([train[relation_col].unique(), valid[relation_col].unique(), test[relation_col].unique()]))
    
    return {str(r):ix for ix, r in enumerate(relations)}
    

def get_graphs(dataset, pct, delimiter='\t'):

    triples_factory = TriplesFactory

    dstf = find_dataset_betae(dataset, pct)
    r2id = create_relation_id_mapping(dstf)

    train_df = pd.read_csv(dstf['train']['graph'], delimiter=delimiter, header=None)
    valid_df = pd.read_csv(dstf['valid']['graph'], delimiter=delimiter, header=None)
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
    validate = triples_factory.from_path(dstf['valid']['triplets'], create_inverse_triples=False,
                                        entity_to_id=validate_graph.entity_to_id, relation_to_id=validate_graph.relation_to_id)
    # pykeen is going to remap the indices for the training dataset, so pass this new map to test dataset
    test = triples_factory.from_path(dstf['test']['triplets'], create_inverse_triples=False,
                                        entity_to_id=test_graph.entity_to_id, relation_to_id=test_graph.relation_to_id)

    return train, validate, test

def graph_entity_inclusion_map(subgraph_tf, graph_tf):
    subgraph_entity_id_to_label = subgraph_tf.entity_id_to_label
    graph_label_to_entity_id = graph_tf.entity_to_id
    return {k:graph_label_to_entity_id[v] for k,v in subgraph_entity_id_to_label.items() if v in graph_label_to_entity_id}

def graph_relation_inclusion_map(subgraph_tf, graph_tf):
    subgraph_relation_id_to_label = subgraph_tf.relation_id_to_label
    graph_label_to_relation_id = graph_tf.relation_to_id
    # relations should always be the same
    return {k:graph_label_to_relation_id[v] for k,v in subgraph_relation_id_to_label.items() if v in graph_label_to_relation_id}

def tensorize_p(q):
    s = torch.tensor([q[0]])
    rs = torch.tensor(list(q[1]))
    return {'sources': s, 'relations': rs}

def tensorize_i(q):
    s = torch.tensor([t[0] for t in q])
    rs = torch.tensor([t[1][0] for t in q])
    return {'sources': s, 'relations': rs}
    
def tensorize_ip(q):
    s = torch.tensor([q[0][0][0], q[0][1][0]])
    rs = torch.tensor([q[0][0][1][0], q[0][1][1][0], q[1][0]])
    return {'sources': s, 'relations': rs}

def tensorize_pi(q):
    s = torch.tensor([q[0][0], q[1][0]])
    rs = torch.tensor(list(q[0][1]) + [q[1][1][0]])
    return {'sources': s, 'relations': rs}

def tensorize(q, query_structure):
    if query_structure in ['1p', '2p', '3p']:
        t = tensorize_p(q)
    elif query_structure in ['2i', '3i']:
        t = tensorize_i(q)
    elif query_structure == 'ip':
        t = tensorize_ip(q)
    elif query_structure == 'pi':
        t = tensorize_pi(q)
    else:
        raise ValueError(f'query structure {query_structure} not implemented')
    t['structure'] = query_structure
    return t

def generate_mapped_triples_both(query_loc_hard, answer_loc_easy, answer_loc_hard, 
                                query_structures=QUERY_STRUCTURES, filter_fun=None, remap_fun=None,
                                skip_ea=False):

    with open(answer_loc_easy, 'rb') as f:
        easy_answers = pickle.load(f)
    with open(query_loc_hard, 'rb') as f:
        hard_queries = pickle.load(f)
    with open(answer_loc_hard, 'rb') as f:
        hard_answers = pickle.load(f)

    mapped_triples = {}
    for query_structure in query_structures:
        print(f'loading query structure {query_structure}')
        qs = hard_queries[name_query_dict[query_structure]] # same for easy and hard queries
        ea = easy_answers[name_query_dict[query_structure]]
        ha = hard_answers[name_query_dict[query_structure]]

        num_filtered = 0
        qlist = []
        for q in qs:
            qtens = tensorize(q, query_structure)
            easy_ans = list(ea[q]) if not skip_ea else []
            hard_ans = list(ha[q])
            num_hard = len(hard_ans)
            ans = hard_ans + easy_ans # combine easy and hard answers
            if remap_fun is not None:
                qtens, ans = remap_fun(qtens, ans)
            if filter_fun is not None:
                if not filter_fun(qtens, ans):
                    num_filtered += 1
                    continue
            qtens['hard'] = torch.LongTensor(ans[:num_hard])
            qtens['easy'] = torch.LongTensor(ans[num_hard:])
            qlist.append(qtens)
        if filter_fun is not None:
            print(f'filtered {num_filtered} queries of {len(qs)} possible for query {query_structure}')
        mapped_triples[query_structure] = qlist
    return mapped_triples

def load_queries_and_answers(dataset, pct, eval_graph_factory, eval_graph_name, 
                            query_structures=QUERY_STRUCTURES, skip_ea=False):

    dstf = find_dataset_betae(dataset, pct)

    e2id = {int(k):v for k,v in eval_graph_factory.entity_to_id.items()}
    r2id = {int(k):v for k,v in eval_graph_factory.relation_to_id.items()}
    def remap_fun(q, ans):
        rq = q.copy()
        rq['relations'] = q['relations'].apply_(r2id.get)
        rq['sources'] = q['sources'].apply_(e2id.get)
        rans = [e2id[a] for a in ans]
        return rq, rans
    
    test_queries = generate_mapped_triples_both(dstf[eval_graph_name]['queries'], 
                                                dstf[eval_graph_name]['answers']['easy'], dstf[eval_graph_name]['answers']['hard'],
                                                query_structures=query_structures, 
                                                remap_fun=remap_fun, skip_ea=skip_ea)
    return test_queries

def get_train_eval_inclusion_data(dataset, dataset_pct, orig_graph_type, eval_graph_type, 
                                  include_complex=False, query_structures=QUERY_STRUCTURES,
                                  skip_ea=False):
    print('loading factories and graphs...')
    train_graph, valid_graph, test_graph = get_graphs(dataset, dataset_pct)
    train_tf, valid_tf, test_tf = get_factories(dataset, dataset_pct)
    
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

    orig_graph, orig_triples = get_train_eval_sets(orig_graph_type)
    eval_graph, eval_triples = get_train_eval_sets(eval_graph_type)

    print('computing orig-->eval inclusion maps...')
    orig_eval_entity_inclusion = graph_entity_inclusion_map(orig_graph, eval_graph)
    orig_eval_relation_inclusion = graph_relation_inclusion_map(orig_graph, eval_graph)

    r = {'orig':{'graph':orig_graph, 'triples':orig_triples},
        'eval':{'graph':eval_graph, 'triples':eval_triples},
        'inclusion':{'entities':orig_eval_entity_inclusion, 'relations':orig_eval_relation_inclusion}}
    
    if include_complex:
        print('loading complex queries...')
        r['complex'] = load_queries_and_answers(dataset, dataset_pct, eval_graph, eval_graph_type,
                                                query_structures=query_structures, skip_ea=skip_ea)

    return r

def split_mapped_triples(triples_factory, train_pct=0.85):
    triples = triples_factory.mapped_triples
    ntrip = triples.size(0)
    perm = torch.randperm(ntrip)
    k = int(train_pct*ntrip)
    msk = torch.zeros(ntrip, dtype=torch.bool)
    idx = perm[:k]
    msk[idx] = True
    train_triples = triples[msk]
    eval_triples = triples[~msk]

    train = triples_factory.clone_and_exchange_triples(train_triples)
    eval = triples_factory.clone_and_exchange_triples(eval_triples)
    return train, eval

def load_hpo_config(hpo_config_name: str) -> dict:
    hpo_config_fname = hpo_config_name if '.json' in hpo_config_name else f'{hpo_config_name}.json'
    hpo_config_loc = os.path.join(os.path.join(BASE_CONFIG_PATH, hpo_config_fname))
    with open(hpo_config_loc, 'r') as f:
        config = json.load(f)
    return config

def get_model_name_from_config(hpo_config_name: str) -> str:
    if '.json' in hpo_config_name:
        hpo_config_name = hpo_config_name[:hpo_config_name.find('.json')]
    config = load_hpo_config(hpo_config_name)
    return config['pipeline']['model'].lower(), hpo_config_name

def get_disjoint_dataset(dataset, version, create_inverse_triples=False):

    if dataset == 'InductiveFB15k237':
        return InductiveFB15k237(version=version, create_inverse_triples=create_inverse_triples)
    if dataset == 'InductiveWN18RR':
        return InductiveWN18RR(version=version, create_inverse_triples=create_inverse_triples)
    if dataset == 'InductiveNELL':
        return InductiveNELL(version=version, create_inverse_triples=create_inverse_triples)
    
def get_eval_graph(dataset, eval_graph):
    if eval_graph == 'valid':
        return dataset.inductive_validation
    if eval_graph == 'test':
        return dataset.inductive_testing