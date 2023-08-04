from learn_sheaf_mine import KnowledgeSheaf
import torch
from scipy.sparse import coo_matrix

# @fixture
# def sheaf(): 
#     # Graph is length 1 path, super simple
#     # a < -- > b 
#     return KnowledgeSheaf(
#         n_nodes=2, 
#         edge_index=coo_matrix([[0,1], [1,0]])
#     )

# def test_degree_inv():
def test_coo():
    coo = coo_matrix([[0,1], [1,0]])
    num_edges = 2
    print(coo)
    assert coo.shape == (2,2)

def test_size(): 
    ones = torch.ones(2,3)
    assert ones.size(1) == 3
    assert ones.size(0) == 2

