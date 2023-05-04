import torch

import torch_geometric.utils

from utilities.learn_sheaf import learn_sheaf_laplacian

from torch_scatter import scatter


def normalized_adjacency(edge_index, n_nodes):
    """
    Given an edge_index, return the same edge_index and edge weights computed as
    \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}} \mathbf{\hat{D}}^{-1/2}.
    This is almost the matrix needed in the iterative scheme, except that I need to reset the known features back to their original values after each iteration. This is done in the propagate function.
    """
    edge_weight = torch.ones((edge_index.size(1),), device=edge_index.device)
    row, col = edge_index[0], edge_index[1]
    deg = scatter(edge_weight, col, dim=0, dim_size=n_nodes)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0)
    DAD = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    return edge_index, DAD

def sheaf_diffusion_iter(X, Y, edge_index, n_nodes):  
    """
    EXPLANATION
    Given an edge_index, return the same edge_index and some edge weights computed as
    the matrix I - \Slap.
    This is almost the matrix needed in the iterative scheme:
    \begin{bmatrix}
    \matmat{I} & \mat{0}\\
    -\Slap[U,B] & \matmat{I}-\Slap[U,U]
    \end{bmatrix} 
    except that I need to reset the known features back to their original values after each iteration. This is done in the propagate method.

    INPUTS
    X: This is a [n_nodes, feature_dim] tensor
    Y: This is a [n_nodes, 1] tensor that assigns each node to its class
    edge_index: Graph connectivity in COO format with shape [2, num_edges] and type torch.long
    n_nodes: Number of nodes in the graph

    OUTPUTS
    Eventually, I need to return a 
    edge_index, edge_weights
    representing a Laplacian so that really efficient computations can be done with it.
    """
    
    edge_index, sheaf_prop_mat = learn_sheaf_laplacian(X, Y, edge_index)

    return edge_index, sheaf_prop_mat