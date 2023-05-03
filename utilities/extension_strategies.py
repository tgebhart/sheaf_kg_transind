import torch

from torch import Tensor
from torch_geometric.typing import Adj, OptTensor

from scipy.sparse import csr_array
import numpy as np

from utilities.laplacian import normalized_adjacency, sheaf_diffusion_iter


class FeaturePropagation(torch.nn.Module):
    def __init__(self, num_iterations: int):
        super(FeaturePropagation, self).__init__()
        self.num_iterations = num_iterations

    def propagate(self, x: Tensor, mask: Tensor, propagation_mat) -> Tensor:
        # out is inizialized to 0 for missing values. However, its initialization does not matter for the final
        # value at convergence
        out = x
        if mask is not None:
            out = torch.zeros_like(x)
            out[mask] = x[mask]

        for _ in range(self.num_iterations):
            # Diffuse current features
            out = torch.sparse.mm(propagation_mat, out)
            # Reset original known features
            out[mask] = x[mask]

        return out

def random_filling(X):
    return torch.randn_like(X)

def zero_filling(X):
    return torch.zeros_like(X)

def mean_filling(X, feature_mask):
    n_nodes = X.shape[0]
    return compute_mean(X, feature_mask).repeat(n_nodes, 1)

def neighborhood_mean_filling(edge_index, X, feature_mask):
    n_nodes = X.shape[0]
    X_zero_filled = X
    X_zero_filled[~feature_mask] = 0.0
    edge_values = torch.ones(edge_index.shape[1]).to(edge_index.device)
    edge_index_mm = torch.stack([edge_index[1], edge_index[0]]).to(edge_index.device)

    D = torch.sparse.spmm(edge_index_mm, edge_values, n_nodes, n_nodes, feature_mask.float())
    mean_neighborhood_features = torch.sparse.spmm(edge_index_mm, edge_values, n_nodes, n_nodes, X_zero_filled) / D
    # If a feature is not present on any neighbor, set it to 0
    mean_neighborhood_features[mean_neighborhood_features.isnan()] = 0

    return mean_neighborhood_features


def feature_propagation(edge_index, X, Y, feature_mask, num_iterations, sheaf : bool):
    """
    If sheaf is False, then the extension will be done with respect to the graph laplacian. Otherwise, we need to train a sheaf laplacian and then use that.
    """
    propagation_model = FeaturePropagation(num_iterations=num_iterations)

    n_nodes, stalk_dim = X.shape
    if sheaf == False:
        edge_index, edge_weight = normalized_adjacency(edge_index, n_nodes=n_nodes)

        propagation_mat = torch.sparse.FloatTensor(edge_index, values=edge_weight, size=(n_nodes, n_nodes)).to(edge_index.device)
    else:
        edge_index, edge_blocks = sheaf_diffusion_iter(X, Y, edge_index, n_nodes=n_nodes) #First edge labels are 2-->6

        #At first I created a sparse_bsr_tensor, but this poorly supported. In particular, tensor.sparse.mm() is not supported on lots of CPU's or something. I got it working by working on a linux research server, rather than my mac. Still, it may be worth changing this..

        n_edges = edge_index.shape[1]
        crow_indices = torch.tensor(csr_array((torch.ones(n_edges), (edge_index[0], edge_index[1])), shape = (n_nodes, n_nodes)).indptr) #This is a hacky way to create the compressed row index format....
        col_indices = edge_index[1]
        propagation_mat = torch.sparse_bsr_tensor(crow_indices, col_indices, edge_blocks, size = (n_nodes*2, n_nodes*2), device = edge_index.device)

        #Oh yeah, and I forgot to implement I-Delta in the previous step.

        propagation_mat = torch.sparse.addmm(input=torch.eye(n_nodes*2), mat1 = propagation_mat, mat2 = torch.eye(n_nodes*2), alpha=-1)

        X = X.reshape((-1,1)) # Change a [n_nodes, num_features] to [n_nodes*n_features,1] so I can multiply it by propagation_mat
        #I'll fix the shape later.
        feature_mask = feature_mask.reshape((-1,1))
        
        #My idea to change this from sparse_bsr_tensor is to use the scipy.sparse library to create a sparse block matrix and then convert over to the pytorch sparse_csr format. Maybe this is better supported?

    return propagation_model.propagate(x=X, mask=feature_mask, propagation_mat=propagation_mat)


def filling(filling_method, edge_index, X, Y, feature_mask, num_iterations=None):
    if filling_method == "random":
        X_reconstructed = random_filling(X)
    elif filling_method == "zero":
        X_reconstructed = zero_filling(X)
    elif filling_method == "mean":
        X_reconstructed = mean_filling(X, feature_mask)
    elif filling_method == "neighborhood_mean":
        X_reconstructed = neighborhood_mean_filling(edge_index, X, feature_mask)
    elif filling_method == "sheaf_propagation":
        X_reconstructed = feature_propagation(edge_index, X, Y, feature_mask, num_iterations, sheaf = True)
        X_reconstructed = X_reconstructed.reshape(X.shape)
    elif filling_method == "constant_propagation":
        X_reconstructed = feature_propagation(edge_index, X, Y, feature_mask, num_iterations, sheaf = False)
    else:
        raise ValueError(f"{filling_method} method not implemented")
    return X_reconstructed


def compute_mean(X, feature_mask):
    X_zero_filled = X
    X_zero_filled[~feature_mask] = 0.0
    num_of_non_zero = torch.count_nonzero(feature_mask, dim=0)
    mean_features = torch.sum(X_zero_filled, axis=0) / num_of_non_zero
    # If a feature is not present on any node, set it to 0
    mean_features[mean_features.isnan()] = 0

    return mean_features