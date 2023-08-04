import torch
from torch.optim import Adam
from tqdm import tqdm


import numpy as np


class KnowledgeSheaf(torch.nn.Module):
    """
    A class for representing a knowledge sheaf on a graph. I will write more once I understand what I want out of this class.

    Attributes
    ----------
    edge_index: Graph connectivity in COO format with shape [2, num_edges] and type torch.long
        EXAMPLE: graph with 1 undirected edge between 2 nodes. This has a 1s at (0,1) and (1,0), and 0 else.
        Then the COO format for this graph would be a collection of (i,j,value)s: [(0,1,1), (1,0,1)].
        torch.sparse_coo_tensor separates these like
        torch.sparse_coo_tensor(indices = [[0,1], [1,0]], values = [1,0]  size = (2, num_edges)). You 
        can transform this back into "normal" format by calling .to_dense() on it, which would return 
        [[0,1], [1,0]]. 
    entity_types: This is a [n_nodes, 1] tensor that assigns each node to its class
    entity_reps: This is a [n_nodes, feature_dim] tensor that assigns each node to its feature vector
    stalk_dim: The dimension of the stalks of the sheaf
    """

    def __init__(
        self,
        n_nodes: int,
        edge_index: torch.LongTensor,
        entity_types: torch.Tensor,
        stalk_dim: int,
        verbose: bool=True,
        restriction_maps=None,
        device=None,
    ):
        super().__init__()
        self.edge_index = edge_index
        self.n_edges = edge_index.shape[1]
        self.entity_types = entity_types
        self.n_entity_types = len(torch.unique(entity_types))
        self.stalk_dim = stalk_dim
        self.n_nodes = n_nodes
        self.inv_node_degs = (
            self._get_degs()
        )  # This is a [n_nodes, 1] tensor that gives the degree^{-1/2} of each node.

        self.verbose = verbose  # Will print out progress statements if True.
        self.device = device if device is not None else edge_index.device

        if restriction_maps is not None:
            self.restriction_maps = restriction_maps
            self.register_buffer("restriction_maps", self.restriction_maps)

    def _get_degs(self):
        # This will break if the edge_index doesn't see every node.
        edge_weight = torch.ones(
            (self.edge_index.size(1),), device=self.edge_index.device
        )
        _, col = self.edge_index[1]
        # deg = scatter(src=edge_weight, index=col, dim=0) #sums over  
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0)
        return deg_inv_sqrt
