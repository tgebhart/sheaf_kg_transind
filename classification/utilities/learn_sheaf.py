import torch
from torch.optim import Adam
from math import comb
from tqdm import tqdm

from torch_scatter import scatter

import numpy as np
from scipy.sparse import bsr_matrix, csr_array
from torch_householder import torch_householder_orgqr


class KnowledgeSheaf(torch.nn.Module):
    """
    A class for representing a knowledge sheaf on a graph. I will write more once I understand what I want out of this class.

    Attributes
    ----------
    edge_index: Graph connectivity in COO format with shape [2, num_edges] and type torch.long
    entity_types: This is a [n_nodes, 1] tensor that assigns each node to its class
    entity_reps: This is a [n_nodes, feature_dim] tensor that assigns each node to its feature vector
    stalk_dim: The dimension of the stalks of the sheaf
    """

    def __init__(
        self,
        n_nodes,
        edge_index,
        entity_types,
        stalk_dim: int,
        restriction_maps=None,
        orthogonal=True,
        verbose=False,
        device=None,
    ):
        super().__init__()
        self.edge_index = edge_index
        self.n_edges = edge_index.shape[1]
        self.entity_types = entity_types
        self.n_entity_types = len(torch.unique(entity_types))
        self.num_restrictions = 2 * comb(
            len(self.entity_types), 2
        )  # This is the number of restriction maps on the complete graph formed by each unique entity type. I think in practice, I will include self loops just for ease of implementation.
        # E.g. if there are 3 entity types, then there are 6 = 2*(3 choose 2) restriction maps:
        # (1,2), (1,3)
        # (2,1), (2,3)
        # (3,1), (3,2)
        # Wait, I might be being stupid here. I'm not allowing for multiple different relation types between two entity types. E.g. (1)-->(2) could be "friends" or "enemies". So far, all the datasets I'm using are not inherently knowledge graphs, so I'm CREATING edge types from unique combinatiosn
        self.stalk_dim = stalk_dim
        self.orthogonal = orthogonal
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
        # This also breaks if we have multiple edges between nodes, since we use
        # torch.ones. Duh. 
        weights = torch.ones(
            (self.edge_index.size(1),), device=self.edge_index.device
        )
        dense_adj_matrix = torch.sparse_coo_tensor(
            indices = self.edge_index, 
          values = torch.ones(self.n_edges)
        ).to_dense()
        output_indices = torch.zeros(3, dtype=torch.int64)
        deg = scatter(src=dense_adj_matrix, index=output_indices, dim=0)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0)
        return deg_inv_sqrt

    def sheaf_dirichlet_energy(self, entity_reps):
        """
        This method computes the sheaf Dirichlet energy of the current sheaf. Minimizing this with respect to a fixed set of entity representations will find a good sheaf, and minimizing this with respect to a good sheaf will find a good set of entity representations.
        """

        # This goes ahead and computes the rescaled inputs D_v^{-1/2}*x_v ahead of time for all x_v.
        normalized_entity_reps = torch.matmul(
            entity_reps, torch.diag(self.inv_node_degs)
        )

        # Here is where I'm rescaling the restriction maps to make them norm 1.
        norms = self.map_norms().reshape(-1, 1, 1)
        maps = self.restriction_maps.reshape(-1, 2, 2)
        normalized_maps = (maps / norms).reshape(10, 10, 2, 2)

        # This is the same as edge_index, but with the entity types instead of the node indices. This helps me map the restriction maps to the nodes in the next step.
        labeled_edge_index = torch.cat(
            (
                self.entity_types[self.edge_index[0, :]],
                self.entity_types[self.edge_index[1, :]],
            ),
            dim=0,
        ).reshape(2, -1)

        # What I want to do is to use scatter as much as I can. What I expect in large graphs is that there are much less edge types (unique relationships between different node_types) than there are edges overall. I want to compute all of the same edge_types at the same time using scatter.

        # If edge_index is of size [2, num_edges], then the following tensors are [n_edges, stalk_dim, stalk_dim], storing the restriction map for the head nodes and tail nodes for each edge in edge_index. self.restriction_maps[1,2] = R_{1-->2}
        head_maps = normalized_maps[
            labeled_edge_index[0], labeled_edge_index[1]
        ].reshape(-1, self.stalk_dim, self.stalk_dim)
        tail_maps = normalized_maps[
            labeled_edge_index[1], labeled_edge_index[0]
        ].reshape(-1, self.stalk_dim, self.stalk_dim)

        normalized_head_entities = normalized_entity_reps[:, self.edge_index[0, :]]
        normalized_tail_entities = normalized_entity_reps[:, self.edge_index[1, :]]

        head_embeddings = torch.einsum(
            "nbh,bn->hn", (head_maps, normalized_head_entities)
        )
        tail_embeddings = torch.einsum(
            "nbh,bn->hn", (tail_maps, normalized_tail_entities)
        )

        # Now, we can use scatter to take all these comparisons at once.
        # Revelation: The following is ALWAYS computing sheaf energy = 0! I think this was conceptually wrong to do...? I think what is wrong here is that I'm collecting all the head embeddings and tail embeddings into a SINGLE edge space for each node, and then computing a difference here. I should not do this.

        # comparison_vec = torch.zeros(self.stalk_dim,self.n_nodes)

        # scatter(head_embeddings, index = self.edge_index[0,:] ,dim=1, out=comparison_vec)
        # scatter(-tail_embeddings, self.edge_index[1,:], dim=1, out=comparison_vec)

        # if self.verbose:
        #    print('Computed the sheaf dirichlet energy...')

        return torch.sum(torch.norm(head_embeddings - tail_embeddings, dim=0).pow(2))

    def loss(self, entity_reps):
        """
        This computes our loss. Its basically the sheaf dirichlet energy, but I want to make sure that we penalize restriction maps that have low norm to avoid minimizing to the zero maps.

        entity_reps: This is a [n_nodes, feature_dim] tensor that assigns each node to its feature vector
        """

        sheaf_energy = self.sheaf_dirichlet_energy(entity_reps)

        if self.orthogonal:
            penalty = 0
        else:
            penalty = torch.sum(
                (1 / torch.linalg.matrix_norm(self.restriction_maps)).pow(2)
            )

        # penalty = torch.sum((1/torch.linalg.matrix_norm(self.restriction_maps)).pow(2))

        loss = sheaf_energy + penalty

        return loss, sheaf_energy, penalty

        # TODO: Check the following:
        # Switch to "clamping". M --> lambda*M. norm( lambda M ) = lambda*norm(M)
        # Make them orthonormal the whole time.
        # Run the constant sheaf, recover their results.

    def map_norms(self):
        # This scales all the restriction maps to be matrix norm 1.

        return torch.linalg.matrix_norm(self.restriction_maps)

    def train_maps(self, entity_reps, epochs=1, lr=0.1):
        """
        This method finds restriction maps that finds optimal restriction maps for the sheaf Laplacian, given a choice of entity representation.

        I guess later I could take out the dependence on entity representations.
        """

        if self.verbose:
            print("Starting to train maps...")

        if self.orthogonal:
            ortho_maps = torch.tensor(
                [
                    torch_householder_orgqr(
                        torch.rand(self.stalk_dim, self.stalk_dim).tril(diagonal=-1)
                        + torch.eye(self.stalk_dim, self.stalk_dim)
                    ).tolist()
                    for _ in range(self.n_entity_types * self.n_entity_types)
                ]
            ).reshape(
                self.n_entity_types, self.n_entity_types, self.stalk_dim, self.stalk_dim
            )

            self.restriction_maps = torch.nn.Parameter(ortho_maps).to(self.device)

            # I have not guaranteed that the norm is 1, I really want to parametrize the special orthogonal group. This isn't implemented anywhere and seems difficult to cook up, so I've multiplied by something to make sure the sheaf energy is always done with respect to the normalized maps.
        else:
            self.restriction_maps = torch.nn.Parameter(
                torch.randn(
                    (
                        self.n_entity_types,
                        self.n_entity_types,
                        self.stalk_dim,
                        self.stalk_dim,
                    )
                ).to(self.device)
                / self.stalk_dim
            )  # this normalization means with high probability restriction map norms are <= 1.
            # [head entity type, tail entity type, :, :] would give the restriction map along the edge head -- > tail.
            # This also encodes self loops, but I don't use these.... I just don't know the best way to index through all combinations of entity types.

        optimizer = Adam(self.parameters(), lr=lr)

        pbar = tqdm(range(epochs), desc="train_maps")
        for epoch in pbar:
            loss, sheaf_energy, penalty = self.loss(entity_reps)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(
                {
                    "sheaf energy": sheaf_energy.detach().cpu().numpy(),
                    "trivial penalty": penalty.detach().cpu().numpy(),
                }
            )

            # test_laplacian = self.restriction_maps[6,2].transpose(0,1) @ self.restriction_maps[2,6]

            # print("laplacian for first map: {}".format(test_laplacian))
            # print("laplacian matrix norm for first map: {}".format(torch.linalg.matrix_norm(test_laplacian)))
            if self.verbose:
                print(
                    "Epoch {}/{}, sheaf energy: {}, trivial penalty: {}".format(
                        epoch + 1, epochs, sheaf_energy, penalty
                    )
                )

    def normalized_laplacian(self, restriction_maps=None):
        """
        Use the restriction maps to get the laplacian.
        If each restriction map is [stalk_dim, stalk_dim] then we should expect the laplacian to be [stalk_dim*n_node, stalk_dim*n_nodes]. Its a block matrix whose blocks depends on the entity types of the edge it represents.

        I don't think I want to represent this as an actual big matrix, but instead just return the blocks on demand. Maybe it could return edge_index, edge_block. Then I'm not saving a bunch of empty blocks.
        """

        if restriction_maps != None:
            self.restriction_maps = restriction_maps

        norms = self.map_norms().reshape(-1, 1, 1)
        maps = self.restriction_maps.reshape(-1, 2, 2)
        normalized_maps = (maps / norms).reshape(10, 10, 2, 2)
        # normalized_maps = self.restriction_maps

        # This is the same as edge_index, but with the entity types instead of the node indices. This helps me map the restriction maps to the nodes in the next step.
        labeled_edge_index = torch.cat(
            (
                self.entity_types[self.edge_index[0, :]],
                self.entity_types[self.edge_index[1, :]],
            ),
            dim=0,
        ).reshape(2, -1)

        # If edge_index is of size [2, num_edges], then the following tensors are [stalk_dim, stalk_dim, num_edges], storing the restriction map for the head nodes and tail nodes for each edge in edge_index.
        head_maps = normalized_maps[
            labeled_edge_index[0], labeled_edge_index[1]
        ].reshape(-1, self.stalk_dim, self.stalk_dim)
        tail_maps = normalized_maps[
            labeled_edge_index[1], labeled_edge_index[0]
        ].reshape(-1, self.stalk_dim, self.stalk_dim)

        normalized_laplacian = torch.einsum(
            "nbh,nhk->nbk", (tail_maps.transpose(1, 2), head_maps)
        )

        # edge_index, edge_block
        # normalized_laplacian[i, :, :] = D_t^{-1/2} R_{t-->e}^T R_{h-->e} D_h^{-1/2}

        if self.verbose:
            print("Found the laplacian!")

        return self.edge_index, normalized_laplacian


def learn_sheaf_laplacian(X, Y, edge_index):
    """
    Inputs
    --------
    edge_index: Graph connectivity in COO format with shape [2, num_edges] and type torch.long
    X: This is a [n_nodes, feature_dim] tensor that assigns each node to its feature vector
    Y: This is a [n_nodes, 1] tensor that assigns each node to its class

    Output  (u,v) R_{v-->e}^T R_{u-->e}
    -------
    """

    # print(torch.nonzero(torch.isnan(X[:,0])).shape[0]) This is the number of masked elements

    n_nodes, stalk_dim = X.shape

    # constant_sheaf = KnowledgeSheaf(n_nodes, edge_index, Y, stalk_dim = stalk_dim, verbose = True)
    # constant_maps = torch.zeros(size = (10,10,stalk_dim, stalk_dim))
    # constant_maps[:,:,0,0] = 1
    # constant_maps[:,:,1,1] = 1
    # edge_index, Delta = constant_sheaf.normalized_laplacian(restriction_maps = constant_maps)

    sheaf = KnowledgeSheaf(n_nodes, edge_index, Y, stalk_dim=stalk_dim, verbose=True)
    sheaf.train_maps(torch.transpose(X, 0, 1), epochs=400, lr=0.1)
    edge_index, Delta = sheaf.normalized_laplacian()

    return edge_index, Delta
