import torch
from torch.optim import Adam
from math import comb

from torch_scatter import scatter

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

    def __init__(self, n_nodes, edge_index, entity_types, stalk_dim: int, restriction_maps = None, verbose = False):
        super().__init__()
        self.edge_index = edge_index
        self.n_edges = edge_index.shape[1]
        self.entity_types = entity_types
        self.n_entity_types = len(torch.unique(entity_types))
        self.num_restrictions = 2*comb(len(self.entity_types), 2) # This is the number of restriction maps on the complete graph formed by each unique entity type. I think in practice, I will include self loops just for ease of implementation. 
        # E.g. if there are 3 entity types, then there are 6 = 2*(3 choose 2) restriction maps:
        # (1,2), (1,3)
        # (2,1), (2,3)
        # (3,1), (3,2)
        self.stalk_dim = stalk_dim
        self.n_nodes = n_nodes
        self.inv_node_degs = self._get_degs() #This is a [n_nodes, 1] tensor that gives the degree^{-1/2} of each node.

        self.verbose = verbose # Will print out progress statements if True.

        if restriction_maps != None:
            self.restriction_maps = restriction_maps
            self.register_buffer("restriction_maps",self.restriction_maps)

    def _get_degs(self):
        # This will break if the edge_index doesn't see every node.
        edge_weight = torch.ones((self.edge_index.size(1),), device=self.edge_index.device)
        row, col = self.edge_index[0], self.edge_index[1]
        deg = scatter(src=edge_weight, index=col, dim=0)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0)
        return deg_inv_sqrt

    def sheaf_dirichlet_energy(self, entity_reps):
        """
        This method computes the sheaf Dirichlet energy of the current sheaf. Minimizing this with respect to a fixed set of entity representations will find a good sheaf, and minimizing this with respect to a good sheaf will find a good set of entity representations.
        """

        # This goes ahead and computes D_v^{-1/2}*x_v ahead of time for all x_v.
        normalized_entity_reps =  torch.matmul(entity_reps, torch.diag(self.inv_node_degs))

        # This is the same as edge_index, but with the entity types instead of the node indices.
        labeled_edge_index = torch.cat((self.entity_types[self.edge_index[0,:]],self.entity_types[self.edge_index[1,:]]),dim=0).reshape(2,-1)

        # What I want to do is to use scatter as much as I can. What I expect in large graphs is that there are much less edge types (unique relationships between different node_types) than there are edges overall. I want to compute all of the same edge_types at the same time using scatter. 

        # If edge_index is of size [2, num_edges], then the following tensors are [stalk_dim, stalk_dim, num_edges], storing the restriction map for the head nodes and tail nodes for each edge.
        head_maps = self.restriction_maps[labeled_edge_index[0],labeled_edge_index[1]].reshape(self.stalk_dim,self.stalk_dim,-1)
        tail_maps = self.restriction_maps[labeled_edge_index[1], labeled_edge_index[0]].reshape(self.stalk_dim,self.stalk_dim,-1)

        # This uses the restriction maps we computed before to embed all the entity representations. There HAS to be a faster way to do this, utilizing the broadcasting from pytorch.
        head_embeddings = torch.zeros(self.stalk_dim,self.n_edges)
        tail_embeddings = torch.zeros(self.stalk_dim,self.n_edges)
        for i in range(self.n_edges):
            head_embeddings[:, i] = head_maps[:,:,i] @ normalized_entity_reps[:,self.edge_index[0,i]]
            tail_embeddings[:, i] = tail_maps[:,:,i] @ normalized_entity_reps[:,self.edge_index[1,i]]

        # Now, we can use scatter to take all these comparisons at once.

        comparison_vec = torch.zeros(self.stalk_dim,self.n_nodes)

        scatter(head_embeddings, index = self.edge_index[0,:] ,dim=1, out = comparison_vec)
        scatter(-tail_embeddings, self.edge_index[1,:], dim=1, out=comparison_vec)
        
        return torch.sum(torch.norm(comparison_vec, dim=1).pow(2))


    def loss(self, entity_reps):
        """
        This computes our loss. Its basically the sheaf dirichlet energy, but I want to make sure that we penalize restriction maps that have low norm to avoid minimizing to the zero maps.

        entity_reps: This is a [n_nodes, feature_dim] tensor that assigns each node to its feature vector
        """

        return self.sheaf_dirichlet_energy(entity_reps) + 1/torch.sum(torch.norm(self.restriction_maps).pow(2))

    def train_maps(self, entity_reps, epochs = 1, lr = 0.1):
        """
        This method finds restriction maps that finds optimal restriction maps for the sheaf Laplacian, given a choice of entity representation.

        I guess later I could take out the dependence on entity representations.
        """

        self.restriction_maps = torch.nn.Parameter(torch.randn((self.n_entity_types, self.n_entity_types,self.stalk_dim, self.stalk_dim))/self.stalk_dim) #this normalization means with high probability restriction map norms are <= 1.
        # This also encodes self loops, but I don't use these.... I just don't know the best way to index through all combinations of entity types.

        optimizer = Adam(self.parameters(), lr = lr)

        for epoch in range(epochs):
            loss = self.loss(entity_reps)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if self.verbose:
                print('Epoch {}, sde: {}'.format(epoch+1, loss))

    def get_laplacian(self):
        """
        Use the restriction maps to get the laplacian.
        """
        print('Computing Laplacian...')


def learn_sheaf_laplacian(X, Y, edge_index):
    """
    Inputs
    --------
    edge_index: Graph connectivity in COO format with shape [2, num_edges] and type torch.long
    X: This is a [n_nodes, feature_dim] tensor that assigns each node to its feature vector
    Y: This is a [n_nodes, 1] tensor that assigns each node to its class

    Output
    -------
    """

    n_nodes, stalk_dim = X.shape
    sheaf = KnowledgeSheaf(n_nodes, edge_index, Y, stalk_dim = stalk_dim, verbose = True)
    sheaf.train_maps(torch.transpose(X, 0, 1), epochs = 1, lr = 0.1)
    print('Found the maps!')
