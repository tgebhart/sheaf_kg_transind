import torch

from torch_scatter import scatter

# This is a copy of the file Tom sent me. So far, I'm referencing it, but not using it.

# How should initialization work? Ideally, we would want whatever oprerator is applied to have operator norm close to 1. The norm of the Laplacian is at most the square of the norm of the coboundary. 
# In expectation, this is going to be something like (proportional to?) the average degree times the norm of the restriction maps. So we want the norms of the restriction maps to be 1/d_avg... or something like that.
# At the very least, we can make sure the restriction maps have norm at most 1 when initialized. so we probably need a 1/sqrt(n) factor or something like that

# Initialization: We can guarantee that the norm of $I - \alpha L$ is at most 1 by making each restriction map norm 1 (we can just set this by fiat, normalizing) and setting \alpha = 1/dmax. 
# we could also use a left-normalized version of L, dividing each block row by the degree of the corresponding node in the underlying graph. 
# for restriction maps to be norm 1, we also need the edge features to be normalized

# what does this initialization do for the edge feature gradients? The equivalent of the Laplacian here is the Gram matrix with entries <\delta_i x_e, \delta_j x_e>. for a large graph and a normalized x, these entries will be small, especially if we have normalized restriction maps to norm 1. 
# but if we normalize the feature vector for each node individually, the entries of this matrix will be bounded above by 4 (since we have two unit norm matrices and two unit norm vectors. maybe this is 8? 8 is definitely a bound) this says that the norm of the Gram matrix is bounded above by 4 * N_edge_features.
# so \alpha = 1/4 N_edge_features should work, assuming that the normalization of restriction maps and node features holds. 
class SheafLayer(torch.nn.Module):
    """
    A layer representing a cellular sheaf on an oriented graph. The restriction maps on each edge are determined by input edge attributes by taking a linear combination of basis maps.
    Can compute the coboundary, boundary, Laplacian, and an Euler-discretized Laplacian.
    Can also compute the gradient with respect to edge attributes of \|\delta x\|^2, which generates a sort of reversed diffusion process.
    Attributes
    ----------
    stalk_dim : dimension of the vertex (and edge) stalks of the sheaf. Each node feature has this dimension.
    n_edge_attr : number of edge attributes. This determines the number of basic restriction maps stored by the sheaf.
    matrix_weighted : the two restriction maps for each edge are equal. If this is true, the orientation of the graph does not matter.
    signtype : if None, computes the normal sheaf coboundary and Laplacian. 
              Otherwise, adds an additional set of weights parameterized by edge attributes. This can affect the operations in two ways:
              if 'sheaf', one restriction map on each edge is multiplied by the weight, while the other is multiplied by its absolute value. This is particularly useful for matrix-weighted sheaves, as it adds additional expressivity without requiring an orientation
              if 'gradient', the Laplacian is computed by multiplying the copmonent of \delta x on each edge by the signed weight before multiplying by \delta^*
    signdiag : controls whether the signed edge weights are scalars (one per edge) or vectors (one weight per stalk dimension per edge).
    train_restriction_maps : if True, the basic restriction maps are treated as parameters and are trained when the layer is placed in a network. If False, they are not modified during training and this layer is purely functional.
    euler_stepsize : the default step size for the Euler discretized diffusion operations
    layertype : the default operation performed by the layer when the method forward() is called.
    signtype and euler_stepsize may be changed after construction, but other attributes may not.
    """
    # should we refactor this into something functional that just takes inputs and has no parameters? then keep the parameters elsewhere? that may be cleaner but it may also be slower
    # TODO: implement affine sheaves
    # TODO: add edge stalk dimension
    def __init__(self, stalk_dim: int, n_edge_attr: int, matrix_weighted=False, signtype: str=None, signdiag: bool=False, layertype: str='Laplacian', euler_stepsize: float=0.1, train_restriction_maps: bool=True, degree_normalize: bool=False, adaptive_stepsize: bool=False, **kwargs) -> None:
        super(SheafLayer,self).__init__()

        self.stalk_dim = stalk_dim
        self.n_edge_features = n_edge_attr
        self.matrix_weighted = matrix_weighted
        self.signed = False
        self.signtype = signtype
        self.layertype = layertype
        self.euler_stepsize_cochain = euler_stepsize
        #self.euler_stepsize_gradient = 0.01*euler_stepsize
        self.euler_stepsize_gradient = 1/(4*self.n_edge_features)
        self.degree_normalize = degree_normalize
        self.adaptive_stepsize = adaptive_stepsize

        # initialize the restriction maps and sign parameters
        if self.matrix_weighted:
            restriction_maps = torch.randn((n_edge_attr,stalk_dim,stalk_dim))/stalk_dim #this normalization means with high probability restriction map norms are <= 1.
        else:
            restriction_maps = torch.randn((n_edge_attr,2,stalk_dim,stalk_dim))/stalk_dim
        if self.signtype is not None:
            self.signed = True
            if signdiag:
                signs = torch.ones(n_edge_attr,stalk_dim,1)
            else:
                signs = torch.ones(n_edge_attr,1,1)
        
        # register as parameters or buffers depending on train_restriction_maps
        if train_restriction_maps:
            self.restriction_maps = torch.nn.Parameter(restriction_maps)
            if self.signed:
                self.signs = torch.nn.Parameter(signs)
        else:
            self.restriction_maps = restriction_maps
            self.register_buffer("restriction_maps",self.restriction_maps)
            if self.signed:
                self.signs = signs
                self.register_buffer("signed edge weights",self.signs)        

    def _restriction_maps(self,edge_attr: torch.Tensor) -> torch.FloatTensor:
        # if we wanted a fancier method of choosing these maps, we could replace this in a subclass
        if edge_attr.dtype == torch.float:
            if self.matrix_weighted:
                F = (edge_attr @ self.restriction_maps.reshape((-1,self.stalk_dim*self.stalk_dim))).reshape((-1,self.stalk_dim,self.stalk_dim))
                F_left = F
                F_right = F
            else:
                F = (edge_attr @ self.restriction_maps.reshape((-1,2*self.stalk_dim*self.stalk_dim))).reshape((-1,2,self.stalk_dim,self.stalk_dim))
                F_left = F[:,0,:,:]
                F_right = F[:,1,:,:]
            
        elif edge_attr.dtype == torch.long:
            #edge_attr is then a tensor of indices
            if self.matrix_weighted:
                F = self.restriction_maps[edge_attr,:,:]
                F_left = F
                F_right = F
            else:
                F = self.restriction_maps[edge_attr,:,:,:]
                F_left = F[:,0,:,:]
                F_right = F[:,1,:,:]
            
        if self.signtype == 'sheaf':
            edge_weights = self._edge_weights(edge_attr)
            F_left = torch.abs(edge_weights) * F_left
            F_right = edge_weights * F_right
        
        return (F_left, F_right)

    def _x_to_xe(self,x: torch.FloatTensor,edge_index: torch.LongTensor) -> torch.FloatTensor:
        x_left = x.index_select(0,edge_index[0,:])
        x_right = x.index_select(0,edge_index[1,:])
        return (x_left, x_right)

    def _edge_weights(self,edge_attr: torch.Tensor) -> torch.FloatTensor:
        if self.signed:
            n_edges = edge_attr.shape[0]
            edge_weights = (edge_attr @ self.signs.reshape((self.n_edge_features,-1))).reshape((n_edges,-1,1))
        else:
            raise NotImplementedError()
        
        return edge_weights

    def coboundary(self, x: torch.FloatTensor, edge_index: torch.LongTensor, edge_attr: torch.Tensor) -> torch.FloatTensor:
        """
        Computes the sheaf coboundary of a collection of 0-cochains.
        
        Parameters
        ----------
        x : The tensor representing the 0-cochains. x should have dimensions (num_nodes, stalk_dim, features).
        edge_index : A (2, num_edges) tensor defining an oriented edge from edge_index[0,e] to edge_index[1,e] for each e.
        edge_attr : A (num_edges, n_edge_attr) tensor containing the edge attribute vector for each edge. This tensor determines the restriction maps used in the computation of the sheaf coboundary.
        """
        # x should be indexed (nodes, stalkdim, features)
        # in particular, it should be a 3-tensor
        x_left, x_right = self._x_to_xe(x,edge_index)
        F_left, F_right = self._restriction_maps(edge_attr)

        # for matrix_weighted, we could save half the matmuls by subtracting node features edgewise first. this will almost always be faster
        dx = F_left @ x_left - F_right @ x_right
        return dx

    def universal_coboundary(self, x: torch.FloatTensor, edge_index: torch.LongTensor) -> torch.FloatTensor:
        # this uses the restriction maps corresponding to each edge attribute and computes the coboundary separately. used to calculate the edge diffusion gradient
        x_left, x_right = self._x_to_xe(x,edge_index)

        if self.matrix_weighted:
            if self.signtype == 'sheaf':
                # raise NotImplementedError()
                x_left = torch.abs(self.signs.unsqueeze(1)) *(self.restriction_maps.unsqueeze(1) @ x_left.unsqueeze(0))
                x_right = self.signs.unsqueeze(1) * (self.restriction_maps.unsqueeze(1) @ x_right.unsqueeze(0))
                dx = x_left - x_right
            else:
                dx = self.restriction_maps.unsqueeze(1) @ (x_left - x_right).unsqueeze(0)
        else:
            x_e = torch.cat((x_left.unsqueeze(1),-x_right.unsqueeze(1)),dim=1) #this makes the sum calculate the correctly signed coboundary. x_e is (edges, 2, stalkdim, features)
            x_e = self.restriction_maps.unsqueeze(1) @ x_e.unsqueeze(0) #(edge_attr, 1, 2, stalkdim, stalkdim) @ (1, edges, 2, stalkdim, features) = (edge_attr, edges, 2, stalkdim, features)
            if self.signtype == 'sheaf':
                weights = torch.cat((torch.abs(self.signs.unsqueeze(1)),self.signs.unsqueeze(1)),dim=1).unsqueeze(1) #(edge_attr, 1, 2, ...)
                dx = torch.sum(weights * x_e, dim = 2) #the broadcasting might not work right here? 
                #raise NotImplementedError() #implement later
            else:
                dx = torch.sum(x_e, dim=2)
        #dx is now (edge_attr,edges,stalkdim,features)
        return dx

    def boundary(self, x_e: torch.FloatTensor, edge_index: torch.LongTensor, edge_attr: torch.Tensor, out_dim: int):
        # x_e should be indexed (edges, edgestalkdim, features)
        # out_dim is required because there may be isolated vertices that edge_index doesn't include

        F_left, F_right = self._restriction_maps(edge_attr)

        x_e_left = F_left.permute(0,2,1) @ x_e
        x_e_right = F_right.permute(0,2,1) @ x_e

        x = torch.scatter(x_e_left,edge_index[0,:],dim=0,out=torch.zeros(out_dim))
        torch.scatter(-x_e_right,edge_index[1,:],dim=0,out=x)
        return x

    def Laplacian(self, x: torch.FloatTensor, edge_index: torch.LongTensor, edge_attr: torch.Tensor) -> torch.FloatTensor:
        """
        Computes the sheaf Laplacian of a collection of 0-cochains. The precise Laplacian calculated depends on the module parameter signtype.
        
        Parameters
        ----------
        x : The tensor representing the 0-cochains. x should have dimensions (num_nodes, stalk_dim, features).
        edge_index : A (2, num_edges) tensor defining an oriented edge from edge_index[0,e] to edge_index[1,e] for each e.
        edge_attr : A (num_edges, n_edge_attr) tensor containing the edge attribute vector for each edge. This tensor determines the restriction maps used in the computation of the sheaf Laplacian.
        """
        x_left, x_right = self._x_to_xe(x,edge_index)
        F_left, F_right = self._restriction_maps(edge_attr)

        dx = F_left @ x_left - F_right @ x_right
        if self.signtype == 'gradient':
            edge_weights = self._edge_weights(edge_attr)
            if(len(edge_weights.shape) != len(dx.shape)):
                edge_weights = edge_weights.reshape(torch.cat([edge_weights.shape,torch.ones(len(dx.shape)-len(edge_weights.shape),dtype=torch.int)]))
            dx = edge_weights * dx
        x_e_left = F_left.permute(0,2,1) @ dx
        x_e_right = F_right.permute(0,2,1) @ dx
        
        # Would be straightforward to add a sort of geometrically normalized Laplacian by using mean aggregation for the scatter operation
        # oh, not quite: it would take the mean over in-edges and the mean over out-edges, not the mean over both...

        x_out = torch.zeros_like(x)
        torch.scatter(x_e_left,edge_index[0,:],dim=0,out=x_out)
        torch.scatter(-x_e_right,edge_index[1,:],dim=0,out=x_out)

        if self.degree_normalize:
            degrees = torch_geometric.utils.degree(edge_index)
            x_out = x_out / degrees.reshape((-1,1,1)) # compute D^{-1} x_out

        return x_out
    
    def sheaf_diffusion(self, x: torch.FloatTensor, edge_index: torch.LongTensor, edge_attr: torch.Tensor, alpha: float=None) -> torch.FloatTensor:
        """
        Computes a step in the Euler approximation to the sheaf diffusion equation \dot{x} = -Lx. 
        Equivalent to x - alpha * Laplacian(x, edge_index, edge_attr).
        
        Parameters
        ----------
        x : The tensor representing the 0-cochains. x should have dimensions (num_nodes, stalk_dim, features).
        edge_index : A (2, num_edges) tensor defining an oriented edge from edge_index[0,e] to edge_index[1,e] for each e.
        edge_attr : A (num_edges, n_edge_attr) tensor containing the edge attribute vector for each edge. This tensor determines the restriction maps used in the computation of the sheaf Laplacian.
        alpha : Size of the Euler step. If None, defaults to the layer parameter euler_stepsize
        """
        if alpha is None:
            alpha = self.euler_stepsize_gradient
        Lx = self.Laplacian(x, edge_index, edge_attr)
        if self.adaptive_stepsize:
            alpha = torch.linalg.norm(x)/(torch.clamp(2*torch.linalg.norm(Lx),min=1)) #ensure ||Lx|| = 1/2 ||x||
        return x - alpha * Lx
    
    def edge_diffusion_gradient(self, x: torch.FloatTensor, edge_index: torch.LongTensor, edge_attr: torch.FloatTensor) -> torch.FloatTensor:
        """
        Calculates the gradient with respect to the edge features of the discrepancy map \|\delta x\|_F^2.
        The main purpose of this is to serve as the generator of a diffusion process on edge features that modifies the sheaf to make a given set of cochains more consistent.
        Warning: currently ignores signtype 'gradient,' so the computed diffusion with these settings will be the same as for an unsigned Laplacian.
        Parameters
        ----------
        x : The tensor representing the 0-cochains. x should have dimensions (num_nodes, stalk_dim, features).
        edge_index : A (2, num_edges) tensor defining an oriented edge from edge_index[0,e] to edge_index[1,e] for each e.
        edge_attr : A (num_edges, n_edge_attr) tensor containing the edge attribute vector for each edge. This tensor determines the restriction maps used in the computation of the sheaf Laplacian.
        """
        dx_all = self.universal_coboundary(x,edge_index)
        dx = self.coboundary(x,edge_index,edge_attr)
        #dx = (edge_attr @ dx_all.reshape((self.n_edge_features,-1))).reshape((n_edges,self.stalk_dim,-1)) #this might make it faster
        gradient = torch.sum(dx_all * dx,dim=(2,3)).permute((1,0)) #(n_edge_features, edges)
        return gradient 

    def edge_diffusion(self, x: torch.FloatTensor, edge_index: torch.LongTensor, edge_attr: torch.FloatTensor, alpha: float=None) -> torch.FloatTensor:
        """
        Computes a step in the Euler approximation to the structural sheaf diffusion equation given by (roughly) \dot{\delta} = -\delta x^Tx.
        Equivalent to edge_attr - alpha * edge_diffusion_gradient(x, edge_index, edge_attr)
        Parameters
        ----------
        x : The tensor representing the 0-cochains. x should have dimensions (num_nodes, stalk_dim, features).
        edge_index : A (2, num_edges) tensor defining an oriented edge from edge_index[0,e] to edge_index[1,e] for each e.
        edge_attr : A (num_edges, n_edge_attr) tensor containing the edge attribute vector for each edge. This tensor determines the restriction maps used in the computation of the sheaf Laplacian.
        alpha : Size of the Euler step. If None, defaults to the layer parameter euler_stepsize
        """
        if alpha is None:
            alpha = self.euler_stepsize_cochain
        grad = self.edge_diffusion_gradient(x, edge_index, edge_attr)
        if self.adaptive_stepsize:
            alpha = torch.linalg.norm(grad)/(torch.clamp(5*torch.linalg.norm(edge_attr),min=1))
        return edge_attr - alpha * grad

    def forward(self, x: torch.FloatTensor, edge_index: torch.LongTensor, edge_attr: torch.Tensor):
        if self.layertype == 'Laplacian':
            return self.Laplacian(x, edge_index, edge_attr)
        elif self.layertype == 'cochain_diffusion':
            return self.sheaf_diffusion(x, edge_index, edge_attr, self.euler_stepsize_gradient), edge_attr
        elif self.layertype == 'sheaf_gradient':
            return self.edge_diffusion_gradient(x, edge_index, edge_attr)
        elif self.layertype == 'combined_diffusion':
            return (self.sheaf_diffusion(x, edge_index, edge_attr), self.edge_diffusion(x, edge_index, edge_attr))
        

class SheafMap(torch.nn.Module):
    #TODO: figure out the right initialization
    def __init__(self, in_stalk_dim: int, out_stalk_dim: int, in_feature_dim: int, out_feature_dim: int, affine: bool=False):
        super(SheafMap,self).__init__()
        self.affine = affine
        self.morphism = torch.nn.Parameter(torch.randn(out_stalk_dim,in_stalk_dim)/in_stalk_dim)
        self.feature_map = torch.nn.Parameter(torch.randn(in_feature_dim,out_feature_dim)/in_feature_dim)
        if self.affine:
            self.b = torch.nn.Parameter(torch.zeros(out_stalk_dim,out_feature_dim)) #should this really be a matrix, or should it be uniform across one or both dimensions?
            self.register_parameter("affine_term",self.b)
        self.register_parameter("morphism",self.morphism)
        self.register_parameter("feature_map",self.feature_map)
        self.in_stalk_dim = in_stalk_dim
        self.out_stalk_dim = out_stalk_dim
        self.in_feature_dim = in_feature_dim
        self.out_feature_dim = out_feature_dim

    def forward(self,x: torch.FloatTensor) -> torch.FloatTensor:
        # x should be (nodes, stalkdim, features)
        # there should probably be some sort of affine term here.
        if self.affine:
            return self.morphism @ x @ self.feature_map + self.b
            #return torch.chain_matmul(self.morphism,x,self.feature_map) + self.b
        else:
            return self.morphism @ x @ self.feature_map 
            #return torch.chain_matmul(self.morphism,x,self.feature_map) #this should be more time/memory efficient for large dimensions. okay, I guess you can only use this with actual matrices for some dumb reason.
        # pytorch matmul broadcasting is so weird to me. I guess the idea is that everything is a collection of matrices in the last two indices. so that's how matmul acts.