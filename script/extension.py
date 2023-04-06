from typing import Tuple

import torch
from torch_scatter import scatter
from torch_geometric.utils import degree
from pykeen.models.nbase import ERModel
from pykeen.models.unimodal.structured_embedding import SE
from pykeen.models.unimodal.trans_e import TransE
from pykeen.models.unimodal.trans_r import TransR
from pykeen.models.unimodal.rotate import RotatE
from pykeen.utils import clamp_norm

ALPHA = 1e-1

def coboundary(edge_index,Fh,Ft,relabel=False):
    device = Fh.device
    if relabel:
        _, edge_index = torch.unique(edge_index, sorted=True, return_inverse=True)
    ne = edge_index.shape[-1]
    nv = edge_index.max() + 1
    de = Fh.shape[-2]
    dv = Fh.shape[-1]
    d = torch.zeros((ne*de,nv*dv), device=device)
    for e in range(ne):
        h = edge_index[0,e]
        t = edge_index[1,e]
        d[e*de:(e+1)*de,h*dv:(h+1)*dv] = Fh[e,:,:]
        d[e*de:(e+1)*de,t*dv:(t+1)*dv] = -Ft[e,:,:]
    return d

def diffuse_interior(diffuser, triples, interior_ent_msk):
    edge_index = triples[:,[0,2]].T
    relations = triples[:,1]
    all_ents = edge_index.flatten().unique()
    num_nodes = all_ents.size(0)
    interior_vertices = all_ents[interior_ent_msk]

    xU = diffuser.diffuse_interior(edge_index, relations, interior_vertices, nv=num_nodes)
    return xU

class KGExtender():
    '''Harmonic extension base class.
    '''
    def __init__(self,
                 model: ERModel,
                 alpha: float=ALPHA,
                 degree_normalize: bool = False) -> None:
        self.model = model
        self._check_model_type()
        self.alpha = alpha
        self.degree_normalize = degree_normalize
        self.device = self.model.device

    def _check_model_type(self):
        raise NotImplementedError
    
    def _ht(self, edge_index: torch.LongTensor) -> Tuple[torch.Tensor]:
        nu = torch.LongTensor([0])
        xh, _, xt = self.model._get_representations(h=edge_index[0,:], r=nu, t=edge_index[1,:], mode=None)
        return xh, xt
    
    def laplacian_block_dense(self, edge_index: torch.LongTensor, edge_type: torch.LongTensor):
        Fh, Ft = self._restriction_maps(edge_type)
        d = coboundary(edge_index, Fh, Ft)
        return torch.matmul(torch.transpose(d,0,1), d)
    
    def laplacian_UB_quad(self, edge_index: torch.LongTensor, edge_type: torch.LongTensor):
        '''TODO
        '''
        xh, xt = self._ht(edge_index)
        Fh, Ft = self._restriction_maps(edge_type)

        unique, edge_index_inv = torch.unique(edge_index, sorted=True, return_inverse=True)

        dx = Fh @ xh.unsqueeze(-1) - Ft @ xt.unsqueeze(-1)
        x_e_h = Fh.permute(0,-1,-2) @ dx
        x_e_t = Ft.permute(0,-1,-2) @ dx

        Lx = torch.zeros((unique.shape[0]))
        scatter(x_e_h.squeeze(-1),edge_index_inv[0,:],dim=0,out=Lx)
        scatter(-x_e_t.squeeze(-1),edge_index_inv[1,:],dim=0,out=Lx)

        return Lx
        
    
class SEExtender(KGExtender):
    '''Harmonic extension for Structured Embedding model.'''
    def __init__(self,
                 model: ERModel,
                 alpha: float=ALPHA,
                 degree_normalize: bool=True) -> None:
        super().__init__(model=model, alpha=alpha, degree_normalize=degree_normalize)

    def _check_model_type(self) -> None:
        assert isinstance(self.model, SE)

    def _restriction_maps(self, edge_type:torch.LongTensor) -> Tuple[torch.Tensor]:
        nu = torch.LongTensor([0])
        _, r, _ = self.model._get_representations(h=nu, r=edge_type, t=nu, mode=None)
        return r[0], r[1]
    
    def laplacian_mult(self, edge_index: torch.LongTensor, edge_type: torch.LongTensor, relabel: bool=False, nv: int=None):
        edge_index = edge_index.to(self.device)
        xh, xt = self._ht(edge_index)
        Fh, Ft = self._restriction_maps(edge_type)

        if relabel:
            _, edge_index = torch.unique(edge_index, sorted=True, return_inverse=True)
        if nv is None:
            nv = edge_index.max() + 1
        dx = Fh @ xh.unsqueeze(-1) - Ft @ xt.unsqueeze(-1)
        x_e_h = Fh.permute(0,-1,-2) @ dx
        x_e_t = Ft.permute(0,-1,-2) @ dx

        Lx = torch.zeros((nv, xh.shape[1])).to(self.device)
        scatter(x_e_h.squeeze(-1),edge_index[0,:],dim=0,out=Lx)
        scatter(-x_e_t.squeeze(-1),edge_index[1,:],dim=0,out=Lx)

        if self.degree_normalize:
            # degrees = xh.shape[1]*(degree(edge_index[0,:]) + degree(edge_index[1,:]))
            degrees = xh.shape[1]*(degree(edge_index.flatten())/2)
            Lx = Lx / degrees.reshape((-1,1))

        return Lx
    
    def diffuse_interior(self, edge_index: torch.LongTensor, edge_type: torch.LongTensor, interior_vertices: torch.LongTensor,
                          relabel: bool=False, nv: int=None):
        xU = self.laplacian_mult(edge_index, edge_type, relabel=relabel, nv=nv)
        self.model.entity_representations[0]._embeddings.weight[interior_vertices] -= self.alpha*xU[interior_vertices]
        return xU

    # try getting BU edges and UU edges first
    def harmonic_extension(self, 
                            interior_mask: torch.BoolTensor,
                            interior_boundary_mask: torch.BoolTensor,
                            edge_index: torch.LongTensor,
                            edge_type: torch.LongTensor) -> torch.Tensor:

        # TODO
        LUU = self.laplacian_block_dense(edge_index[:,interior_mask], edge_type[:,interior_mask])
        LUU_inv = torch.linalg.pinv(LUU)
        LUB = self.laplacian_block_dense(edge_index[:,interior_boundary_mask], edge_type[:,interior_boundary_mask])

        xU = -torch.linalg.lstsq(LUU, LUB).solution
        return xU

class TransEExtender(KGExtender):
    '''Harmonic extension for TransE model.'''
    def __init__(self,
                 model: ERModel,
                 alpha: float = ALPHA,
                 degree_normalize: bool = True) -> None:
        super().__init__(model=model, alpha=alpha, degree_normalize=degree_normalize)

    def _check_model_type(self):
        assert isinstance(self.model, TransE)
    
    def _b(self, edge_type:torch.LongTensor) -> Tuple[torch.Tensor]:
        nu = torch.LongTensor([0])
        _, r, _ = self.model._get_representations(h=nu, r=edge_type, t=nu, mode=None)
        return r
    
    def laplacian_mult_translational(self, edge_index: torch.LongTensor, edge_type: torch.LongTensor, relabel: bool=False, nv: int=None):
        edge_index = edge_index.to(self.device)
        xh, xt = self._ht(edge_index)
        b = self._b(edge_type)

        if relabel:
            _, edge_index = torch.unique(edge_index, sorted=True, return_inverse=True)
        if nv is None:
            nv = edge_index.max() + 1
        dx = xh.unsqueeze(-1) + b.unsqueeze(-1) - xt.unsqueeze(-1)
        x_e_h = dx
        x_e_t = dx

        Lx = torch.zeros((nv, xh.shape[1])).to(self.device)
        scatter(x_e_h.squeeze(-1),edge_index[0,:],dim=0,out=Lx)
        scatter(-x_e_t.squeeze(-1),edge_index[1,:],dim=0,out=Lx)

        if self.degree_normalize:
            degrees = xh.shape[1]*(degree(edge_index.flatten())/2)
            Lx = Lx / degrees.reshape((-1,1))

        return Lx
    
    def diffuse_interior(self, edge_index: torch.LongTensor, edge_type: torch.LongTensor, interior_vertices: torch.LongTensor,
                          relabel: bool=False, nv: int=None):
        xU = self.laplacian_mult_translational(edge_index, edge_type, relabel=relabel, nv=nv)
        self.model.entity_representations[0]._embeddings.weight[interior_vertices] -= self.alpha*xU[interior_vertices]
        return xU
    
class RotatEExtender(KGExtender):
    '''Harmonic extension for RotatE model.'''
    def __init__(self,
                 model: ERModel,
                 alpha: float = ALPHA,
                 degree_normalize: bool = True) -> None:
        super().__init__(model=model, alpha=alpha, degree_normalize=degree_normalize)

    def _check_model_type(self):
        assert isinstance(self.model, RotatE)
    
    def _b(self, edge_type:torch.LongTensor) -> Tuple[torch.Tensor]:
        nu = torch.LongTensor([0])
        _, r, _ = self.model._get_representations(h=nu, r=edge_type, t=nu, mode=None)
        return r
    
    def laplacian_mult_translational(self, edge_index: torch.LongTensor, edge_type: torch.LongTensor, relabel: bool=False, nv: int=None):
        edge_index = edge_index.to(self.device)
        xh, xt = self._ht(edge_index)
        b = self._b(edge_type)

        if relabel:
            _, edge_index = torch.unique(edge_index, sorted=True, return_inverse=True)
        if nv is None:
            nv = edge_index.max() + 1
        dx = xh.unsqueeze(-1) * b.unsqueeze(-1) - xt.unsqueeze(-1)
        x_e_h = dx
        x_e_t = dx

        Lx = torch.zeros((nv, xh.shape[1]), dtype=dx.dtype).to(self.device)
        scatter(x_e_h.squeeze(-1),edge_index[0,:],dim=0,out=Lx)
        scatter(-x_e_t.squeeze(-1),edge_index[1,:],dim=0,out=Lx)

        if self.degree_normalize:
            degrees = xh.shape[1]*(degree(edge_index.flatten())/2)
            Lx = Lx / degrees.reshape((-1,1))

        return Lx
    
    def diffuse_interior(self, edge_index: torch.LongTensor, edge_type: torch.LongTensor, interior_vertices: torch.LongTensor,
                          relabel: bool=False, nv: int=None):
        xU = self.laplacian_mult_translational(edge_index, edge_type, relabel=relabel, nv=nv)
        self.model.entity_representations[0]._embeddings.weight[interior_vertices] -= torch.view_as_real(self.alpha*xU[interior_vertices]).reshape((interior_vertices.shape[0],-1))
        return xU

class TransRExtender(KGExtender):
    '''Harmonic extension for TransR model.'''
    def __init__(self,
                 model: ERModel,
                 alpha: float = ALPHA,
                 degree_normalize: bool=True) -> None:
        super().__init__(model=model, alpha=alpha, degree_normalize=degree_normalize)

    def _check_model_type(self):
        assert isinstance(self.model, TransR)

    def _restriction_maps(self, edge_type:torch.LongTensor) -> Tuple[torch.Tensor]:
        nu = torch.LongTensor([0])
        _, r, _ = self.model._get_representations(h=nu, r=edge_type, t=nu, mode=None)
        ret = r[1].permute(0,-1,-2)
        return ret, ret
    
    def _b(self, edge_type:torch.LongTensor) -> Tuple[torch.Tensor]:
        nu = torch.LongTensor([0])
        _, r, _ = self.model._get_representations(h=nu, r=edge_type, t=nu, mode=None)
        return r[0]
    
    def laplacian_mult_translational(self, edge_index: torch.LongTensor, edge_type: torch.LongTensor, relabel: bool=False, nv: int=None):
        edge_index = edge_index.to(self.device)
        xh, xt = self._ht(edge_index)
        b = self._b(edge_type)
        Fh, Ft = self._restriction_maps(edge_type)

        if relabel:
            _, edge_index = torch.unique(edge_index, sorted=True, return_inverse=True)
        if nv is None:
            nv = edge_index.max() + 1
        dx = clamp_norm(Fh @ xh.unsqueeze(-1), p=2, dim=-2, maxnorm=1) + b.unsqueeze(-1) - clamp_norm(Ft @ xt.unsqueeze(-1), p=2, dim=-2, maxnorm=1)
        # dx = Fh @ xh.unsqueeze(-1) + b.unsqueeze(-1) - Ft @ xt.unsqueeze(-1)
        x_e_h = Fh.permute(0,-1,-2) @ dx
        x_e_t = Ft.permute(0,-1,-2) @ dx

        Lx = torch.zeros((nv, xh.shape[1])).to(self.device)
        scatter(x_e_h.squeeze(-1),edge_index[0,:],dim=0,out=Lx)
        scatter(-x_e_t.squeeze(-1),edge_index[1,:],dim=0,out=Lx)

        if self.degree_normalize:
            degrees = xh.shape[1]*(degree(edge_index.flatten())/2)
            Lx = Lx / degrees.reshape((-1,1))

        return Lx
    
    def diffuse_interior(self, edge_index: torch.LongTensor, edge_type: torch.LongTensor, interior_vertices: torch.LongTensor,
                          relabel: bool=False, nv: int=None):
        xU = self.laplacian_mult_translational(edge_index, edge_type, relabel=relabel, nv=nv)
        self.model.entity_representations[0]._embeddings.weight[interior_vertices] -= self.alpha*xU[interior_vertices]
        return xU
    

def get_extender(model_type):
    if model_type == 'se':
        return SEExtender
    if model_type == 'transe':
        return TransEExtender
    if model_type == 'rotate':
        return RotatEExtender
    if model_type == 'transr':
        return TransRExtender