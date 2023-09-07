from typing import Tuple, List
from abc import ABC, abstractmethod
from tqdm import tqdm

import torch
from torch_scatter import scatter
from torch_geometric.utils import degree
from pykeen.models.nbase import ERModel
from pykeen.models.unimodal.trans_e import TransE
from pykeen.models.unimodal.trans_r import TransR
from pykeen.models.unimodal.rotate import RotatE
from pykeen.utils import clamp_norm

from models import SE

import batch_harmonic_extension as bhe

class ComplexExtender(ABC):
    '''Harmonic extension base class.
    '''
    def __init__(self,
                 model: ERModel,
                 dtype: torch.Type = torch.float32) -> None:
        self.model = model
        self.num_entities = self._get_num_entities()
        self._check_model_type()
        self.device = self.model.device
        self.dtype = dtype

    def _check_model_type(self):
        raise NotImplementedError
    
    def _get_num_entities(self) -> int:
        # use this instead of model.num_entities because num_entities 
        # may not get reset during the extension phase.
        nu = torch.LongTensor([0])
        _,_,t = self.model._get_representations(h=nu, r=nu, t=None, mode=None)
        return t.shape[0]
    
    def _ht(self, edge_index: torch.LongTensor) -> Tuple[torch.Tensor]:
        nu = torch.LongTensor([0])
        xh, _, xt = self.model._get_representations(h=edge_index[0,:], r=nu, t=edge_index[1,:], mode=None)
        return xh, xt
    
    def _h(self, hix: torch.LongTensor) -> Tuple[torch.Tensor]:
        nu = torch.LongTensor([0])
        xh, _, _ = self.model._get_representations(h=hix, r=nu, t=nu, mode=None)
        return xh
    
    def _t(self, tix: torch.LongTensor) -> Tuple[torch.Tensor]:
        nu = torch.LongTensor([0])
        _, _, xt = self.model._get_representations(h=nu, r=nu, t=tix, mode=None)
        return xt
    
    def query_structure_2p(self) -> dict:
        return None

    def query_structure_2p(self) -> dict:
        return dict(
                edge_index = torch.LongTensor([[0,1],[1,2]]).T,
                boundary_vertices = torch.LongTensor([0,2]),
                interior_vertices = torch.LongTensor([1]),
                source_vertices = torch.LongTensor([0]),
                target_vertices = torch.LongTensor([1])
        )

    def query_structure_3p(self) -> dict:
        return dict(
                edge_index = torch.LongTensor([[0,1],[1,2],[2,3]]).T,
                boundary_vertices = torch.LongTensor([0,3]),
                interior_vertices = torch.LongTensor([1,2]),
                source_vertices = torch.LongTensor([0]),
                target_vertices = torch.LongTensor([1])
        )
    
    def query_structure_2i(self) -> dict:
        return dict(
                edge_index = torch.LongTensor([[0,2],[1,2]]).T,
                boundary_vertices = torch.LongTensor([0,1,2]),
                source_vertices = torch.LongTensor([0,1]),
                target_vertices = torch.LongTensor([2])
        )

    def query_structure_3i(self) -> dict:
        return dict(
                edge_index = torch.LongTensor([[0,3],[1,3],[2,3]]).T,
                boundary_vertices = torch.LongTensor([0,1,2,3]),
                source_vertices = torch.LongTensor([0,1,2]),
                target_vertices = torch.LongTensor([3])
        )
    
    def query_structure_pi(self) -> dict:
        return dict(
                edge_index = torch.LongTensor([[0,2],[2,3],[1,3]]).T,
                boundary_vertices = torch.LongTensor([0,1,3]),
                interior_vertices = torch.LongTensor([2]),
                source_vertices = torch.LongTensor([0,1]),
                target_vertices = torch.LongTensor([2])
        )
    
    def query_structure_ip(self) -> dict:
        return dict(
                edge_index = torch.LongTensor([[0,2],[1,2],[2,3]]).T,
                boundary_vertices = torch.LongTensor([0,1,3]),
                interior_vertices = torch.LongTensor([2]),
                source_vertices = torch.LongTensor([0,1]),
                target_vertices = torch.LongTensor([2])
        )
    
    def score_complex(self, query_structure: str, queries: List[dict], tails: torch.Tensor) -> torch.Tensor:
        if query_structure == '1p':
            return self.score_1p(queries, tails)
        if query_structure == '2p':
            return self.score_2p(queries, tails)
        if query_structure == '3p':
            return self.score_3p(queries, tails)
        if query_structure == '2i':
            return self.score_2i(queries, tails)
        if query_structure == '3i':
            return self.score_3i(queries, tails)
        if query_structure == 'pi':
            return self.score_pi(queries, tails)
        if query_structure == 'ip':
            return self.score_ip(queries, tails)
        raise ValueError(f'query structure {query_structure} not implemented.')
    
    def score_1p(self, queries: List[dict], tails: torch.LongTensor):
        h = torch.cat([q['sources'] for q in queries])
        r = torch.cat([q['relations'] for q in queries])
        hr_batch = torch.cat([h.unsqueeze(-1),r.unsqueeze(-1)], dim=1)
        return self.model.score_t(hr_batch, tails=tails)
    
    @abstractmethod
    def score_2p(self, queries: List[dict], tails: torch.LongTensor) -> torch.FloatTensor:
        pass

    @abstractmethod
    def score_3p(self, queries: List[dict], tails: torch.LongTensor) -> torch.FloatTensor:
        pass

    @abstractmethod
    def score_2i(self, queries: List[dict], tails: torch.LongTensor) -> torch.FloatTensor:
        pass
    
    @abstractmethod
    def score_3i(self, queries: List[dict], tails: torch.LongTensor) -> torch.FloatTensor:
        pass

    @abstractmethod
    def score_pi(self, queries: List[dict], tails: torch.LongTensor) -> torch.FloatTensor:
        pass

    @abstractmethod
    def score_ip(self, queries: List[dict], tails: torch.LongTensor) -> torch.FloatTensor:
        pass

    def unpack_path_query_indices(self, queries: List[dict]) -> Tuple[torch.LongTensor]:
        hix = torch.cat([q['sources'] for q in queries], dim=0)
        rix = torch.cat([q['relations'].unsqueeze(0) for q in queries], dim=0)
        return hix, rix
    
    def unpack_intersection_query_indices(self, queries: List[dict]) -> Tuple[torch.LongTensor]:
        hix = torch.cat([q['sources'].unsqueeze(0) for q in queries], dim=0)
        rix = torch.cat([q['relations'].unsqueeze(0) for q in queries], dim=0)
        return hix, rix
    
    def unpack_pi_query_indices(self, queries: List[dict]) -> Tuple[torch.LongTensor]:
        hix = torch.cat([q['sources'].unsqueeze(0) for q in queries], axis=0)
        rix = torch.cat([q['relations'].unsqueeze(0) for q in queries], axis=0)
        return hix,rix

    def unpack_ip_query_indices(self, queries: List[dict]) -> Tuple[torch.LongTensor]:
        hix = torch.cat([q['sources'].unsqueeze(0) for q in queries], axis=0)
        rix = torch.cat([q['relations'].unsqueeze(0) for q in queries], axis=0)
        return hix,rix
    
    def slice_and_score_complex(self, query_structure: str, queries: List[dict],
                                batch_size: int, slice_size: int = None, progress=True) -> torch.FloatTensor:
        
        slice_size = self.num_entities if slice_size is None else slice_size
        ent_idxs = torch.arange(self.num_entities)
        res = torch.zeros((len(queries), self.num_entities))
        for bix in tqdm(range(0,len(queries), batch_size), desc=f'batched {query_structure}', disable=not progress):
            batch = queries[bix:bix+batch_size]
            
            for slix in range(0, self.num_entities, slice_size):
                slice = ent_idxs[slix:slix+slice_size]

                scores = self.score_complex(query_structure, batch, slice)

                res[bix:bix+batch_size,slix:slix+slice_size] = scores

        return res


class SEComplex(ComplexExtender):
    '''Harmonic extension for Structured Embedding model.'''
    def __init__(self,
                 model: ERModel) -> None:
        super().__init__(model=model)

    def _check_model_type(self) -> None:
        assert isinstance(self.model, SE)

    def _restriction_maps(self, edge_type:torch.LongTensor) -> torch.Tensor:
        nu = torch.LongTensor([0])
        _, r, _ = self.model._get_representations(h=nu, r=edge_type, t=nu, mode=None)
        restriction_maps = torch.cat([tr.unsqueeze(2) for tr in r], dim=2)
        return restriction_maps
    
    def score_schur_batched(self, edge_index, restriction_maps, 
                            boundary_vertices, interior_vertices,
                            source_vertices, target_vertices, 
                            xS, xT, dv) -> torch.FloatTensor:
        LSchur = bhe.Kron_reduction(edge_index, restriction_maps, boundary_vertices, interior_vertices)
        return bhe.compute_costs(LSchur, source_vertices, target_vertices, xS, xT, dv)
    
    def score_intersect_batched(self, edge_index, restriction_maps, 
                            source_vertices, target_vertices, 
                            xS, xT, dv):
        L = bhe.Laplacian(edge_index, restriction_maps)
        return bhe.compute_costs(L, source_vertices, target_vertices, xS, xT, dv)
    
    def score_2p(self, queries: List[dict], tails: torch.LongTensor) -> torch.FloatTensor:

        qs = self.query_structure_2p()
        hix, rix = self.unpack_path_query_indices(queries)
        h = self._h(hix)
        t = self._t(tails).unsqueeze(-1)
        restriction_maps = self._restriction_maps(rix)

        nbatch = restriction_maps.shape[0]
        
        scores = self.score_schur_batched(qs['edge_index'], restriction_maps, 
                                            qs['boundary_vertices'], qs['interior_vertices'], 
                                            qs['source_vertices'], qs['target_vertices'], 
                                            h.reshape(nbatch, -1, 1) , t, t.shape[-2])
        return -torch.sum(scores, dim=(-1))
    
    def score_3p(self, queries: List[dict], tails: torch.LongTensor) -> torch.FloatTensor:
        qs = self.query_structure_3p()
        hix, rix = self.unpack_path_query_indices(queries)
        h = self._h(hix)
        t = self._t(tails).unsqueeze(-1)
        restriction_maps = self._restriction_maps(rix)

        nbatch = restriction_maps.shape[0]
        
        scores = self.score_schur_batched(qs['edge_index'], restriction_maps, 
                                            qs['boundary_vertices'], qs['interior_vertices'], 
                                            qs['source_vertices'], qs['target_vertices'], 
                                            h.reshape(nbatch, -1, 1) , t, t.shape[-2])
        return -torch.sum(scores, dim=(-1))

    def score_2i(self, queries: List[dict], tails: torch.LongTensor) -> torch.FloatTensor:
        qs = self.query_structure_2i()
        hix, rix = self.unpack_intersection_query_indices(queries)
        h = self._h(hix)
        t = self._t(tails).unsqueeze(-1)
        restriction_maps = self._restriction_maps(rix)

        nbatch = restriction_maps.shape[0]
        
        scores = self.score_intersect_batched(qs['edge_index'], restriction_maps,
                                            qs['source_vertices'], qs['target_vertices'], 
                                            h.reshape(nbatch, -1, 1) , t, t.shape[-2])
        return -torch.sum(scores, dim=(-1))
    
    def score_3i(self, queries: List[dict], tails: torch.LongTensor) -> torch.FloatTensor:
        qs = self.query_structure_3i()
        hix, rix = self.unpack_intersection_query_indices(queries)
        h = self._h(hix)
        t = self._t(tails).unsqueeze(-1)
        restriction_maps = self._restriction_maps(rix)

        nbatch = restriction_maps.shape[0]
        
        scores = self.score_intersect_batched(qs['edge_index'], restriction_maps, 
                                            qs['source_vertices'], qs['target_vertices'], 
                                            h.reshape(nbatch, -1, 1) , t, t.shape[-2])
        return -torch.sum(scores, dim=(-1))

    def score_pi(self, queries: List[dict], tails: torch.LongTensor) -> torch.FloatTensor:
        qs = self.query_structure_pi()
        hix, rix = self.unpack_pi_query_indices(queries)
        h = self._h(hix)
        t = self._t(tails).unsqueeze(-1)
        restriction_maps = self._restriction_maps(rix)

        nbatch = restriction_maps.shape[0]
        
        scores = self.score_schur_batched(qs['edge_index'], restriction_maps, 
                                        qs['boundary_vertices'], qs['interior_vertices'], 
                                        qs['source_vertices'], qs['target_vertices'], 
                                        h.reshape(nbatch, -1, 1) , t, t.shape[-2])
        return -torch.sum(scores, dim=(-1))

    def score_ip(self, queries: List[dict], tails: torch.LongTensor) -> torch.FloatTensor:
        qs = self.query_structure_ip()
        hix, rix = self.unpack_ip_query_indices(queries)
        h = self._h(hix)
        t = self._t(tails).unsqueeze(-1)
        restriction_maps = self._restriction_maps(rix)

        nbatch = restriction_maps.shape[0]
        
        scores = self.score_schur_batched(qs['edge_index'], restriction_maps, 
                                        qs['boundary_vertices'], qs['interior_vertices'], 
                                        qs['source_vertices'], qs['target_vertices'], 
                                        h.reshape(nbatch, -1, 1) , t, t.shape[-2])
        return -torch.sum(scores, dim=(-1))

class TransEComplexVerbose(ComplexExtender):
    '''Harmonic extension for TransE model.'''
    def __init__(self,
                 model: ERModel) -> None:
        super().__init__(model=model)

    def _check_model_type(self):
        assert isinstance(self.model, TransE)

    def _restriction_maps(self, edge_type:torch.LongTensor) -> torch.Tensor:
        dv = self.model.entity_representations[0].shape[0]
        nbatch = edge_type.shape[0]
        nedges = edge_type.shape[1]
        IL = torch.eye(dv).view(1,1,1,dv,dv).to(self.device)
        IR = torch.eye(dv).view(1,1,1,dv,dv).to(self.device)
        restriction_maps = torch.cat([IL, IR], dim=2).repeat(nbatch,nedges,1,1,1)
        return restriction_maps
    
    def _b(self, edge_type:torch.LongTensor) -> Tuple[torch.Tensor]:
        nu = torch.LongTensor([0])
        _, r, _ = self.model._get_representations(h=nu, r=edge_type, t=nu, mode=None)
        return r
    
    def score_schur_batched(self, edge_index, restriction_maps, 
                            boundary_vertices, interior_vertices,
                            source_vertices, target_vertices, 
                            xS, xT, b, dv):
        LSchur, affine = bhe.Kron_reduction_translational(edge_index, restriction_maps, boundary_vertices, interior_vertices)
        return bhe.compute_costs_translational(LSchur, affine, source_vertices, target_vertices, xS, xT, b, dv)
                            
    def score_intersect_batched(self, edge_index, restriction_maps, 
                            source_vertices, target_vertices, 
                            xS, xT, b, dv):
        L = bhe.Laplacian(edge_index, restriction_maps)
        d = bhe.coboundary(edge_index, restriction_maps)
        return bhe.compute_costs_translational(L, d, source_vertices, target_vertices, xS, xT, b, dv)
    
    def score_2p(self, queries: List[dict], tails: torch.LongTensor) -> torch.FloatTensor:

        qs = self.query_structure_2p()
        hix, rix = self.unpack_path_query_indices(queries)
        h = self._h(hix)
        b = self._b(rix)
        t = self._t(tails).unsqueeze(-1)
        restriction_maps = self._restriction_maps(rix)

        nbatch = restriction_maps.shape[0]
        
        scores = self.score_schur_batched(qs['edge_index'], restriction_maps, 
                                            qs['boundary_vertices'], qs['interior_vertices'], 
                                            qs['source_vertices'], qs['target_vertices'], 
                                            h.reshape(nbatch, -1, 1), t, b.reshape(nbatch, -1, 1), t.shape[-2])
        return -torch.sum(scores, dim=(-1))
    
    def score_3p(self, queries: List[dict], tails: torch.LongTensor) -> torch.FloatTensor:
        qs = self.query_structure_3p()
        hix, rix = self.unpack_path_query_indices(queries)
        h = self._h(hix)
        b = self._b(rix)
        t = self._t(tails).unsqueeze(-1)
        restriction_maps = self._restriction_maps(rix)

        nbatch = restriction_maps.shape[0]
        
        scores = self.score_schur_batched(qs['edge_index'], restriction_maps, 
                                            qs['boundary_vertices'], qs['interior_vertices'], 
                                            qs['source_vertices'], qs['target_vertices'], 
                                            h.reshape(nbatch, -1, 1), t, b.reshape(nbatch, -1, 1), t.shape[-2])
        return -torch.sum(scores, dim=(-1))

    def score_2i(self, queries: List[dict], tails: torch.LongTensor) -> torch.FloatTensor:
        qs = self.query_structure_2i()
        hix, rix = self.unpack_intersection_query_indices(queries)
        h = self._h(hix)
        b = self._b(rix)
        t = self._t(tails).unsqueeze(-1)
        restriction_maps = self._restriction_maps(rix)

        nbatch = restriction_maps.shape[0]
        
        scores = self.score_intersect_batched(qs['edge_index'], restriction_maps,
                                            qs['source_vertices'], qs['target_vertices'], 
                                            h.reshape(nbatch, -1, 1), t, b.reshape(nbatch, -1, 1), t.shape[-2])
        return -torch.sum(scores, dim=(-1))
    
    def score_3i(self, queries: List[dict], tails: torch.LongTensor) -> torch.FloatTensor:
        qs = self.query_structure_3i()
        hix, rix = self.unpack_intersection_query_indices(queries)
        h = self._h(hix)
        b = self._b(rix)
        t = self._t(tails).unsqueeze(-1)
        restriction_maps = self._restriction_maps(rix)

        nbatch = restriction_maps.shape[0]
        
        scores = self.score_intersect_batched(qs['edge_index'], restriction_maps, 
                                            qs['source_vertices'], qs['target_vertices'], 
                                            h.reshape(nbatch, -1, 1), t, b.reshape(nbatch, -1, 1), t.shape[-2])
        return -torch.sum(scores, dim=(-1))

    def score_pi(self, queries: List[dict], tails: torch.LongTensor) -> torch.FloatTensor:
        qs = self.query_structure_pi()
        hix, rix = self.unpack_pi_query_indices(queries)
        h = self._h(hix)
        b = self._b(rix)
        t = self._t(tails).unsqueeze(-1)
        restriction_maps = self._restriction_maps(rix)

        nbatch = restriction_maps.shape[0]
        
        scores = self.score_schur_batched(qs['edge_index'], restriction_maps, 
                                        qs['boundary_vertices'], qs['interior_vertices'], 
                                        qs['source_vertices'], qs['target_vertices'], 
                                        h.reshape(nbatch, -1, 1), t, b.reshape(nbatch, -1, 1), t.shape[-2])
        return -torch.sum(scores, dim=(-1))

    def score_ip(self, queries: List[dict], tails: torch.LongTensor) -> torch.FloatTensor:
        qs = self.query_structure_ip()
        hix, rix = self.unpack_ip_query_indices(queries)
        h = self._h(hix)
        b = self._b(rix)
        t = self._t(tails).unsqueeze(-1)
        restriction_maps = self._restriction_maps(rix)

        nbatch = restriction_maps.shape[0]
        
        scores = self.score_schur_batched(qs['edge_index'], restriction_maps, 
                                        qs['boundary_vertices'], qs['interior_vertices'], 
                                        qs['source_vertices'], qs['target_vertices'], 
                                        h.reshape(nbatch, -1, 1), t, b.reshape(nbatch, -1, 1), t.shape[-2])
        return -torch.sum(scores, dim=(-1))

class TransEComplex(ComplexExtender):
    '''Harmonic extension for TransE model.'''
    def __init__(self,
                 model: ERModel) -> None:
        super().__init__(model=model)

    def _check_model_type(self):
        assert isinstance(self.model, TransE)
    
    def _b(self, edge_type:torch.LongTensor) -> Tuple[torch.Tensor]:
        nu = torch.LongTensor([0])
        _, r, _ = self.model._get_representations(h=nu, r=edge_type, t=nu, mode=None)
        return r
    
    def score_2p(self, queries: List[dict], tails: torch.LongTensor) -> torch.FloatTensor:
        hix, rix = self.unpack_path_query_indices(queries)
        h = self._h(hix).unsqueeze(1)
        b = self._b(rix).sum(dim=1).unsqueeze(1)
        t = self._t(tails).unsqueeze(0)

        return self.model.interaction.score(h=h, r=b, t=t)
    
    def score_3p(self, queries: List[dict], tails: torch.LongTensor) -> torch.FloatTensor:
        hix, rix = self.unpack_path_query_indices(queries)
        h = self._h(hix).unsqueeze(1)
        b = self._b(rix).sum(dim=1).unsqueeze(1)
        t = self._t(tails).unsqueeze(0)

        return self.model.interaction.score(h=h, r=b, t=t)

    def score_2i(self, queries: List[dict], tails: torch.LongTensor) -> torch.FloatTensor:
        hix, rix = self.unpack_intersection_query_indices(queries)
        h = self._h(hix).unsqueeze(1)
        b = self._b(rix).unsqueeze(1)
        t = self._t(tails).unsqueeze(0)

        hnew = (h + b).sum(dim=2)

        return self.model.interaction.score(h=hnew, r=torch.zeros_like(hnew), t=t)
    
    def score_3i(self, queries: List[dict], tails: torch.LongTensor) -> torch.FloatTensor:
        hix, rix = self.unpack_intersection_query_indices(queries)
        h = self._h(hix).unsqueeze(1)
        b = self._b(rix).unsqueeze(1)
        t = self._t(tails).unsqueeze(0)

        hnew = (h + b).sum(dim=2)

        return self.model.interaction.score(h=hnew, r=torch.zeros_like(hnew), t=t)

    def score_pi(self, queries: List[dict], tails: torch.LongTensor) -> torch.FloatTensor:
        hix, rix = self.unpack_pi_query_indices(queries)
        h = self._h(hix).unsqueeze(1)
        b = self._b(rix).unsqueeze(1)
        t = self._t(tails).unsqueeze(0)

        hnew = h.sum(dim=2) + b.sum(dim=2)
        return self.model.interaction.score(h=hnew, r=torch.zeros_like(hnew), t=t)

    def score_ip(self, queries: List[dict], tails: torch.LongTensor) -> torch.FloatTensor:
        hix, rix = self.unpack_pi_query_indices(queries)
        h = self._h(hix).unsqueeze(1)
        b = self._b(rix).unsqueeze(1)
        t = self._t(tails).unsqueeze(0)

        hnew = h.sum(dim=2) + b.sum(dim=2)
        return self.model.interaction.score(h=hnew, r=torch.zeros_like(hnew), t=t)
        
class RotatEComplex(ComplexExtender):
    '''Harmonic extension for RotatE model.'''
    def __init__(self,
                 model: ERModel) -> None:
        super().__init__(model=model, dtype=torch.cfloat)

    def _check_model_type(self):
        assert isinstance(self.model, RotatE)
    
    def _b(self, edge_type:torch.LongTensor) -> Tuple[torch.Tensor]:
        nu = torch.LongTensor([0])
        _, r, _ = self.model._get_representations(h=nu, r=edge_type, t=nu, mode=None)
        return r
    
    def score_2p(self, queries: List[dict], tails: torch.LongTensor) -> torch.FloatTensor:
        hix, rix = self.unpack_path_query_indices(queries)
        h = self._h(hix).unsqueeze(1)
        b = self._b(rix).prod(dim=1).unsqueeze(1)
        t = self._t(tails).unsqueeze(0)

        return self.model.interaction.score(h=h, r=b, t=t)
    
    def score_3p(self, queries: List[dict], tails: torch.LongTensor) -> torch.FloatTensor:
        hix, rix = self.unpack_path_query_indices(queries)
        h = self._h(hix).unsqueeze(1)
        b = self._b(rix).prod(dim=1).unsqueeze(1)
        t = self._t(tails).unsqueeze(0)

        return self.model.interaction.score(h=h, r=b, t=t)

    def score_2i(self, queries: List[dict], tails: torch.LongTensor) -> torch.FloatTensor:
        hix, rix = self.unpack_intersection_query_indices(queries)
        h = self._h(hix).unsqueeze(1)
        b = self._b(rix).unsqueeze(1)
        t = self._t(tails).unsqueeze(0)

        hnew = (h * b).sum(dim=2)

        return self.model.interaction.score(h=hnew, r=torch.ones_like(hnew, dtype=self.dtype), t=t)
    
    def score_3i(self, queries: List[dict], tails: torch.LongTensor) -> torch.FloatTensor:
        hix, rix = self.unpack_intersection_query_indices(queries)
        h = self._h(hix).unsqueeze(1)
        b = self._b(rix).unsqueeze(1)
        t = self._t(tails).unsqueeze(0)

        hnew = (h * b).sum(dim=2)

        return self.model.interaction.score(h=hnew, r=torch.ones_like(hnew, dtype=self.dtype), t=t)

    def score_pi(self, queries: List[dict], tails: torch.LongTensor) -> torch.FloatTensor:
        hix, rix = self.unpack_pi_query_indices(queries)
        h = self._h(hix).unsqueeze(1)
        b = self._b(rix).unsqueeze(1)
        t = self._t(tails).unsqueeze(0)

        hnew = h[:,:,0,:] * b[:,:,:2,:].prod(dim=2) + h[:,:,1,:] * b[:,:,2,:]
        return self.model.interaction.score(h=hnew, r=torch.ones_like(hnew, dtype=self.dtype), t=t)

    def score_ip(self, queries: List[dict], tails: torch.LongTensor) -> torch.FloatTensor:
        hix, rix = self.unpack_pi_query_indices(queries)
        h = self._h(hix).unsqueeze(1)
        b = self._b(rix).unsqueeze(1)
        t = self._t(tails).unsqueeze(0)

        hnew = (h * b[:,:,:2,:]).sum(dim=2) * b[:,:,2,:]
        return self.model.interaction.score(h=hnew, r=torch.ones_like(hnew, dtype=self.dtype), t=t)
    
class TransRComplex(ComplexExtender):
    '''Harmonic extension for TransR model.'''
    def __init__(self,
                 model: ERModel) -> None:
        super().__init__(model=model)

    def _check_model_type(self):
        assert isinstance(self.model, TransR)

    def _restriction_maps(self, edge_type:torch.LongTensor) -> torch.Tensor:
        nu = torch.LongTensor([0])
        _, r, _ = self.model._get_representations(h=nu, r=edge_type, t=nu, mode=None)
        ret = torch.transpose(r[1],-1,-2)
        restriction_maps = torch.cat([ret.unsqueeze(2), ret.unsqueeze(2)], dim=2)
        return restriction_maps
    
    def _b(self, edge_type:torch.LongTensor) -> Tuple[torch.Tensor]:
        nu = torch.LongTensor([0])
        _, r, _ = self.model._get_representations(h=nu, r=edge_type, t=nu, mode=None)
        return r[0]

    def score_schur_batched(self, edge_index, restriction_maps, 
                            boundary_vertices, interior_vertices,
                            source_vertices, target_vertices, 
                            xS, xT, b, dv):
        LSchur, affine = bhe.Kron_reduction_translational(edge_index, restriction_maps, boundary_vertices, interior_vertices)
        return bhe.compute_costs_translational(LSchur, affine, source_vertices, target_vertices, xS, xT, b, dv)
                            
    def score_intersect_batched(self, edge_index, restriction_maps, 
                            source_vertices, target_vertices, 
                            xS, xT, b, dv):
        L = bhe.Laplacian(edge_index, restriction_maps)
        d = bhe.coboundary(edge_index, restriction_maps)
        return bhe.compute_costs_translational(L, d, source_vertices, target_vertices, xS, xT, b, dv)
    
    def score_2p(self, queries: List[dict], tails: torch.LongTensor) -> torch.FloatTensor:

        qs = self.query_structure_2p()
        hix, rix = self.unpack_path_query_indices(queries)
        h = self._h(hix)
        b = self._b(rix)
        t = self._t(tails).unsqueeze(-1)
        restriction_maps = self._restriction_maps(rix)

        nbatch = restriction_maps.shape[0]
        
        scores = self.score_schur_batched(qs['edge_index'], restriction_maps, 
                                            qs['boundary_vertices'], qs['interior_vertices'], 
                                            qs['source_vertices'], qs['target_vertices'], 
                                            h.reshape(nbatch, -1, 1), t, b.reshape(nbatch, -1, 1), t.shape[-2])
        return -torch.sum(scores, dim=(-1))
    
    def score_3p(self, queries: List[dict], tails: torch.LongTensor) -> torch.FloatTensor:
        qs = self.query_structure_3p()
        hix, rix = self.unpack_path_query_indices(queries)
        h = self._h(hix)
        b = self._b(rix)
        t = self._t(tails).unsqueeze(-1)
        restriction_maps = self._restriction_maps(rix)

        nbatch = restriction_maps.shape[0]
        
        scores = self.score_schur_batched(qs['edge_index'], restriction_maps, 
                                            qs['boundary_vertices'], qs['interior_vertices'], 
                                            qs['source_vertices'], qs['target_vertices'], 
                                            h.reshape(nbatch, -1, 1), t, b.reshape(nbatch, -1, 1), t.shape[-2])
        return -torch.sum(scores, dim=(-1))

    def score_2i(self, queries: List[dict], tails: torch.LongTensor) -> torch.FloatTensor:
        qs = self.query_structure_2i()
        hix, rix = self.unpack_intersection_query_indices(queries)
        h = self._h(hix)
        b = self._b(rix)
        t = self._t(tails).unsqueeze(-1)
        restriction_maps = self._restriction_maps(rix)

        nbatch = restriction_maps.shape[0]
        
        scores = self.score_intersect_batched(qs['edge_index'], restriction_maps,
                                            qs['source_vertices'], qs['target_vertices'], 
                                            h.reshape(nbatch, -1, 1), t, b.reshape(nbatch, -1, 1), t.shape[-2])
        return -torch.sum(scores, dim=(-1))
    
    def score_3i(self, queries: List[dict], tails: torch.LongTensor) -> torch.FloatTensor:
        qs = self.query_structure_3i()
        hix, rix = self.unpack_intersection_query_indices(queries)
        h = self._h(hix)
        b = self._b(rix)
        t = self._t(tails).unsqueeze(-1)
        restriction_maps = self._restriction_maps(rix)

        nbatch = restriction_maps.shape[0]
        
        scores = self.score_intersect_batched(qs['edge_index'], restriction_maps, 
                                            qs['source_vertices'], qs['target_vertices'], 
                                            h.reshape(nbatch, -1, 1), t, b.reshape(nbatch, -1, 1), t.shape[-2])
        return -torch.sum(scores, dim=(-1))

    def score_pi(self, queries: List[dict], tails: torch.LongTensor) -> torch.FloatTensor:
        qs = self.query_structure_pi()
        hix, rix = self.unpack_pi_query_indices(queries)
        h = self._h(hix)
        b = self._b(rix)
        t = self._t(tails).unsqueeze(-1)
        restriction_maps = self._restriction_maps(rix)

        nbatch = restriction_maps.shape[0]
        
        scores = self.score_schur_batched(qs['edge_index'], restriction_maps, 
                                        qs['boundary_vertices'], qs['interior_vertices'], 
                                        qs['source_vertices'], qs['target_vertices'], 
                                        h.reshape(nbatch, -1, 1), t, b.reshape(nbatch, -1, 1), t.shape[-2])
        return -torch.sum(scores, dim=(-1))

    def score_ip(self, queries: List[dict], tails: torch.LongTensor) -> torch.FloatTensor:
        qs = self.query_structure_ip()
        hix, rix = self.unpack_ip_query_indices(queries)
        h = self._h(hix)
        b = self._b(rix)
        t = self._t(tails).unsqueeze(-1)
        restriction_maps = self._restriction_maps(rix)

        nbatch = restriction_maps.shape[0]
        
        scores = self.score_schur_batched(qs['edge_index'], restriction_maps, 
                                        qs['boundary_vertices'], qs['interior_vertices'], 
                                        qs['source_vertices'], qs['target_vertices'], 
                                        h.reshape(nbatch, -1, 1), t, b.reshape(nbatch, -1, 1), t.shape[-2])
        return -torch.sum(scores, dim=(-1))

def get_complex_extender(model_type):
    if model_type == 'se':
        return SEComplex
    if model_type == 'transe':
        return TransEComplex
    if model_type == 'rotate':
        return RotatEComplex
    if model_type == 'transr':
        return TransRComplex