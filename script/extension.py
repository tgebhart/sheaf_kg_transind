from itertools import product
from typing import List, Tuple, Literal

import scipy.sparse as sps
import torch
from models import SE
from pykeen.models.nbase import ERModel
from pykeen.models.unimodal.rotate import RotatE
from pykeen.models.unimodal.trans_e import TransE
from pykeen.models.unimodal.trans_r import TransR
from pykeen.utils import clamp_norm
from torch_geometric.utils import degree, k_hop_subgraph
from torch_scatter import scatter

ALPHA = 1e-1


class KGExtender:
    """Harmonic extension base class."""

    def __init__(
        self, model: ERModel, alpha: float = ALPHA, degree_normalize: bool = False
    ) -> None:
        self.model = model
        self._check_model_type()
        self.alpha = alpha
        self.degree_normalize = degree_normalize
        self.device = self.model.device

    def _check_model_type(self):
        raise NotImplementedError

    def _ht(self, edge_index: torch.LongTensor) -> Tuple[torch.Tensor]:
        nu = torch.LongTensor([0])
        xh, _, xt = self.model._get_representations(
            h=edge_index[0, :], r=nu, t=edge_index[1, :], mode=None
        )
        return xh, xt

    def _x(self, entities: torch.LongTensor) -> torch.Tensor:
        nu = torch.LongTensor([0])
        xh, _, _ = self.model._get_representations(h=entities, r=nu, t=nu, mode=None)
        return xh

    def laplacian_block_dense(
        self, edge_index: torch.LongTensor, edge_type: torch.LongTensor
    ):
        Fh, Ft = self._restriction_maps(edge_type)
        d = coboundary(edge_index, Fh, Ft)
        return torch.matmul(torch.transpose(d, 0, 1), d)

    def laplacian_block_sparse(
        self, edge_index: torch.LongTensor, edge_type: torch.LongTensor
    ):
        Fh, Ft = self._restriction_maps(edge_type)
        d = coboundary(edge_index, Fh, Ft)
        return torch.sparse.mm(torch.transpose(d, 0, 1), d)

    def update_representations(
        self, xU, interior_vertices_model, interior_vertices_diffused=None
    ):
        interior_vertices_diffused = (
            interior_vertices_model
            if interior_vertices_diffused is None
            else interior_vertices_diffused
        )
        self.model.entity_representations[0]._embeddings.weight[
            interior_vertices_model
        ] -= (self.alpha * xU[interior_vertices_diffused])

    def laplacian_UB_quad(
        self, edge_index: torch.LongTensor, edge_type: torch.LongTensor
    ):
        """TODO"""
        xh, xt = self._ht(edge_index)
        Fh, Ft = self._restriction_maps(edge_type)

        unique, edge_index_inv = torch.unique(
            edge_index, sorted=True, return_inverse=True
        )

        dx = Fh @ xh.unsqueeze(-1) - Ft @ xt.unsqueeze(-1)
        x_e_h = Fh.permute(0, -1, -2) @ dx
        x_e_t = Ft.permute(0, -1, -2) @ dx

        Lx = torch.zeros((unique.shape[0]))
        scatter(x_e_h.squeeze(-1), edge_index_inv[0, :], dim=0, out=Lx)
        scatter(-x_e_t.squeeze(-1), edge_index_inv[1, :], dim=0, out=Lx)

        return Lx


def coboundary(
    edge_index: torch.Tensor, Fh: torch.Tensor, Ft: torch.Tensor, relabel=False
) -> torch.Tensor:
    """Computes the coboundary matrix of the embedding of the
    relation?

    Args:
        edge_index (_type_): list representation of a graph.
        ex: [[0, 1], [1, 2]].T, where we have vertices 0,1,2 and edges 0-->1-->2
        Fh (torch.Tensor): restriction map from h -> e
        Ft (torch.Tensor): restriction map from t -> e
        relabel (bool, optional): _description_. Defaults to False.

    Returns:
        torch.Tensor: the coboundary matrix from C^0 to C^1.
    """
    device = Fh.device
    if relabel:
        _, edge_index = torch.unique(edge_index, sorted=True, return_inverse=True)
    ne = edge_index.shape[-1]  # num of edges
    nv = edge_index.max() + 1  # num of vertices
    de = Fh.shape[-2]  # dim of edge space
    dv = Fh.shape[-1]  # dim of vertex space
    idxs = []
    vals = torch.zeros(0, device=device)
    for e in range(ne):
        h = edge_index[0, e]
        t = edge_index[1, e]
        r = list(range(e * de, (e + 1) * de))
        idxs += list(product(r, list(range(h * dv, (h + 1) * dv)))) + list(
            product(r, list(range(t * dv, (t + 1) * dv)))
        )
        vals = torch.cat((vals, Fh[e, :, :].flatten(), -Ft[e, :, :].flatten()))
        # ok so here ^^ we're stacking the restriction maps, but
        # switching the 'orientation' of the map t --> e
    return torch.sparse_coo_tensor(
        torch.LongTensor(idxs).T, vals, size=(ne * de, nv * dv), device=device
    )
    # this ^^ allows us to specify the non-zero entries of the result that we want to populate with vals


def diffuse_interior(
    diffuser: KGExtender,
    triples: List[list],
    interior_ent_msk: torch.Tensor,
    batch_size: int = None,
):
    """
        Diffuse by application of the extension model's sheaf
        Laplacian. The terms 'extender' and 'diffuser' seem to be used interchangeably.

    Args:
        diffuser (KGExtender): The extension method to use.
        triples (List[list]): a list of entity - relation - entity triples, each of which is also a list.
        interior_ent_msk (torch.Tensor): indices of vertices in the interior
        batch_size (int, optional): Batch size. Defaults to None.

    Returns:
        _type_: _description_
    """
    edge_index = triples[:, [0, 2]].T
    relations = triples[:, 1]
    all_ents = edge_index.flatten().unique()
    num_nodes = all_ents.size(0)
    interior_vertices = all_ents[interior_ent_msk]

    if batch_size is None:
        batch_size = edge_index.shape[1]
    if batch_size > edge_index.shape[1]:
        batch_size = edge_index.shape[1]

    degree_normalize = diffuser.degree_normalize
    # we will normalize after all batches are processed, if required
    diffuser.degree_normalize = False

    xU = None
    for bix in range(0, edge_index.shape[1], batch_size):
        # multiply a batch of entity embeddings by the laplacian
        xUb, _ = diffuser.diffuse_interior(
            edge_index[:, bix : bix + batch_size],  # edges
            relations[bix : bix + batch_size],  # edge types
            nv=num_nodes,
        )
        if xU is None:
            xU = xUb
        else:
            xU += xUb

    if degree_normalize:
        degrees = xU.shape[1] * degree(edge_index.flatten().to(diffuser.device))
        xU = xU / degrees.reshape((-1, 1))
    diffuser.update_representations(xU, interior_vertices)
    diffuser.degree_normalize = degree_normalize
    return xU


def extend_interior(extender, triples, interior_ent_msk, batch_size=None):
    edge_index = triples[:, [0, 2]].T
    relations = triples[:, 1]
    all_ents = edge_index.flatten().unique()
    num_nodes = all_ents.size(0)
    interior_vertices = all_ents[interior_ent_msk]
    boundary_vertices = all_ents[~interior_ent_msk]

    interior_interior_msk = torch.isin(
        edge_index[0, :], interior_vertices
    ) & torch.isin(edge_index[1, :], interior_vertices)
    interior_boundary_msk = (
        torch.isin(edge_index[0, :], interior_vertices)
        & torch.isin(edge_index[1, :], boundary_vertices)
    ) | (
        torch.isin(edge_index[1, :], interior_vertices)
        & torch.isin(edge_index[0, :], boundary_vertices)
    )

    if batch_size is None:
        batch_size = edge_index.shape[1]
    if batch_size > edge_index.shape[1]:
        batch_size = edge_index.shape[1]

    xU = extender.harmonic_extension(
        edge_index,
        relations,
        interior_interior_msk,
        interior_boundary_msk,
        boundary_vertices,
    )


class SEExtender(KGExtender):
    """Harmonic extension for Structured Embedding model."""

    def __init__(
        self, model: ERModel, alpha: float = ALPHA, degree_normalize: bool = True
    ) -> None:
        super().__init__(model=model, alpha=alpha, degree_normalize=degree_normalize)

    def _check_model_type(self) -> None:
        assert isinstance(self.model, SE)

    def _restriction_maps(self, edge_type: torch.LongTensor) -> Tuple[torch.Tensor]:
        nu = torch.LongTensor([0])
        _, r, _ = self.model._get_representations(h=nu, r=edge_type, t=nu, mode=None)
        return r[0], r[1]

    def laplacian_mult(
        self,
        edge_index: torch.LongTensor,
        edge_type: torch.LongTensor,
        relabel: bool = False,
        nv: int = None,
    ):
        edge_index = edge_index.to(self.device)
        xh, xt = self._ht(edge_index)
        Fh, Ft = self._restriction_maps(edge_type)

        if relabel:
            _, edge_index = torch.unique(edge_index, return_inverse=True)
        if nv is None:
            nv = edge_index.max() + 1

        dx = Fh @ xh.unsqueeze(-1) - Ft @ xt.unsqueeze(-1)
        x_e_h = Fh.permute(0, -1, -2) @ dx
        x_e_t = Ft.permute(0, -1, -2) @ dx

        Lx = torch.zeros((nv, xh.shape[1])).to(self.device)
        scatter(x_e_h.squeeze(-1), edge_index[0, :], dim=0, out=Lx)
        scatter(-x_e_t.squeeze(-1), edge_index[1, :], dim=0, out=Lx)

        if self.degree_normalize:
            degrees = xh.shape[1] * degree(edge_index.flatten())
            Lx = Lx / degrees.reshape((-1, 1))

        return Lx, edge_index

    def diffuse_interior(
        self,
        edge_index: torch.LongTensor,
        edge_type: torch.LongTensor,
        relabel: bool = False,
        nv: int = None,
    ):
        return self.laplacian_mult(edge_index, edge_type, relabel=relabel, nv=nv)

    # try getting BU edges and UU edges first
    def harmonic_extension(
        self,
        edge_index: torch.LongTensor,
        edge_type: torch.LongTensor,
        interior_interior_mask: torch.BoolTensor,
        interior_boundary_mask: torch.BoolTensor,
        boundary_entities,
        solution_device="cpu",
    ) -> torch.Tensor:
        LUU = self.laplacian_block_sparse(
            edge_index[:, interior_interior_mask], edge_type[interior_interior_mask]
        )
        LUB = self.laplacian_block_sparse(
            edge_index[:, interior_boundary_mask], edge_type[interior_boundary_mask]
        )

        # move everything to cpu due to memory constraints
        xB = self._x(boundary_entities).to(solution_device)
        LUU_inv = torch.linalg.pinv(
            LUU.to(solution_device).to(solution_device).to_dense()
        )
        xU = -LUU_inv @ LUB.to(solution_device) @ xB
        return xU.to(self.device)


class TransEExtender(KGExtender):
    """Harmonic extension for TransE model."""

    def __init__(
        self, model: ERModel, alpha: float = ALPHA, degree_normalize: bool = True
    ) -> None:
        super().__init__(model=model, alpha=alpha, degree_normalize=degree_normalize)

    def _check_model_type(self):
        assert isinstance(self.model, TransE)

    def _b(self, edge_type: torch.LongTensor) -> Tuple[torch.Tensor]:
        nu = torch.LongTensor([0])
        _, r, _ = self.model._get_representations(h=nu, r=edge_type, t=nu, mode=None)
        return r

    def laplacian_mult_translational(
        self,
        edge_index: torch.LongTensor,
        edge_type: torch.LongTensor,
        relabel: bool = False,
        nv: int = None,
    ):
        edge_index = edge_index.to(self.device)
        xh, xt = self._ht(edge_index)
        b = self._b(edge_type)

        if relabel:
            _, edge_index = torch.unique(edge_index, return_inverse=True)
        if nv is None:
            nv = edge_index.max() + 1
        dx = xh.unsqueeze(-1) + b.unsqueeze(-1) - xt.unsqueeze(-1)
        x_e_h = dx
        x_e_t = dx

        Lx = torch.zeros((nv, xh.shape[1])).to(self.device)
        scatter(x_e_h.squeeze(-1), edge_index[0, :], dim=0, out=Lx)
        scatter(-x_e_t.squeeze(-1), edge_index[1, :], dim=0, out=Lx)

        if self.degree_normalize:
            degrees = xh.shape[1] * degree(edge_index.flatten())
            Lx = Lx / degrees.reshape((-1, 1))

        return Lx, edge_index

    def diffuse_interior(
        self,
        edge_index: torch.LongTensor,
        edge_type: torch.LongTensor,
        relabel: bool = False,
        nv: int = None,
    ):
        return self.laplacian_mult_translational(
            edge_index, edge_type, relabel=relabel, nv=nv
        )


class RotatEExtender(KGExtender):
    """Harmonic extension for RotatE model."""

    def __init__(
        self, model: ERModel, alpha: float = ALPHA, degree_normalize: bool = True
    ) -> None:
        super().__init__(model=model, alpha=alpha, degree_normalize=degree_normalize)

    def _check_model_type(self):
        assert isinstance(self.model, RotatE)

    def _b(self, edge_type: torch.LongTensor) -> Tuple[torch.Tensor]:
        nu = torch.LongTensor([0])
        _, r, _ = self.model._get_representations(h=nu, r=edge_type, t=nu, mode=None)
        return r

    def update_representations(
        self, xU, interior_vertices_model, interior_vertices_diffused=None
    ):
        interior_vertices_diffused = (
            interior_vertices_model
            if interior_vertices_diffused is None
            else interior_vertices_diffused
        )
        self.model.entity_representations[0]._embeddings.weight[
            interior_vertices_model
        ] -= torch.view_as_real(self.alpha * xU[interior_vertices_diffused]).reshape(
            (interior_vertices_diffused.shape[0], -1)
        )

    def laplacian_mult_translational(
        self,
        edge_index: torch.LongTensor,
        edge_type: torch.LongTensor,
        relabel: bool = False,
        nv: int = None,
    ):
        edge_index = edge_index.to(self.device)
        xh, xt = self._ht(edge_index)
        b = self._b(edge_type)

        if relabel:
            _, edge_index = torch.unique(edge_index, return_inverse=True)
        if nv is None:
            nv = edge_index.max() + 1
        dx = xh.unsqueeze(-1) * b.unsqueeze(-1) - xt.unsqueeze(-1)
        x_e_h = dx
        x_e_t = dx

        Lx = torch.zeros((nv, xh.shape[1]), dtype=dx.dtype).to(self.device)
        scatter(x_e_h.squeeze(-1), edge_index[0, :], dim=0, out=Lx)
        scatter(-x_e_t.squeeze(-1), edge_index[1, :], dim=0, out=Lx)

        if self.degree_normalize:
            degrees = xh.shape[1] * degree(edge_index.flatten())
            Lx = Lx / degrees.reshape((-1, 1))

        return Lx, edge_index

    def diffuse_interior(
        self,
        edge_index: torch.LongTensor,
        edge_type: torch.LongTensor,
        relabel: bool = False,
        nv: int = None,
    ):
        return self.laplacian_mult_translational(
            edge_index, edge_type, relabel=relabel, nv=nv
        )


class TransRExtender(KGExtender):
    """Harmonic extension for TransR model."""

    def __init__(
        self, model: ERModel, alpha: float = ALPHA, degree_normalize: bool = True
    ) -> None:
        super().__init__(model=model, alpha=alpha, degree_normalize=degree_normalize)

    def _check_model_type(self):
        assert isinstance(self.model, TransR)

    def _restriction_maps(self, edge_type: torch.LongTensor) -> Tuple[torch.Tensor]:
        nu = torch.LongTensor([0])
        _, r, _ = self.model._get_representations(h=nu, r=edge_type, t=nu, mode=None)
        ret = r[1].permute(0, -1, -2)
        return ret, ret

    def _b(self, edge_type: torch.LongTensor) -> Tuple[torch.Tensor]:
        nu = torch.LongTensor([0])
        _, r, _ = self.model._get_representations(h=nu, r=edge_type, t=nu, mode=None)
        return r[0]

    def laplacian_mult_translational(
        self,
        edge_index: torch.LongTensor,
        edge_type: torch.LongTensor,
        relabel: bool = False,
        nv: int = None,
    ):
        edge_index = edge_index.to(self.device)
        xh, xt = self._ht(edge_index)
        b = self._b(edge_type)
        Fh, Ft = self._restriction_maps(edge_type)

        if relabel:
            _, edge_index = torch.unique(edge_index, return_inverse=True)
        if nv is None:
            nv = edge_index.max() + 1
        dx = (
            clamp_norm(Fh @ xh.unsqueeze(-1), p=2, dim=-2, maxnorm=1)
            + b.unsqueeze(-1)
            - clamp_norm(Ft @ xt.unsqueeze(-1), p=2, dim=-2, maxnorm=1)
        )
        # dx = Fh @ xh.unsqueeze(-1) + b.unsqueeze(-1) - Ft @ xt.unsqueeze(-1)
        x_e_h = Fh.permute(0, -1, -2) @ dx
        x_e_t = Ft.permute(0, -1, -2) @ dx

        Lx = torch.zeros((nv, xh.shape[1])).to(self.device)
        scatter(x_e_h.squeeze(-1), edge_index[0, :], dim=0, out=Lx)
        scatter(-x_e_t.squeeze(-1), edge_index[1, :], dim=0, out=Lx)

        if self.degree_normalize:
            degrees = xh.shape[1] * degree(edge_index.flatten())
            Lx = Lx / degrees.reshape((-1, 1))

        return Lx, edge_index

    def diffuse_interior(
        self,
        edge_index: torch.LongTensor,
        edge_type: torch.LongTensor,
        relabel: bool = False,
        nv: int = None,
    ):
        return self.laplacian_mult_translational(
            edge_index, edge_type, relabel=relabel, nv=nv
        )


def get_extender(model_type: Literal['se', 'transe', 'rotate', 'transr']):
    if model_type == "se":
        return SEExtender
    if model_type == "transe":
        return TransEExtender
    if model_type == "rotate":
        return RotatEExtender
    if model_type == "transr":
        return TransRExtender
