from typing import Any, ClassVar, Mapping, Optional

from class_resolver import Hint
from torch.nn import functional

from pykeen.models import ERModel
from pykeen.constants import DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE
from pykeen.nn.init import xavier_uniform_, xavier_uniform_norm_
from pykeen.nn.modules import SEInteraction
from pykeen.typing import Constrainer, Initializer
from pykeen.utils import clamp_norm


__all__ = [
    "SE",
]


class SE(ERModel):
    r"""An implementation of the Structured Embedding (SE) published by [bordes2011]_.

    SE applies role- and relation-specific projection matrices
    $\textbf{M}_{r}^{h}, \textbf{M}_{r}^{t} \in \mathbb{R}^{d \times d}$ to the head and tail
    entities' embeddings before computing their differences. Then, the $l_p$ norm is applied
    and the result is negated such that smaller differences are considered better.

    .. math::

        f(h, r, t) = - \|\textbf{M}_{r}^{h} \textbf{e}_h  - \textbf{M}_{r}^{t} \textbf{e}_t\|_p

    By employing different projections for the embeddings of the head and tail entities, SE explicitly differentiates
    the role of an entity as either the subject or object.
    ---
    name: Structured Embedding
    citation:
        author: Bordes
        year: 2011
        link: https://www.aaai.org/ocs/index.php/AAAI/AAAI11/paper/download/3659/3898
    """

    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default: ClassVar[Mapping[str, Any]] = dict(
        embedding_dim=DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE,
        scoring_fct_norm=dict(type=int, low=1, high=2),
    )

    def __init__(
        self,
        *,
        embedding_dim: int = 50,
        scoring_fct_norm: int = 1,
        entity_initializer: Hint[Initializer] = xavier_uniform_,
        entity_constrainer: Hint[Constrainer] = clamp_norm,
        entity_constrainer_kwargs: Optional[Mapping[str, Any]] = None,
        relation_initializer: Hint[Initializer] = xavier_uniform_norm_,
        relation_constrainer: Hint[Constrainer] = clamp_norm,
        **kwargs,
    ) -> None:
        r"""Initialize SE.

        :param embedding_dim: The entity embedding dimension $d$. Is usually $d \in [50, 300]$.
        :param scoring_fct_norm: The $l_p$ norm. Usually 1 for SE.
        :param entity_initializer: Entity initializer function. Defaults to :func:`pykeen.nn.init.xavier_uniform_`
        :param entity_constrainer: Entity constrainer function. Defaults to :func:`torch.nn.functional.normalize`
        :param entity_constrainer_kwargs: Keyword arguments to be used when calling the entity constrainer
        :param relation_initializer: Relation initializer function. Defaults to
            :func:`pykeen.nn.init.xavier_uniform_norm_`
        :param kwargs:
            Remaining keyword arguments to forward to :class:`pykeen.models.EntityEmbeddingModel`
        """
        super().__init__(
            interaction=SEInteraction(
                p=scoring_fct_norm,
                power_norm=False,
            ),
            entity_representations_kwargs=dict(
                shape=embedding_dim,
                initializer=entity_initializer,
                constrainer=entity_constrainer,
                constrainer_kwargs=dict(maxnorm=1.0, p=2, dim=-1),
            ),
            relation_representations_kwargs=[
                dict(
                    shape=(embedding_dim, embedding_dim),
                    initializer=relation_initializer,
                    constrainer=relation_constrainer,
                    constrainer_kwargs=dict(maxnorm=1.0, p=2, dim=-1)
                ),
                dict(
                    shape=(embedding_dim, embedding_dim),
                    initializer=relation_initializer,
                    constrainer=relation_constrainer,
                    constrainer_kwargs=dict(maxnorm=1.0, p=2, dim=-1)
                ),
            ],
            **kwargs,
        )