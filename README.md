# TRANSIND #

Code implementation for paper *Extending Transductive Knowledge Graph Embedding Models for Inductive Logical Relational Inference*.

## Requirements

This repository was developed using Python 3.11 and [Pytorch](https://pytorch.org/) (== 2.0.0).
Other versions of both packages have not been tested.
The code also makes heavy use of [Pykeen](https://pykeen.readthedocs.io/en/stable/) (==1.10.1) to train and evaluate knowledge graph embeddings.
All requirements and their versions are listed in `requirements.txt`. 
In particular, note that [pytorch-geometric](https://pytorch-geometric.readthedocs.io/en/latest/) and 
[torch-scatter](https://github.com/rusty1s/pytorch_scatter) are dependencies. 

## Data

The data for the fully-inductive experiments on InductiveFB15k237 and InductiveWN18RR (as originally described in [Teru et al 2020](https://arxiv.org/abs/1911.06962)) can be loaded dynamically using [functionality from pykeen](https://pykeen.readthedocs.io/en/stable/api/pykeen.datasets.inductive.InductiveFB15k237.html#pykeen.datasets.inductive.InductiveFB15k237).

The semi-inductive data preparation for FB15k-237 (originally described in [Galkin et al 2022](https://arxiv.org/abs/2210.08008)) can be downloaded from [Zenodo](https://zenodo.org/record/7306046). 
Upon downloading, the code expects each ratio `r`'s data to be present in `data/fb15k-237/<r>`.

## Running

The experiments presented in the paper can be reproduced using the three runner scripts:

- `full_run_from_transductive_hpo.py`: (semi-inductive) train models transductively and tune hyperparameters transductively, extend representations to `fb15k-237` test graph (Galkin et al 2022), and perform logical query reasoning over 1p, 2p, 3p, 2i, 3i, ip, pi query structures. 
- `full_run_from_inductive_hpo.py`: (semi-inductive) train models transductively but tune hyperparameters after extending representations to `fb15k-237` validation graph (Galkin et al 2022) and reasoning over ip and pi queries within validation graph. Once hyperparameters are chosen, train transductive embeddings and extend representations to test graph, reason over 1p, 2p, 3p, 2i, 3i, ip, and pi query structures.
- `full_run_from_inductive_hpo_disjoint.py`: (inductive) train models transductively on a version split of InductiveFB15k237 or InductiveWN18RR (Teru et al 2020). Then extend these models to the validation graph within the split, choose step size and diffusion iterations, then extend representations to the test graph and evaluate knowledge graph completion task. 

An example run of each of these using the RotatE model can be found below

semi-inductive and logical reasoning, transductive hpo:
```
python script/full_run_from_transductive_hpo.py --hpo-config-name rotate_hpo --eval-graph test --dataset-pct 175;
```

semi-inductive and logical reasoning, inductive hpo:
```
python script/full_run_from_inductive_hpo.py --hpo-config-name rotate_hpo_extension --eval-graph test --dataset-pct 175;
```

inductive knowledge graph completion (InductiveFB15k237):
```
python script/full_run_from_inductive_hpo_disjoint.py --hpo-config-name rotate_hpo_extension_disjoint --eval-graph test --dataset InductiveFB15k237 --version v1;
```
inductive knowledge graph completion (InductiveWN18RR):
```
python script/full_run_from_inductive_hpo_disjoint.py --hpo-config-name rotate_hpo_extension_disjoint --eval-graph test --dataset InductiveWN18RR --version v2;
```

## Figures

The `notebooks` directory provides code for reproducing paper figures and tables. 