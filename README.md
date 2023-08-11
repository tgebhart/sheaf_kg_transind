# TRANSIND #

Code implementation for paper *Extending Transductive Knowledge Graph Embedding Models for Inductive Logical Relational Inference*.

## Requirements

This repository was developed using Python 3.11 and [Pytorch](https://pytorch.org/) (== 2.0.0).
Other versions of both packages have not been tested.
The code also makes heavy use of [Pykeen](https://pykeen.readthedocs.io/en/stable/) (==1.10.1) to train and evaluate knowledge graph embeddings.
All requirements and their versions are listed in `requirements.txt`. 
In particular, note that [pytorch-geometric](https://pytorch-geometric.readthedocs.io/en/latest/) and 
[torch-scatter](https://github.com/rusty1s/pytorch_scatter) are dependencies. 