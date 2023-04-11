# Complex Relational Inference with Missing Data through Generalized Harmonic Extension

## Introduction

## Getting started
To set up the environment, run the following command:

```bash
conda env create --file=environment.yml
conda activate harmonic_extension
```

or, if you prefer, you can 

### How to update the environment

It seems like its [good practice](https://stackoverflow.com/questions/48787250/set-up-virtualenv-using-a-requirements-txt-generated-by-conda) to include both requirements.txt and environment.yml, but that environment.yml can set up more information.

To update environment.yml, use
```bash
conda env export --from-history --name ghe > environment.yml
```

To update requirements.txt, use 
```bash 
pip freeze > requirements.txt
```

To sync your environment to new requirements, 

```bash 
conda env update --file environment.yml --prune
```

## Running experiments

Homophilic feature propagation
```bash
sh ./experiments/scripts/homophilic_fp.sh
```

You might have a way to do hyperparameter sweeps, but I'm impressed by the implementation [here](https://github.com/twitter-research/neural-sheaf-diffusion). They use this thing called wandb that just needs a config 





