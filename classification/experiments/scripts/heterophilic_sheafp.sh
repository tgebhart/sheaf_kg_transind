#!/bin/sh

python -m experiments.run \
    --dataset_name MixHopSynthetic\
    --model sage\
    --filling_method sheaf_propagation\
    --missing_rate 0.99\
    --homophily 0.1\
    --n_runs 1