#!/bin/sh

python -m experiments.run \
    --dataset_name MixHopSynthetic\
    --model sage\
    --filling_method zero\
    --missing_rate 0.99\
    --homophily 0.1