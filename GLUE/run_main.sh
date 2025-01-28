#!/bin/bash

datasets=('sst2' 'mrpc' 'cola' 'qnli' 'mnli' 'qqp' 'rte')

for dataset in "${datasets[@]}"; do
    CUDA_VISIBLE_DEVICES=0 python main_slora_glue_het.py --lora_r 64 --dataset "$dataset"
done
