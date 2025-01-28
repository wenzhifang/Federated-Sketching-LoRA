#!/bin/bash

datasets=('sst2' 'mrpc' 'cola' 'qnli' 'mnli' 'qqp')

for dataset in "${datasets[@]}"; do
    CUDA_VISIBLE_DEVICES=1 python main_svd_het_lora_glue.py --lora_r 16 --dataset "$dataset"
done
