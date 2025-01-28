#!/bin/bash

datasets=('sst2' 'mrpc' 'cola' 'qnli' 'mnli' 'qqp')

for dataset in "${datasets[@]}"; do
    CUDA_VISIBLE_DEVICES=0 python main_stack_het_lora_glue.py --lora_r 64 --dataset "$dataset"
done
