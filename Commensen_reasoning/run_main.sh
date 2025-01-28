#!/bin/bash
export TOKENIZERS_PARALLELISM=true
r=64
ratio=0.015625
algorithm="slora_r64_ratio_1_64"

#accelerate launch --multi_gpu evaluation_par.py --algorithm "slora" --dataset 'hellaswag'
#accelerate launch --multi_gpu main.py --client-batch 4 --lora_r $r --sketching_ratio $ratio --num_epochs 15 --local_iter_per_round 20 --dataset 'commensense' --algorithm $algorithm
#accelerate launch --multi_gpu main_CMU.py --client-batch 4 --lora_r 64 --num_epochs 15 --local_iter_per_round 20 --dataset 'commensense'
# accelerate launch --multi_gpu main_SVD.py --client-batch 4 --lora_r 64 --num_epochs 15 --local_iter_per_round 20 --dataset 'commensense'
#accelerate launch --multi_gpu main_STACK.py --client-batch 4 --lora_r 64 --num_epochs 10 --local_iter_per_round 200 --dataset 'commensense'

datasets=("boolq" "winogrande" "openbookqa" "ARC-Easy" "ARC-Challenge" "social_i_qa" "piqa" "hellaswag")

for dataset in "${datasets[@]}"; do
    echo "Running evaluation for dataset: $dataset"
    accelerate launch --multi_gpu evaluation_par.py --algorithm $algorithm --dataset "$dataset"
done

#datasets=("boolq" "winogrande" "openbookqa" "ARC-Easy" "ARC-Challenge" "social_i_qa" "piqa" "hellaswag")
# datasets=("hellaswag")
# for dataset in "${datasets[@]}"; do
#     echo "Running evaluation for dataset: $dataset"
#     accelerate launch --multi_gpu evaluation_par.py --algorithm "slora_r64_ratio_0125" --dataset "$dataset"
# done

#accelerate launch --multi_gpu evaluate_stack.py --algorithm "stack"