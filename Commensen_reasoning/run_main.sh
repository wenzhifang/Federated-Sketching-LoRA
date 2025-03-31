#!/bin/bash
#source ~/.bashrc
#conda activate llama_slora

export TOKENIZERS_PARALLELISM=true
rank_type="normal"

accelerate launch --num_processes 4 main.py --client-batch 4 --lora_r 64 --clients 10 --num_epochs 15 --num_comm_rounds 30 --local_iter_per_round 20 --rank_type $rank_type
