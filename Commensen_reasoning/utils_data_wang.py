from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
import numpy as np
import re
import random
import os
import torch
from arg import parse
args = parse()
base_model = args.base_model

# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

from huggingface_hub import login
login("")

tokenizer = AutoTokenizer.from_pretrained(base_model, token = True )
tokenizer.pad_token_id = 0
tokenizer.padding_side = "left"
def tokenize(prompt, cutoff_len):
    
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=cutoff_len,
        padding=False,
        return_tensors=None,
    )
    
    if (result["input_ids"][-1] != tokenizer.eos_token_id and len(result["input_ids"]) < cutoff_len):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()
    
    return result

def collate_fn_left(batch):
    # Extract input_ids, attention_mask, and labels from each item in the batch
    input_ids = [torch.tensor(x['input_ids']) for x in batch]
    attention_mask = [torch.tensor(x['attention_mask']) for x in batch]
    labels = [torch.tensor(x['labels']) for x in batch]

    # Flip sequences for left padding
    input_ids_flipped = [seq.flip(0) for seq in input_ids]
    attention_mask_flipped = [seq.flip(0) for seq in attention_mask]
    labels_flipped = [seq.flip(0) for seq in labels]

    # Pad sequences
    input_ids_padded = torch.nn.utils.rnn.pad_sequence(
        input_ids_flipped, 
        batch_first=True, 
        padding_value=tokenizer.pad_token_id
    ).flip(1)  # flip back after padding
    
    attention_mask_padded = torch.nn.utils.rnn.pad_sequence(
        attention_mask_flipped, 
        batch_first=True, 
        padding_value=0
    ).flip(1)  # flip back after padding
    
    labels_padded = torch.nn.utils.rnn.pad_sequence(
        labels_flipped, 
        batch_first=True, 
        padding_value=-100
    ).flip(1)  # flip back after padding

    return {
        "input_ids": input_ids_padded,
        "attention_mask": attention_mask_padded,
        "labels": labels_padded
    }

############################ widz ######################################


def build_datasets_widz(args, base_seed):
    clients = []
    for i in range(10):
        file_path = f"./data_wiz/10/local_training_{i}.json"
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"can not find dataset file : {file_path}")
        
        local_data = load_dataset("json", data_files=file_path)
        local_train_dataset = local_data["train"].map(generate_and_tokenize_prompt_widz)
        indices = list(range(len(local_train_dataset)))
        random.shuffle(indices)
        shuffled_local_trainset = local_train_dataset.select(indices)
        clients.append(DataLoader(
            shuffled_local_trainset,
            batch_size=args.client_batch,
                shuffle=False,  # No need to shuffle again here
                num_workers=4,
                collate_fn=collate_fn_left,
                pin_memory=True
        ))
    return clients


def generate_and_tokenize_prompt_widz(data_point, train_on_inputs=True):
    full_prompt = generate_prompt_widz(data_point)
    tokenized_full_prompt = tokenize(full_prompt, cutoff_len=512)
    
    return tokenized_full_prompt

def generate_prompt_widz(data_point):
    instruction = data_point["instruction"]
    response = data_point["output"]
    return f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n{response}"

############################ dolly ######################################

def build_datasets_dolly(args, base_seed):
    clients = []
    for i in range(10):
        file_path = f"./data_dolly/10/local_training_{i}.json"
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"can not find dataset file : {file_path}")
        
        local_data = load_dataset("json", data_files=file_path)
        local_train_dataset = local_data["train"].map(generate_and_tokenize_prompt_dolly)
        indices = list(range(len(local_train_dataset)))
        random.shuffle(indices)
        shuffled_local_trainset = local_train_dataset.select(indices)
        clients.append(DataLoader(
            shuffled_local_trainset,
            batch_size=args.client_batch,
                shuffle=False,  # No need to shuffle again here
                num_workers=4,
                collate_fn=collate_fn_left,
                pin_memory=True
        ))
    return clients


def generate_and_tokenize_prompt_dolly(data_point, train_on_inputs=True):
    full_prompt = generate_prompt_dolly(data_point)
    tokenized_full_prompt = tokenize(full_prompt, cutoff_len=512)
    return tokenized_full_prompt

def generate_prompt_dolly(data_point):
    instruction = data_point["instruction"]
    input = data_point["context"]
    response = data_point["response"]
    return f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{response}"



############################ alpaca ######################################

def build_datasets_alpaca(args, base_seed):
    data = load_dataset("tatsu-lab/alpaca")
    dataset = data["train"].map(generate_and_tokenize_prompt_alpaca)
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    shuffled_trainset = dataset.select(indices)
    clients = [DataLoader(
        shuffled_trainset.shard(num_shards=args.clients, index=i),
        batch_size=args.client_batch,
        shuffle=False,  # No need to shuffle again here
        num_workers=4,
        collate_fn=collate_fn_left,
        pin_memory=True) for i in range(args.clients)
               ]
    return clients


def generate_and_tokenize_prompt_alpaca(data_point):
    full_prompt = generate_prompt_alpaca(data_point)
    tokenized_full_prompt = tokenize(full_prompt, cutoff_len=512)
    return tokenized_full_prompt

def generate_prompt_alpaca(data_point):
    prompt = data_point["text"]
    return prompt

############################ evaluation mmlu ######################################


def build_datasets_eval(args):
    batch_size = args.test_batch
    file_path = './data_wiz/mmlu_test_1444.jsonl'
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"can not find dataset file : {file_path}")
    
    dataset = load_dataset("json", data_files=file_path)
    data = dataset["train"]
    
    # Ensure dataset size is no larger than 3000
    max_samples = 3000
    if len(data) > max_samples:
        data = Subset(data, range(max_samples))
    
    valloader = DataLoader(data, batch_size=batch_size, num_workers=4, pin_memory=True, shuffle=False)
    return valloader


def generate_prompt_eval(instruction, input):
    return f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\nThe answer is: "



                
                
