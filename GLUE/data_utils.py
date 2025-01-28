import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
import torch.nn.functional as F


from sklearn.datasets import fetch_20newsgroups
import os
import json
import numpy as np
from scipy import stats
from PIL import Image
from train_utils import test_batch_cls, test_batch_nwp

from arg import parse
args = parse()

#os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

DATA = "." # defined twice, see below, may encounter some issue

def test_batch_ds(model, batch, dataset):
    if dataset == "reddit": 
        x = batch
        loss, stats = test_batch_nwp(model, x.to(device))
    elif dataset == "flair":
        x, y = batch
        loss, stats = test_batch_cls(model, x.to(device), y.to(device), multilabel=True)
    elif dataset == "cifar10" or dataset == "20newsgroups":
        x, y = batch
        loss, stats = test_batch_cls(model, x.to(device), y.to(device), multilabel=False)
    return loss, stats

def test_batch_glue(model, batch, dataset):
    """
    Compute loss and accuracy for a batch of data (for both training and evaluation).
    
    Args:
        model: The fine-tuned model.
        batch: A batch of data from the DataLoader (containing inputs and labels).
        task_name: The name of the GLUE task (e.g., 'sst2', 'mnli', 'mrpc', 'cola', 'qqp', 'qnli', 'rte', 'stsb').
        device: The device (CPU/GPU) to run the model on.

    Returns:
        loss: Computed loss for the batch.
        accuracy: Accuracy for classification tasks or equivalent metric for regression tasks.
    """
    # Unpack the batch: x contains the inputs (input_ids and attention_mask), y contains the labels
    input_ids, attention_mask, y = batch

    # Move inputs and labels to the correct device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    y = y.to(device)  # y is the ground truth labels

    # Create a dictionary of inputs (for models like AutoModelForSequenceClassification)
    inputs = {
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }

    # Forward pass: get logits
    outputs = model(**inputs)  # Pass the inputs to the model
    logits = outputs.logits

    # Initialize loss and accuracy
    loss = None
    correct = 0
    total = y.size(0)  # Total number of examples in the batch

    # Classification tasks: SST-2, MNLI, MRPC, CoLA, QQP, QNLI, RTE
    if dataset in ['sst2', 'mnli', 'mrpc', 'cola', 'qqp', 'qnli', 'rte']:
        # Loss: Cross-entropy for classification tasks
        loss = F.cross_entropy(logits, y)

        # Accuracy: Compare predicted class (argmax of logits) with true labels
        preds = torch.argmax(logits, dim=-1)  # Get the index of the max logit
        correct = (preds == y).sum().item()

    # Regression task: STS-B
    elif dataset == 'stsb':
        # Loss: Mean Squared Error (MSE) for regression
        loss = F.mse_loss(logits.squeeze(), y.float())

        # For regression tasks, we compute "accuracy" based on how close the predictions are to the true values
        correct = ((logits.squeeze() - y).abs() < 0.5).sum().item()  # Count how many predictions are within 0.5 of the true label

    # Compute accuracy (classification) or "correctness" for regression
    accuracy = correct / total if total > 0 else 0

    return loss, accuracy


def build_dataset(dataset, batch_size, n_clients, alpha=-1, seed=0, eval_frac=1):
    valset = None
    if dataset == 'cifar10':
        clients, valset, testset = build_cifar10(n_clients, alpha, seed)
        TEST_BATCH = 32
    elif dataset == '20newsgroups':
        clients, valset, testset = build_20newsgroups(n_clients, alpha, seed)
        TEST_BATCH = 16
        print('20newsgroups')
    elif dataset == 'reddit':
        clients, valset, testset = build_reddit(alpha, seed)
        TEST_BATCH = 16
    elif dataset in ['sst2', 'mnli', 'mrpc', 'cola', 'qqp', 'qnli', 'rte', 'stsb']:
        clients, valset, testset = build_glue(dataset, n_clients, alpha, seed)
        TEST_BATCH = 16
    else:
        print('unkown dataset')

    clientloaders = [DataLoader(client, batch_size=batch_size, shuffle=True, num_workers=0) for client in clients]
    if valset is not None:
        valloader = DataLoader(valset, batch_size=TEST_BATCH, shuffle=False, num_workers=1)
    else:
        valloader = None
    testloader = DataLoader(testset, batch_size=TEST_BATCH, shuffle=False, num_workers=1)
    def test_batch(model, batch):
        if dataset in ['sst2', 'mnli', 'mrpc', 'cola', 'qqp', 'qnli', 'rte', 'stsb']:
            return test_batch_glue(model, batch, dataset)
        else:
            return test_batch_ds(model, batch, dataset)
    

    return clientloaders, valloader, testloader, test_batch

def partition_iidmix(client_lens, p):
    total_lens = np.cumsum(client_lens)
    total_lens = total_lens - total_lens[0]
    clients = [ # keep first (1-p)*client_len examples
        np.arange(curr_idx,int(curr_idx+(1-p)*client_len)) for 
        client_len,curr_idx in zip(client_lens, total_lens)
    ]
    pool_idx = np.concatenate([
        # pool last p*client_len examples
        np.arange(int(curr_idx+(1-p)*client_len), curr_idx+client_len) for 
        client_len,curr_idx in zip(client_lens, total_lens)
    ])
    pool_idx = pool_idx[np.random.permutation(len(pool_idx))] # random shuffle
    S = int(len(pool_idx) / len(client_lens))
    for i,keep_idx in enumerate(clients):
        clients[i] = np.concatenate((keep_idx, pool_idx[S*i:S*(i+1)]))
    return clients

def partition_dirichlet(Y, n_clients, alpha, seed):
    clients = []
    ex_per_class = np.unique(Y, return_counts=True)[1]
    n_classes = len(ex_per_class)
    print(f"Found {n_classes} classes")
    rv_tr = stats.dirichlet.rvs(np.repeat(alpha, n_classes), size=n_clients, random_state=seed) 
    rv_tr = rv_tr / rv_tr.sum(axis=0)
    rv_tr = (rv_tr*ex_per_class).round().astype(int)
    class_to_idx = {i: np.where(Y == i)[0] for i in range(n_classes)}
    curr_start = np.zeros(n_classes).astype(int)
    for client_classes in rv_tr:
        curr_end = curr_start + client_classes
        client_idx = np.concatenate([class_to_idx[c][curr_start[c]:curr_end[c]] for c in range(n_classes)])
        curr_start = curr_end
        clients.append(client_idx)
        # will be empty subset if all examples have been exhausted
    return clients

def build_cifar10(n_clients, alpha, seed):
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        normalize,
    ])
    trainset = torchvision.datasets.CIFAR10(root=f"{DATA}/cifar10", train=True, download=True, transform=transform)
    N = len(trainset)
    trainidx = np.arange(0, int(N*0.8))
    Y_tr = np.array([trainset.targets[i] for i in trainidx])
    clientidx = partition_dirichlet(Y_tr, n_clients, alpha, seed)
    clients = [torch.utils.data.Subset(trainset, trainidx[cidx]) for cidx in clientidx]
    validx = np.arange(int(N*0.8), N)
    valset = torch.utils.data.Subset(trainset, validx)
    testset = torchvision.datasets.CIFAR10(root=f"{DATA}/cifar10", train=False, download=True, transform=test_transform)
    return clients, valset, testset

def build_20newsgroups(n_clients, alpha, seed):
    train_pt = f"{DATA}/20newsgroups/20newsgroups_train.pt"
    test_pt = f"{DATA}/20newsgroups/20newsgroups_test.pt"
    if not os.path.exists(train_pt) or not os.path.exists(test_pt):
        generate_20newsgroups_dump()
    tr_d = torch.load(train_pt)
    ev_d = torch.load(test_pt)
    trainset = list(zip(tr_d['X'], tr_d['Y']))
    testset = list(zip(ev_d['X'], ev_d['Y']))
    N = len(trainset)
    trainidx = np.arange(0, int(N*0.8))
    Y_tr = tr_d['Y'][trainidx]
    clientidx = partition_dirichlet(Y_tr, n_clients, alpha, seed)
    clients = [torch.utils.data.Subset(trainset, trainidx[cidx]) for cidx in clientidx]
    validx = np.arange(int(N*0.8), N)
    valset = torch.utils.data.Subset(trainset, validx)
    return clients, valset, testset

def generate_20newsgroups_dump():
    print("Generating 20newsgroups cache...")
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token_id = 50256
    ng_train = fetch_20newsgroups(subset='train')
    tr_X = torch.LongTensor([tokenizer.encode(x, max_length=128, padding='max_length', truncation=True) for x in ng_train['data']])

    ng_test = fetch_20newsgroups(subset='test')
    ev_X = torch.LongTensor([tokenizer.encode(x, max_length=128, padding='max_length', truncation=True) for x in ng_test['data']])

    tr_Y = torch.LongTensor(ng_train['target'])
    ev_Y = torch.LongTensor(ng_test['target'])

    os.makedirs(f"{DATA}/20newsgroups", exist_ok=True)
    torch.save({'X': tr_X, 'Y': tr_Y}, f"{DATA}/20newsgroups/20newsgroups_train.pt")
    torch.save({'X': ev_X, 'Y': ev_Y}, f"{DATA}/20newsgroups/20newsgroups_test.pt")

def build_reddit(alpha, seed):
    train_X = []
    with open(f"{DATA}/reddit/train_clients.json") as f:
        client_names = json.load(f)
        for client_name in client_names.keys():
            client_X = torch.load(f"{DATA}/reddit/cache/{client_name}.pt")['X']
            train_X.append(client_X)
    trainlen = int(len(train_X)*0.8)
    
    eval_X = train_X[trainlen:]
    eval_X = torch.cat(eval_X)
    eval_Y = [-1 for i in range(len(eval_X))]
    evalset = list(zip(eval_X,eval_Y))
    train_X[:trainlen]

    test_X = []
    with open(f"{DATA}/reddit/eval_clients.json") as f:
        client_names = json.load(f)
        for client_name in client_names.keys():
            client_X = torch.load(f"{DATA}/reddit/cache/{client_name}.pt")['X']
            test_X.append(client_X)
    test_X = torch.cat(test_X)
    test_Y = [-1 for i in range(len(test_X))]
    testset = list(zip(test_X,test_Y))

    assert 0 <= alpha <= 1, "For Reddit, set iid_alpha >= 0 (non-iid default) and <= 1 (iid)"
    client_idx = partition_iidmix([len(X) for X in train_X], alpha)
    train_X_flat = torch.cat(train_X)
    clients = [[(train_X_flat[i],-1) for i in idx] for idx in client_idx]
    return clients, evalset, testset

DATA = "./data"


def generate_glue_dump(task_name, model_name='roberta-base', max_length=128):
    print(f"Generating {task_name} cache using {model_name}...")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Assign pad token if missing (for GPT-like models)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load GLUE dataset
    dataset = load_dataset("glue", task_name)
    
    # Print dataset column names to check the structure
    print(f"Columns in training set: {dataset['train'].column_names}")
    available_splits = dataset.keys()
    print(f"Available splits: {available_splits}")

    # Determine the validation split based on task
    if task_name == 'mnli':
        validation_key = 'validation_matched'  # MNLI uses this validation set by default
    elif 'validation' in available_splits:
        validation_key = 'validation'  # Default case for most other GLUE tasks
    else:
        raise KeyError(f"No validation set found for {task_name}. Available splits: {available_splits}")

    # Define the preprocessing function based on the task structure
    def preprocess_function(examples):
        if 'premise' in examples and 'hypothesis' in examples:  # MNLI
            return tokenizer(examples['premise'], examples['hypothesis'], truncation=True, padding='max_length', max_length=max_length)
        elif 'sentence1' in examples and 'sentence2' in examples:  # MRPC, STS-B, RTE
            return tokenizer(examples['sentence1'], examples['sentence2'], truncation=True, padding='max_length', max_length=max_length)
        elif 'question1' in examples and 'question2' in examples:  # QQP
            return tokenizer(examples['question1'], examples['question2'], truncation=True, padding='max_length', max_length=max_length)
        elif 'question' in examples and 'sentence' in examples:  # QNLI
            return tokenizer(examples['question'], examples['sentence'], truncation=True, padding='max_length', max_length=max_length)
        elif 'sentence' in examples:  # CoLA, SST-2
            return tokenizer(examples['sentence'], truncation=True, padding='max_length', max_length=max_length)
        else:
            raise KeyError(f"Dataset does not contain expected fields for task {task_name}.")

    # Apply preprocessing to the training set
    tokenized_train = dataset['train'].map(preprocess_function, batched=True)
    
    # Apply preprocessing to the validation set
    tokenized_test = dataset[validation_key].map(preprocess_function, batched=True)

    # Convert inputs and attention masks to tensors for both training and validation
    tr_X = {
        'input_ids': torch.LongTensor(tokenized_train['input_ids']),
        'attention_mask': torch.LongTensor(tokenized_train['attention_mask'])
    }
    ev_X = {
        'input_ids': torch.LongTensor(tokenized_test['input_ids']),
        'attention_mask': torch.LongTensor(tokenized_test['attention_mask'])
    }

    # Convert labels to tensors
    if task_name == 'stsb':  # Regression task
        tr_Y = torch.FloatTensor(tokenized_train['label'])
        ev_Y = torch.FloatTensor(tokenized_test['label'])
    else:  # Classification tasks
        tr_Y = torch.LongTensor(tokenized_train['label'])
        ev_Y = torch.LongTensor(tokenized_test['label'])

    # Create directory to store the processed data
    os.makedirs(f"data/{task_name}", exist_ok=True)

    # Save tokenized dataset to disk
    torch.save({'X': tr_X, 'Y': tr_Y}, f"data/{task_name}/train.pt")
    torch.save({'X': ev_X, 'Y': ev_Y}, f"data/{task_name}/test.pt")

    print(f"Data for {task_name} saved successfully.")


def build_glue(dataset, n_clients, alpha, seed):
    train_pt = f"data/{dataset}/train.pt"
    test_pt = f"data/{dataset}/test.pt"

    if not os.path.exists(train_pt) or not os.path.exists(test_pt):
        generate_glue_dump(dataset)
    
    # Load the preprocessed datasets
    tr_d = torch.load(train_pt)
    ev_d = torch.load(test_pt)
    
    # Combine input tensors and labels into a dataset
    trainset = list(zip(tr_d['X']['input_ids'], tr_d['X']['attention_mask'], tr_d['Y']))
    testset = list(zip(ev_d['X']['input_ids'], ev_d['X']['attention_mask'], ev_d['Y']))
    
    # Split the training data for validation and partitioning among clients
    N = len(trainset)
    trainidx = np.arange(0, int(N * 0.8))  # 80% training data
    Y_tr = tr_d['Y'][trainidx]  # Use labels to partition clients
    clientidx = partition_dirichlet(Y_tr, n_clients, alpha, seed)

    # Partition data across clients
    clients = [torch.utils.data.Subset(trainset, trainidx[cidx]) for cidx in clientidx]
    
    # Validation set is the remaining 20%
    validx = np.arange(int(N * 0.8), N)
    valset = torch.utils.data.Subset(trainset, validx)

    return clients, valset, testset

