from copy import deepcopy
import random
from arg import parse, parse_stack
import data_utils
import models
from heterogeneous_LoRA_stack_glue import fl_stack_lora_train_glue_het
#['sst2', 'mnli', 'mrpc', 'cola', 'qqp', 'qnli', 'rte', 'stsb']
#args = parse()
args = parse_stack()

clients, valloader, testloader, test_batch = data_utils.build_dataset(
    args.dataset, args.client_batch, args.clients, args.iid_alpha, args.seed, args.eval_frac)

r = args.lora_r
alpha = args.lora_alpha

random_numbers = [random.random() for _ in range(args.clients)]

sketch_list = [0.125, 0.25, 0.5]
#sketch_list = [1]*3
k_list=[]

for i in range(args.clients):
    if 2/3 <= random_numbers[i] <=1:
        k_list.append(int(r*sketch_list[2]))
        #print(int(r*sketch_list[2]))
    elif 1/3 <= random_numbers[i] <2/3:
        k_list.append(int(r*sketch_list[1]))
        #print(int(r*sketch_list[1]))
    else:
        k_list.append(int(r*sketch_list[0]))
        #print(int(r*sketch_list[0]))
print('k_list:', k_list)


model = models.build_model(args.dataset)
fronzen_model = deepcopy(model)
total = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"(Parameters: {total})")

##########################
k_list = [r]*args.clients
##########################

fl_stack_lora_train_glue_het(args.dataset, fronzen_model, clients, testloader, test_batch,
    rounds=args.server_rounds,
    eval_freq=args.eval_freq,
    server_opt=args.server_opt,
    server_batch=args.server_batch,
    server_lr=args.server_lr,
    client_lr=args.client_lr,
    client_epochs=args.client_epochs,
    r = r,
    m_list = k_list
)

