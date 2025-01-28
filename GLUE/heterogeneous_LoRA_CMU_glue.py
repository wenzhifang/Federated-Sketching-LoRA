import torch
from tqdm import tqdm
from copy import deepcopy
import random
import models
from arg import parse
from data_utils import *
from train_utils import *

args = parse()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)


def fl_cmu_lora_train_glue_het(dataset, server_model, fronzen_model, clients, testloader, test_batch,
             rounds, eval_freq, server_opt,
             server_batch, server_lr, client_lr, client_epochs, r, m_list):

    pbar = tqdm(range(rounds))

    server_params = {}
    for n, p in server_model.named_parameters():
        if p.requires_grad == True:
            server_params[n] = p
    

    if server_opt == 'sgd':
        server_opt = torch.optim.SGD(server_params.values(), lr=server_lr)
    elif server_opt == 'adam':
        server_opt = torch.optim.AdamW(server_params.values(), lr=server_lr)
    else:
        raise ValueError()
    sched = torch.optim.lr_scheduler.StepLR(server_opt, step_size=1, gamma=1)

    def eval_loop(model, loader):
        model.eval()
        stats_acc = []
        stats_loss = []

        for batch in loader:
            with torch.no_grad():
                loss, accu = test_batch(model, batch)
                stats_acc.append(accu)
                stats_loss.append(loss)
        model.train()

        return sum(stats_loss)/len(stats_loss), sum(stats_acc)/len(stats_acc)

    server_model = server_model.to(device)

    log_training = './training_log/train_CMU_hetLoRA_Glue_{}_r{:1d}.txt'.format(dataset, r)
    with open(log_training, 'w') as log_file:
        # Optionally write an initial message or leave it empty to just clear the file
        log_file.write('Starting: {}__r{:1d} \n'.format(dataset, r))
    
    eval_loss, eval_accu = eval_loop(server_model, testloader)
    log_eval = './training_log/eval_CMU_hetLoRA_Glue_{}_r{:1d}.txt'.format(dataset, r)
    with open(log_eval, 'w') as log_file:
        # Optionally write an initial message or leave it empty to just clear the file
        log_file.write('Starting: {}_r{:1d} \n'.format(dataset, r))
        log_file.write('Round {:3d}, eval accuracy {:.3f}\n'.format(0, eval_accu))

    for rnd in pbar:
        aggregate = None
        norm_sum = None
        client_ids = torch.randperm(len(clients))[:server_batch]
        for i,client_id in enumerate(client_ids):
            #client_model = deepcopy(server_model)
            #client_model.to(device)
            #'''
            client_model = deepcopy(fronzen_model)
            models.add_adapters_dataset(dataset, client_model, lora_rank=m_list[client_id], lora_alpha=args.lora_alpha)
            # initialize client model with server params                    
            client_params = {n: cp for n, cp in client_model.named_parameters() if cp.requires_grad==True}
            for n, sp in server_params.items():
                if "lora_B" in n: # LoRA module: truncate server's weights for the client
                    _, r_s = sp.shape  # Server rank
                    r_c = m_list[client_id]  # Client rank
                    if r_c <= r_s:
                        client_params[n].data = sp[:, :r_c].clone()
                    else:
                        raise ValueError(f"Client LoRA rank ({r_c}) exceeds server LoRA rank ({r_s}).")
                elif "lora_A" in n:
                    r_s, _ = sp.shape  # Server rank
                    r_c = m_list[client_id]  # Client rank
                    if r_c <= r_s:
                        client_params[n].data = sp[:r_c, :].clone()
                    else:
                        raise ValueError(f"Client LoRA rank ({r_c}) exceeds server LoRA rank ({r_s}).")
                else:
                    # Final linear layer or other trainable parameters: directly copy
                    client_params[n].data = sp.clone()
            client_model.to(device)
            #'''
            # Local Training
            client_opt = torch.optim.SGD(client_model.parameters(), lr=client_lr, momentum=0.9)
            client_loader = clients[client_id]        
            for epoch in range(client_epochs):
                for batch in client_loader:
                    loss, _ = test_batch(client_model, batch)
                    client_opt.zero_grad()
                    loss.backward()
                    client_opt.step()
            
            norm_sum, aggregate = delta_average(client_model, server_params, norm_sum, aggregate, r, device) 

            #norm_sum, aggregate, aggregate_weight = weighted_delta_average(client_model, server_params, norm_sum, aggregate, r, device)   
            #print(norm_sum)
            #print(i) 
        # Server model update
        #for n, sp in server_params.items():
            #sp.data = aggregate_weight[n]/norm_sum[n]
        server_opt.zero_grad()
        for n, sp in server_params.items():
            sp.grad = aggregate[n] / norm_sum[n]
        server_opt.step()
        sched.step()
       
            

        # Eval and Logging
        if (rnd+1) % eval_freq == 0:
            print("evaluation")
            #eval_model = deepcopy(server_model)
            eval_model = deepcopy(server_model).to(device)
            eval_loss, eval_accu = eval_loop(eval_model, testloader)
            with open(log_eval, 'a') as log_file:
                log_file.write('Round {:3d}, eval loss {:.3f}, eval accuracy {:.3f}\n'.format(rnd, eval_loss, eval_accu))

        pbar.set_description(f"eval: {eval_loss, eval_accu}")
