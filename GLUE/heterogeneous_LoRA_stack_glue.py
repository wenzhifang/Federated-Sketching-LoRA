import torch
from tqdm import tqdm
from copy import deepcopy
import random
import models
from arg import parse, parse_stack
from data_utils import *
from train_utils import *

#args = parse()
args = parse_stack()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)


def fl_stack_lora_train_glue_het(dataset, fronzen_model, clients, testloader, test_batch,
             rounds, eval_freq, server_opt,
             server_batch, server_lr, client_lr, client_epochs, r, m_list):

    pbar = tqdm(range(rounds))

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

    fronzen_model = fronzen_model.to(device)

    log_training = './training_log/train_Stack_hetLoRA_Glue_{}_r{:1d}.txt'.format(dataset, r)
    with open(log_training, 'w') as log_file:
        # Optionally write an initial message or leave it empty to just clear the file
        log_file.write('Starting: {}__r{:1d} \n'.format(dataset, r))
    
    eval_loss, eval_accu = eval_loop(fronzen_model, testloader)
    log_eval = './training_log/eval_Stack_hetLoRA_Glue_{}_r{:1d}.txt'.format(dataset, r)
    with open(log_eval, 'w') as log_file:
        # Optionally write an initial message or leave it empty to just clear the file
        log_file.write('Starting: {}_r{:1d} \n'.format(dataset, r))
        log_file.write('Round {:3d}, eval accuracy {:.3f}\n'.format(0, eval_accu))
        
    for rnd in pbar:
        aggregate = None
        client_ids = torch.randperm(len(clients))[:server_batch]
        for i,client_id in enumerate(client_ids):
            client_model = deepcopy(fronzen_model)
            models.add_adapters_dataset(dataset, client_model, lora_rank=m_list[client_id], lora_alpha=args.lora_alpha)
            client_model.to(device)
            
            # Local Training
            #client_opt = torch.optim.SGD(client_model.parameters(), lr=client_lr, momentum=0.9)
            client_opt = torch.optim.AdamW(client_model.parameters(), lr=client_lr)
            client_loader = clients[client_id]        
            for epoch in range(client_epochs):
                for batch in client_loader:
                    loss, _ = test_batch(client_model, batch)
                    client_opt.zero_grad()
                    loss.backward()
                    client_opt.step()
            
            aggregate = weight_aggregation_rlora(client_model, aggregate, m_list[client_id], device) 
        merge_lora_adapeters_into_original_model(fronzen_model, aggregate, server_batch) # update fronzen model        

        # Eval and Logging
        if (rnd+1) % eval_freq == 0:
            print("evaluation")
            #eval_loss_client, eval_accu_client = eval_loop(client_model, testloader)
            eval_loss, eval_accu = eval_loop(fronzen_model, testloader)
            with open(log_eval, 'a') as log_file:
                log_file.write('Round {:3d}, eval loss {:.3f}, eval accuracy {:.3f}\n'.format(rnd, eval_loss, eval_accu))
                #log_file.write('Round {:3d}, client loss {:.3f}, client accuracy {:.3f}\n'.format(rnd, eval_loss_client, eval_accu_client))

        pbar.set_description(f"eval: {eval_loss, eval_accu}")
