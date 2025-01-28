import torch
from tqdm import tqdm
from copy import deepcopy
import random
from arg import parse
from data_utils import *

args = parse()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)


def fl_lora_train_glue(dataset, server_model, clients, testloader, test_batch,
             rounds, eval_freq, server_opt,
             server_batch, server_lr, client_lr, client_epochs, r):

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

    log_training = './training_log/train_LoRA_Glue_{}_r{:1d}.txt'.format(dataset, r)
    with open(log_training, 'w') as log_file:
        # Optionally write an initial message or leave it empty to just clear the file
        log_file.write('Starting: {}__r{:1d} \n'.format(dataset, r))
    
    eval_loss, eval_accu = eval_loop(server_model, testloader)
    log_eval = './training_log/eval_LoRA_Glue_{}_r{:1d}.txt'.format(dataset, r)
    with open(log_eval, 'w') as log_file:
        # Optionally write an initial message or leave it empty to just clear the file
        log_file.write('Starting: {}_r{:1d} \n'.format(dataset, r))
        log_file.write('Round {:3d}, eval accuracy {:.3f}\n'.format(0, eval_accu))

    for rnd in pbar:
        aggregate = None
        stats_acc = {}
        client_ids = torch.randperm(len(clients))[:server_batch]
        for i,client_id in enumerate(client_ids):

            client_model = deepcopy(server_model)
            client_model.to(device)

            # Local Training
            client_opt = torch.optim.SGD(client_model.parameters(), lr=client_lr, momentum=0.9)
            client_loader = clients[client_id]

            for epoch in range(client_epochs):
                for batch in client_loader:
                    loss, _ = test_batch(client_model, batch)
                    client_opt.zero_grad()
                    loss.backward()
                    client_opt.step()
            
            neg_client_delta = {n: server_params[n].data - cp.data for n,cp
                                    in client_model.named_parameters() if cp.requires_grad}

            # Aggregation
            if aggregate is None:
                aggregate = neg_client_delta
            else:
                for n, delta in neg_client_delta.items():
                    aggregate[n] += delta
            # Log last iteration
            
        # training state after one round of A or B
        '''
        loss = 
        accu = 
        with open(log_training, 'a') as log_file:
            log_file.write('Round {:3d}, training loss {:.3f}, testing accuracy {:.3f}\n'.format(rnd, loss, accu))
        '''

        # Server model update
        server_opt.zero_grad()
        for n, sp in server_params.items():
            sp.grad = aggregate[n] / server_batch
        server_opt.step()
        sched.step()

        # Eval and Logging
        if (rnd+1) % eval_freq == 0:
            eval_model = deepcopy(server_model)
            eval_loss, eval_accu = eval_loop(eval_model, testloader)
            with open(log_eval, 'a') as log_file:
                log_file.write('Round {:3d}, eval loss {:.3f}, eval accuracy {:.3f}\n'.format(rnd, eval_loss, eval_accu))

        pbar.set_description(f"eval: {eval_loss, eval_accu}")
