from arg import parse
import data_utils
import models
from LoRA_glue import fl_lora_train_glue
args = parse()

clients, valloader, testloader, test_batch = data_utils.build_dataset(
    args.dataset, args.client_batch, args.clients, args.iid_alpha, args.seed, args.eval_frac)

r = args.lora_r
alpha = args.lora_alpha

model = models.build_model(args.dataset)
total = sum(p.numel() for p in model.parameters() if p.requires_grad)
### Adding adapter
models.add_adapters_dataset(args.dataset, model, lora_rank=r, lora_alpha=alpha)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Training {trainable} parameters ({100*trainable/total:.2f}% of original {total})")

fl_lora_train_glue(args.dataset, model, clients, testloader, test_batch,
    rounds=args.server_rounds,
    eval_freq=args.eval_freq,
    server_opt=args.server_opt,
    server_batch=args.server_batch,
    server_lr=args.server_lr,
    client_lr=args.client_lr,
    client_epochs=args.client_epochs,
    r = r
)
