import argparse

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu',    default="0",     type=str)
    parser.add_argument('--dataset',    default='20newsgroups',  type=str)
    parser.add_argument('--iid-alpha',  default=0.1,  type=float)
    parser.add_argument('--clients',    default=10,       type=int) #50, 20
    parser.add_argument('--seed',       default=0,          type=int)
    parser.add_argument('--eval-freq',  default=5,         type=int) #1 for stack 5 for others
    parser.add_argument('--eval-frac',  default=1,        type=float)
    #
    parser.add_argument('--server-opt',       default='adam',  type=str) #SGD
    parser.add_argument('--server-lr',        default=5e-4,    type=float)
    parser.add_argument('--server-batch',     default=10,       type=int) #
    parser.add_argument('--server-rounds',    default=200,      type=int) # 10 for stack 200 for others
    parser.add_argument('--client-lr',        default=5e-4,    type=float)
    parser.add_argument('--client-batch',     default=16,      type=int) 
    parser.add_argument('--client-epochs',    default=1,       type=int) #20 for stack 1 for others
    parser.add_argument('--lora_r',     default=16,      type=int)
    parser.add_argument('--lora-alpha', default=1, type=int)
    parser.add_argument('--sketching_ratio',     default=1,      type=float) # this will be set in the method file

    return parser.parse_args()
