import argparse
import torch
import random
import numpy as np
import optuna
import os
import sys

from configs import add_task_parser, add_dataset_parser, add_optim_parser, add_gpu_parser
from utils.model_params import model_parser_dict

from utils.dataloader import load_data
from trainer.long_term_forecasting import LTF_Trainer

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

def objective(trial, args):
    """
    Single parameter selection.

    :param trial: An optuna parameter selection trial.
    :param args: Hyperparameters from 'argparse'.

    :return: The MSE/MAE result for each Optuna trial to record the best result.
    """
    # fixed random seed
    fix_seed = args.seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # parameters settings
    args.learning_rate = trial.suggest_float("learning_rate", 0.0001, 0.0016, step=0.00007)
    args.dropout = trial.suggest_float("dropout", 0.2, 0.4, step=0.05)
    # args.d_conv = trial.suggest_int("d_conv", 1, 4,step=1)
    # args.d_state = trial.suggest_int("d_state", 1, 32, step=1)

    if hasattr(args, "use_gcn") and args.use_gcn:
        topk = f'_topk{args.subgraph_size}'
        prop = '_prop{:.2f}'.format(args.prop_alpha)
        mlptype = f'_mlp{args.mlp_type}'
    else:
        topk, prop, mlptype = '', '', ''

    if hasattr(args, "use_gcn") and args.use_gcn:
        B = 'B' if args.gcn_bi else ''
        H = 'H' if args.gcn_heterogeneity else ''
        GCN = f'_gcn({B}{H}{args.subgraph_size}+{args.gcn_depth})'
    else:
        GCN = ''

    
    task = '{}({})_{}_{}{}_loss({})'.format(
        args.dataset_name,          # dataset
        args.task,
        args.seq_len,               # look-back window size
        args.pred_len,              # prediction horizons
        GCN,
        args.loss
    )

    setting = '{0}_bs{1}{2}{3}{4}{5}{6}{7}{8}{9}{10}'.format(
        args.model,                 # 0. model name
        args.batch_size,            # 1. batch-size
        f'_el{args.e_layers}' if hasattr(args, 'e_layers') else '',             # 2. layers of encoder
        f'_(dm{args.d_model}+df{args.d_ff})' if hasattr(args, 'd_model') else '',           # 3. the latent state dimension
        f'_nh{args.n_heads}' if hasattr(args, 'n_heads') else '',               # 4. multi-heads
        f'_(dc{args.d_conv}+ds{args.d_state})' if hasattr(args, 'd_conv') else '',          # 5. mamba block
        f'_dp{args.dropout:.2f}' if hasattr(args, 'dropout') else '',           # 6. dropout
        f'_(pl{args.patch_len}+st{args.stride})' if hasattr(args, 'patch_len') else '',     # 7. patching
        topk,
        mlptype,
        prop,
    )

    print('>>>>>> Args in experiment: <<<<<<')
    print(args)

    data, corr = load_data(args)
    engine = LTF_Trainer(args, task, setting, corr)

    if args.is_training == 1:
        engine.train(data=data)
        torch.cuda.empty_cache()
    if args.is_training >=0:
        mse = engine.test(test_loader=data['test_loader'])
        return mse
    else:
        engine.predict(pred_loader=data['pred_loader'])
        return -1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transformer & non-Transformer family for Time Series Forecasting & Anomaly Detection')

    parser_choice = None
    if len(sys.argv) > 1:
        print(sys.argv[1])
        for p in model_parser_dict[sys.argv[1]]:
            p(parser=parser)
    add_task_parser(parser=parser)
    add_dataset_parser(parser=parser)
    add_optim_parser(parser=parser)
    add_gpu_parser(parser=parser)

    args, model_name = parser.parse_known_args()
    args.model = model_name[0]
    
    # find usable GPUs, otherwise cpu
    if torch.cuda.is_available() and not args.use_cpu:
        args.device='cuda:{}'.format(args.gpu)
        # multi_gpu
        if args.use_multi_gpu:
            args.device_ids = [int(i) for i in args.device_ids.split(',')]
    else:
        args.device='cpu'

    search_space = {
        "weather": {"learning_rate": [4e-4, 1e-4, 2e-4, 4e-5, 1e-3]},
        "traffic": {"learning_rate": [2.4e-3, 1e-3, 1.4e-3, 2e-3]},
        "electricity": {"learning_rate": [8e-4, 4e-4, 1e-3, 2e-4]},
        "solar": {"learning_rate": [8e-4, 4e-4, 1e-3, 2e-3]},
        "ETTh1": {"learning_rate": [4e-4, 1e-4, 2e-4, 4e-5], "dropout": [0.1, 0.2, 0.0]},
        "ETTh2": {"learning_rate": [4e-4, 1e-4, 2e-4, 4e-5], "dropout": [0.1, 0.2, 0.0]},
        "ETTm1": {"learning_rate": [4e-4, 1e-4, 2e-4, 4e-5], "dropout": [0.1, 0.2, 0.0]},
        "ETTm2": {"learning_rate": [4e-4, 1e-4, 2e-4, 4e-5], "dropout": [0.1, 0.2, 0.0]},
    }

    # study = optuna.create_study(direction='minimize')
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.GridSampler(search_space[args.dataset_name]))
    # bind 'args' to 'objective' function
    objective_with_args = lambda trial: objective(trial, args)
    study.optimize(objective_with_args, n_trials=12)
    # print the best parameters and metrics
    print('Best parameters:', study.best_params)
    print('Best score:', study.best_value)
