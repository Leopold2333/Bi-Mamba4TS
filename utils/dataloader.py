import os
import pandas as pd
import numpy as np
from utils.dataset import Dataset_Forecasting, Dataset_Forecasting_Solar
from torch.utils.data import DataLoader


# Partition training/valid/test set for specific datasets
def construct_borders(length:int, seq_len:int=336, pred_len:int=48, dataset_type:str='custom'):
    if dataset_type == 'ETTh1' or dataset_type == 'ETTh2':
        border1s = [0,              12 * 30 * 24 - seq_len,     12 * 30 * 24 + 4 * 30 * 24 - seq_len,   length - seq_len - pred_len]
        border2s = [12 * 30 * 24,   12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24,             length]
    elif dataset_type == 'ETTm1' or dataset_type == 'ETTm2':
        border1s = [0,                  12 * 30 * 24 * 4 - seq_len,         12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - seq_len,   length - seq_len - pred_len]
        border2s = [12 * 30 * 24 * 4,   12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4,             length]
    else:
        num_train = int(length * 0.7)
        num_test = int(length * 0.2)
        num_vali = length - num_train - num_test
        border1s = [0,          num_train - seq_len,    length - num_test - seq_len,    length - seq_len - pred_len]
        border2s = [num_train,  num_train + num_vali,   length,                         length]
    return border1s, border2s

def load_data(args):
    data_type_param = {
        'train':    { 'shuffle_flag': True, 'drop_last': True, 'batch_size': args.batch_size },
        'val':      { 'shuffle_flag': False, 'drop_last': True, 'batch_size': args.batch_size },
        'test':     { 'shuffle_flag': False, 'drop_last': True, 'batch_size': args.batch_size },
        'pred':     { 'shuffle_flag': False, 'drop_last': False, 'batch_size': 1 }
    }

    timeenc = 0 if args.embed != 'timeF' else 1
    data = {}
    # df_raw.columns: ['date', ...(other features), target feature]
    if args.dataset_name in ['weather', 'traffic', 'electricity', 'ETTh1', 'ETTh2', 'ETTm1', 'ETTm2']:
        df_raw = pd.read_csv(os.path.join(args.root_path, args.data_path))
        cols = list(df_raw.columns)
        cols.remove(args.target)
        cols.remove('date')
        cols = cols[0:args.enc_in-1]
        df_raw = df_raw[['date'] + cols + [args.target]]
    elif args.dataset_name == 'solar':
        df_raw = []
        with open(os.path.join(args.root_path, args.data_path), "r", encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip('\n').split(',')
                data_line = np.stack([float(i) for i in line])
                df_raw.append(data_line)
        df_raw = np.stack(df_raw, 0)
        df_raw = pd.DataFrame(df_raw)

    border1s, border2s = construct_borders(len(df_raw), args.seq_len, args.pred_len, args.dataset_type)
    for catagory in ['train', 'val', 'test', 'pred']:
        if args.dataset_name in ['weather', 'traffic', 'electricity', 'ETTh1', 'ETTh2', 'ETTm1', 'ETTm2']:
            data[catagory] = Dataset_Forecasting(
                raw_data=df_raw, 
                border1s=border1s, 
                border2s=border2s,
                flag=catagory,
                size=[args.seq_len, args.label_len, args.pred_len],
                task=args.task,
                target=args.target,
                timeenc=timeenc,
                freq=args.freq
            )
        elif args.dataset_name == 'solar':
            data[catagory] = Dataset_Forecasting_Solar(
                raw_data=df_raw, 
                border1s=border1s, 
                border2s=border2s,
                flag=catagory,
                size=[args.seq_len, args.label_len, args.pred_len],
                task=args.task
            )
        batch_size = 1 if catagory=='pred' else args.batch_size
        data[catagory+"_loader"] = DataLoader(
            data[catagory],
            batch_size=batch_size,
            shuffle=data_type_param[catagory]['shuffle_flag'],
            num_workers=args.num_workers,
            drop_last=data_type_param[catagory]['drop_last'],
            pin_memory=True if args.use_multi_gpu else False
        )

        print(catagory, len(data[catagory]))

    if (hasattr(args, 'use_gcn') and args.use_gcn) or (hasattr(args, 'SRA') and args.SRA):
        if args.dataset_name in ['weather', 'traffic', 'electricity', 'ETTh1', 'ETTh2', 'ETTm1', 'ETTm2']:
            df_train = df_raw[cols+[args.target]][border1s[0]:border2s[0]]
        else:
            df_train = df_raw[border1s[0]:border2s[0]]
        corr = df_train.corr(method=args.relation)
        return data, np.array(corr)
    else:
        return data, None
