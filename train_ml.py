import numpy as np
# import pandas as pd
import torch
import os
import torch.nn as nn
import random
import argparse
from pathlib import Path
from torch.utils.data import DataLoader
from model import Classifier
from data import TyphoonDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression,LogisticRegression
import time
import copy
from ga import GA
parser = argparse.ArgumentParser()
parser.add_argument(
        "--exp_name",
        type=Path,
        default="exp",
    )
## configuration
parser.add_argument("--do_train",action="store_true")
parser.add_argument("--do_test",action="store_true")
parser.add_argument("--ga",action="store_true")
parser.add_argument("--label", type=int, default=36, help="Which typhoon is used as test data")
parser.add_argument('--code', type=int, nargs='+')
parser.add_argument("--minmax",action="store_true")
## training
# parser.add_argument("--batch_size", type=int, default=128, help="The number of batch size")
parser.add_argument("--concat_n", type=int, default=1, help="The number of concat (>1)")
# parser.add_argument("--num_epoch", type=int, default=500, help="The number of training epochs")
# parser.add_argument("--patience", type=int, default=20, help="If no improvement in 'patience' epochs, early stop")
## GA
parser.add_argument("--max_node", type=int, default=3)
parser.add_argument("--bit_range", type=int, default=3)
parser.add_argument("--num_chrom", type=int, default=20)
parser.add_argument("--num_iter", type=int, default=100)
parser.add_argument("--rate_cross", type=float, default=0.8)
parser.add_argument("--rate_mutate", type=float, default=0.3)
## reproduciable
parser.add_argument("--seed", type=int, default=6666, help="set a random seed for reproducibility")

args = parser.parse_args()

def set_random_seed(seed):    
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# set_random_seed(args.seed)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(torch.cuda.get_device_name(device))

def decode(code,train_list,eval_list,label,args=None):
    """# Dataloader"""
    if len(code) != 7:
        raise ValueError("Code wrong",code)
    columns = ['Shihimen', 'Feitsui', 'TPB', 'inflow', 'outflow', 'Feitsui_outflow', 'Tide']
    select_columns = []
    for c,col in zip(code,columns):
        if c:
            select_columns.append(col)
    # all_typhoon = train_list.tolist()+eval_list.tolist()+[label]
    # train_set = TyphoonDataset(data_path = './完整數據集.csv',typhoon_ids=train_list.tolist(),concat_n=args.concat_n,split='train')
    # train_set.min_max_data(all_typhoon,selected_columns = select_columns,label='TPB_level',split = train_set.split)
    # train_set.fetch_data(train_set.typhoon_ids,selected_columns = select_columns,label='TPB_level',table='normalized_specified_typhoon_data')

    # valid_set = TyphoonDataset(data_path = './完整數據集.csv',typhoon_ids=eval_list.tolist(),concat_n=args.concat_n,split='valid')
    # valid_set.min_max_data(all_typhoon,selected_columns = select_columns,label='TPB_level')
    # valid_set.fetch_data(valid_set.typhoon_ids,selected_columns = select_columns,label='TPB_level',table='normalized_specified_typhoon_data')

    # test_set = TyphoonDataset(data_path = './完整數據集.csv',typhoon_ids=[label],concat_n=args.concat_n,split='test')
    # test_set.min_max_data(all_typhoon,selected_columns = select_columns,label='TPB_level')
    # test_set.fetch_data(test_set.typhoon_ids,selected_columns = select_columns,label='TPB_level',table='normalized_specified_typhoon_data')

    all_typhoon = train_list.tolist()+eval_list.tolist()+[label]
    train_set = TyphoonDataset(data_path = './完整數據集.csv',typhoon_ids=train_list.tolist(),concat_n=args.concat_n,split='train')
    # train_set.min_max_data(all_typhoon,selected_columns = select_columns,label='TPB_level',split = train_set.split)
    if args.minmax:
        train_set.fetch_data(train_set.typhoon_ids,selected_columns = select_columns,label='TPB_level',table='normalized_specified_typhoon_train')
    else:
        train_set.fetch_data(train_set.typhoon_ids,selected_columns = select_columns,label='TPB_level')
    
    # train_set.fetch_data(train_set.typhoon_ids,selected_columns = select_columns,label='TPB_level')
    # train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    # exit()
    # print('train_loader',len(train_loader))
    valid_set = TyphoonDataset(data_path = './完整數據集.csv',typhoon_ids=eval_list.tolist(),concat_n=args.concat_n,split='valid')
    # valid_set.min_max_data(all_typhoon,selected_columns = select_columns,label='TPB_level',split = valid_set.split)
    if args.minmax:
        valid_set.fetch_data(valid_set.typhoon_ids,selected_columns = select_columns,label='TPB_level',table='normalized_specified_typhoon_valid')
    else:
        valid_set.fetch_data(valid_set.typhoon_ids,selected_columns = select_columns,label='TPB_level')
    
    # valid_set.fetch_data(valid_set.typhoon_ids,selected_columns = select_columns,label='TPB_level')
    # valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    # print('valid_loader',len(valid_loader))
    test_set = TyphoonDataset(data_path = './完整數據集.csv',typhoon_ids=[label],concat_n=args.concat_n,split='test')
    # test_set.min_max_data(all_typhoon,selected_columns = select_columns,label='TPB_level',split = test_set.split)
    if args.minmax:
        test_set.fetch_data(test_set.typhoon_ids,selected_columns = select_columns,label='TPB_level',table='normalized_specified_typhoon_test')
    else:
        test_set.fetch_data(test_set.typhoon_ids,selected_columns = select_columns,label='TPB_level')
    
    # test_set.fetch_data(test_set.typhoon_ids,selected_columns = select_columns,label='TPB_level')
    # test_loader = DataLoader(test_set, batch_size=1, shuffle=False, pin_memory=True)

    return train_set,valid_set,test_set,len(select_columns) + 4 #time columns



def train(code,train_set,valid_set,args,decode):
    st = time.time()
    train_set,valid_set,test_set,input_dim = decode(code)
    model = LinearRegression(fit_intercept=True)
    """# Start Training"""
    x,y = train_set.get_all_data()
    nsamples, nx, ny = x.shape
    x_ = x.reshape((nsamples,nx*ny))
    # print(x.shape,y.shape)
    model.fit(x_, y)
    train_loss = model.score(x_, y)
    train_acc=train_loss

    x,y = valid_set.get_all_data()
    nsamples, nx, ny = x.shape
    x_ = x.reshape((nsamples,nx*ny))
    valid_loss = model.score(x_, y)
    valid_acc=valid_loss
    total_time = time.time()-st
    l={'code':code,
        'v-acc':valid_acc,
        't-acc':train_acc,
        'v-loss':valid_loss,
        't-loss':train_loss,
        'time':total_time,
        'best epoch':0,
        'model':copy.deepcopy(model)}
    
    return l

# print(pred.shape,target.shape)


def test(best,decode,args):
    code = best['code']
    model = best['model']
    with open(args.exp_name / f"log.txt","a") as f:
        f.write(f"=====Test and Plot=====\n")
        f.write(f"BEST {code}\n")
    train_set,valid_set,test_set,input_dim = decode(code)
    x,y = test_set.get_all_data()
    nsamples, nx, ny = x.shape
    x_ = x.reshape((nsamples,nx*ny))
    test_loss = model.score(x_, y)
    # valid_acc=test_loss
    pred = model.predict(x_)
    target = y
    fig, ax = plt.subplots(1,figsize=(32,12))
    ax.plot(np.arange(pred.shape[0]),pred,color='r',label='pred')
    ax.plot(np.arange(target.shape[0]),target,color='b',label='true')
    ax.set_title(f"{code} | valid-loss: {best['v-acc']}| test-loss: {test_loss}")
    ax.legend()
    plt.savefig(args.exp_name / './pred.png')

if __name__ == '__main__':
    args = parser.parse_args()
    os.makedirs(args.exp_name, exist_ok = True)
    set_random_seed(args.seed)
    with open(args.exp_name / f"log.txt","a") as f:
        f.write(str(args))
        f.write(f"\n")
    train_list = np.random.choice(list(set(np.arange(38))-set([args.label])), size=int(38*0.8),replace=False)
    eval_list = np.array(list(set(np.arange(38)) - set([args.label]) - set(train_list)))

    all_typhoon = train_list.tolist()+eval_list.tolist()+[args.label]
    columns = ['Shihimen', 'Feitsui', 'TPB', 'inflow', 'outflow', 'Feitsui_outflow', 'Tide']
    train_set = TyphoonDataset(data_path = './完整數據集.csv',typhoon_ids=train_list.tolist(),concat_n=args.concat_n,split='train')
    train_set.load_db(train_set.data_path)
    
    valid_set = TyphoonDataset(data_path = './完整數據集.csv',typhoon_ids=eval_list.tolist(),concat_n=args.concat_n,split='valid')
    
    test_set = TyphoonDataset(data_path = './完整數據集.csv',typhoon_ids=[args.label],concat_n=args.concat_n,split='test')
    
    if args.minmax:
        train_set.min_max_data(all_typhoon,selected_columns = columns,label='TPB_level',split = train_set.split)
        valid_set.min_max_data(all_typhoon,selected_columns = columns,label='TPB_level',split = valid_set.split)
        test_set.min_max_data(all_typhoon,selected_columns = columns,label='TPB_level',split = test_set.split)
    print('Train typhoon:',sorted(train_list))
    print('Valid typhoon:',sorted(eval_list))
    print('Test typhoon:', args.label)

    _decode = lambda x: decode(x,train_list=train_list,eval_list=eval_list,label=args.label,args=args)
    _train = lambda code,train_loader,valid_loader,args: train(code,train_loader,valid_loader,args,decode=_decode)
    if args.ga and args.do_train:
        print('GA')
        
        algo = GA(args.num_chrom,
                args.num_iter,
                args.rate_cross,
                args.rate_mutate,
                args.max_node,
                args.bit_range,
                None,
                None,
                _train,
                args)
        best = algo.genetic_algo()
    elif args.do_train:
        print('predefine code',args.code)
        best = {}
        best['code'] = args.code
        # train_loader,valid_loader,test_loader,input_dim = _decode(args.code)
        # best['model'] = Classifier(input_dim=input_dim, output_dim=1, hidden_layers=args.hidden_layers, hidden_dim=args.hidden_dim, dropout=args.dropout, m_type = args.m_type).to(device)
        best = _train(best['code'],None,None,args)
    else:
        print('load model')
        best = {}
        best['code'] = args.code
        train_loader,valid_loader,test_loader,input_dim = _decode(args.code)
        model = LinearRegression(fit_intercept=True)
        
    if args.do_test:
        test(best,_decode,args)