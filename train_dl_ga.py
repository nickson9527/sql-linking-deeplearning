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
import time
from tqdm import tqdm, trange
# from deepcopy import copy
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
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--batch_size", type=int, default=128, help="The number of batch size")
parser.add_argument("--concat_n", type=int, default=1, help="The number of concat (>1)")
parser.add_argument("--num_epoch", type=int, default=500, help="The number of training epochs")
parser.add_argument("--patience", type=int, default=20, help="If no improvement in 'patience' epochs, early stop")
## model
parser.add_argument("--hidden_layers", type=int, default=4, help="The number of hidden layers of model")
parser.add_argument("--hidden_dim", type=int, default=512, help="The number of hidden dimensions of model")
parser.add_argument("--nhead", type=int, default=64, help="The number of head in Transformer")
parser.add_argument("--dropout", type=float, default=0.3)
parser.add_argument("--m_type", type=str, default="lstm")
## GA
parser.add_argument("--max_node", type=int, default=3)
parser.add_argument("--bit_range", type=int, default=3)
parser.add_argument("--num_chrom", type=int, default=20)
parser.add_argument("--num_iter", type=int, default=100)
parser.add_argument("--rate_cross", type=float, default=0.8)
parser.add_argument("--rate_mutate", type=float, default=0.3)
## reproduciable
parser.add_argument("--seed", type=int, default=6666, help="set a random seed for reproducibility")



def set_random_seed(seed):    
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(torch.cuda.get_device_name(device))

# evaluation function for feature selection
def decode(code,train_list,eval_list,label,args=None):
    """# Dataloader"""
    if len(code) != 7:
        raise ValueError("Code wrong",code)
    columns = ['Shihimen', 'Feitsui', 'TPB', 'inflow', 'outflow', 'Feitsui_outflow', 'Tide']
    select_columns = []
    for c,col in zip(code,columns):
        if c:
            select_columns.append(col)
    all_typhoon = train_list.tolist()+eval_list.tolist()+[label]
    train_set = TyphoonDataset(data_path = './完整數據集.csv',typhoon_ids=train_list.tolist(),concat_n=args.concat_n,split='train')
    # train_set.min_max_data(all_typhoon,selected_columns = select_columns,label='TPB_level',split = train_set.split)
    if args.minmax:
        train_set.fetch_data(train_set.typhoon_ids,selected_columns = select_columns,label='TPB_level',table='normalized_specified_typhoon_train')
    else:
        train_set.fetch_data(train_set.typhoon_ids,selected_columns = select_columns,label='TPB_level')
    # train_set.fetch_data(train_set.typhoon_ids,selected_columns = select_columns,label='TPB_level')
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    # exit()
    # print('train_loader',len(train_loader))
    valid_set = TyphoonDataset(data_path = './完整數據集.csv',typhoon_ids=eval_list.tolist(),concat_n=args.concat_n,split='valid')
    # valid_set.min_max_data(all_typhoon,selected_columns = select_columns,label='TPB_level',split = valid_set.split)
    if args.minmax:
        valid_set.fetch_data(valid_set.typhoon_ids,selected_columns = select_columns,label='TPB_level',table='normalized_specified_typhoon_valid')
    else:
        valid_set.fetch_data(valid_set.typhoon_ids,selected_columns = select_columns,label='TPB_level')
    # valid_set.fetch_data(valid_set.typhoon_ids,selected_columns = select_columns,label='TPB_level')
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    # print('valid_loader',len(valid_loader))
    test_set = TyphoonDataset(data_path = './完整數據集.csv',typhoon_ids=[label],concat_n=args.concat_n,split='test')
    # test_set.min_max_data(all_typhoon,selected_columns = select_columns,label='TPB_level',split = test_set.split)
    if args.minmax:
        test_set.fetch_data(test_set.typhoon_ids,selected_columns = select_columns,label='TPB_level',table='normalized_specified_typhoon_test')
    else:
        test_set.fetch_data(test_set.typhoon_ids,selected_columns = select_columns,label='TPB_level')
    # test_set.fetch_data(test_set.typhoon_ids,selected_columns = select_columns,label='TPB_level')
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, pin_memory=True)
    # print('test_loader',len(test_loader))
    # pass
    # train_set.correlation_coefficient(train_set.typhoon_ids,selected_columns = select_columns,label='TPB_level')
    # train_set._correlation_coefficient(train_set.typhoon_ids,selected_columns = select_columns,label='TPB_level')
    
    # exit()
    return train_loader,valid_loader,test_loader,len(select_columns) + 4 #time columns

def train(code,train_loader,valid_loader,args,decode):
    """# Start Training"""
    st = time.time()
    train_loader,valid_loader,test_loader,input_dim = decode(code)
    # print()
    model = Classifier(input_dim=input_dim, 
                       output_dim=1, 
                       hidden_layers=args.hidden_layers, 
                       hidden_dim=args.hidden_dim, 
                       dropout=args.dropout, 
                       m_type = args.m_type, 
                       nhead=args.nhead).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, len(train_loader), T_mult=2, eta_min=0, last_epoch=-1, verbose=False)
    # print(model)
    # exit()
    criterion = nn.MSELoss()
    best_model = None
    best = {}
    max_val_acc = -5e8
    train_loss = []
    train_accs = []
    epoch_pbar = trange(args.num_epoch, desc="{} Epoch".format(code))
    for epoch in epoch_pbar:
        model.train()
        train_loss = []
        # predict_all = np.array([],dtype=int)
        # label_all = np.array([],dtype=int)
        for step, (x,y) in enumerate(train_loader):
            # print('x',x.size(),'y',y.size())
            # print(np.all(np.isfinite(x.data.cpu().numpy())),np.all(np.isfinite(y.data.cpu().numpy())))
            
            if torch.isnan(x).any():
                print('error',torch.isnan(x).any())
                exit()
            x = x.to(device)
            y = y.to(device)
            # print(x.shape,y.shape,input_dim)
            prediction = model(x)
            loss = criterion(prediction, y)
            # predict_all = np.append(predict_all,torch.argmax(prediction, dim=-1).int().data.cpu().numpy())
            # label_all = np.append(label_all,y.data.cpu().numpy())
            optimizer.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
            optimizer.step()
            scheduler.step()
            train_loss.append(loss.item())
        train_loss = sum(train_loss) / len(train_loss)
        
        model.eval()
        valid_loss = []
        # print(label_all.shape,predict_all.shape)
        train_acc=-train_loss
        
            # predict_all = np.array([],dtype=int)
            # label_all = np.array([],dtype=int)
        for step, (x,y) in enumerate(valid_loader):
            x = x.to(device)
            y = y.to(device)
            with torch.no_grad():
                prediction = model(x)
            loss = criterion(prediction, y)
            valid_loss.append(loss.item())
                
        valid_loss = sum(valid_loss) / len(valid_loss)
        acc = -valid_loss
        # scheduler.step()
        if acc > max_val_acc:
            
            max_val_acc = acc
            best_model = copy.deepcopy(model)
            best['v-acc'] = acc
            best['t-acc'] = train_acc
            best['v-loss'] = valid_loss
            best['t-loss'] = train_loss
            best['epoch'] = epoch
        # ep={'code':code,'epoch':epoch,
        #     'v-acc':acc*100,
        #     't-acc':train_acc*100,
        #     'v-loss':valid_loss,
        #     't-loss':train_loss,
        #     'best-v-acc':best['v-acc']*100,
        #     'best-t-acc':best['t-acc']*100,
        #     'best-v-loss':best['v-loss'],
        #     'best-t-loss':best['t-loss'],
        #     'best epoch':best['epoch']}
        
        epoch_pbar.set_postfix({'v-acc': acc,'t-acc': train_acc,'best epoch':best['epoch']})
        # print('train: {:.4f}| val: {:.4f}| best-t: {:.2f}| best-v: {:.2f}| epoch: {}'.format(train_loss,valid_loss,best['t-acc'],best['v-acc'],best['epoch']))
    total_time = time.time()-st
    l={'code':code,
        'v-acc':best['v-acc'],
        't-acc':best['t-acc'],
        'v-loss':best['v-loss'],
        't-loss':best['t-loss'],
        'time':total_time,
        'best epoch':best['epoch'],
        'model':best_model}
    
    return l



def test(best,decode,args):
    code = best['code']
    model = best['model']
    with open(args.exp_name / f"log.txt","a") as f:
        f.write(f"=====Test and Plot=====\n")
        f.write(f"BEST {code}\n")
    train_loader,valid_loader,test_loader,input_dim = decode(code)
    # model = Classifier(input_dim=input_dim, output_dim=1, hidden_layers=args.hidden_layers, hidden_dim=args.hidden_dim, dropout=args.dropout, m_type = args.m_type).to(device)model = Classifier(input_dim=input_dim, output_dim=1, hidden_layers=args.hidden_layers, hidden_dim=args.hidden_dim, dropout=args.dropout, m_type = args.m_type).to(device)
    # model.load_state_dict(torch.load(f"{args.exp_name}_best.ckpt"))
    criterion = nn.MSELoss()
    print(best)
    """Make prediction."""

    pred = np.array([], dtype=np.int32)
    target = np.array([], dtype=np.int32)
    model.eval()
    test_loss = []
    with torch.no_grad():
        for i, (imgs, labels) in enumerate(tqdm(test_loader)):
            # features = batch
            features = imgs.to(device)

            outputs = model(features)
            
            # print(outputs)

            # _, test_pred = torch.max(outputs, 1) # get the index of the class with the highest probability
            pred = np.concatenate((pred, outputs.cpu().numpy()), axis=0)
            target = np.concatenate((target, labels.numpy()), axis=0)

            loss = criterion(outputs, labels.to(device))
            test_loss.append(loss.item())
        test_loss = sum(test_loss) / len(test_loss)
            

    # plt.clf()
    # plt.
    with open(args.exp_name / f"log.txt","a") as f:
        f.write(f"Test loss{test_loss:.5f}\n")
    print(pred.shape,target.shape)
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
    # if args.label is None:
    #     args.label = 36
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
    # predefine = TyphoonDataset(data_path = './完整數據集.csv',typhoon_ids=train_list.tolist(),concat_n=args.concat_n,split='pre')
    # predefine.load_db(predefine.data_path)
    # print(train_list,eval_list)
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
        best['model'] = Classifier(input_dim=input_dim, output_dim=1, hidden_layers=args.hidden_layers, hidden_dim=args.hidden_dim, dropout=args.dropout, m_type = args.m_type).to(device)
        
    if args.do_test:
        test(best,_decode,args)

