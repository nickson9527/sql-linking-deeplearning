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

parser = argparse.ArgumentParser()
parser.add_argument(
        "--exp_name",
        type=Path,
        default="exp",
    )
parser.add_argument("--batch_size", type=int, default=128, help="The number of batch size")
parser.add_argument("--concat_n", type=int, default=1, help="The number of concat (>1)")
parser.add_argument("--n_epochs", type=int, default=500, help="The number of training epochs")
parser.add_argument("--patience", type=int, default=20, help="If no improvement in 'patience' epochs, early stop")
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

set_random_seed(args.seed)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(torch.cuda.get_device_name(device))
# Initialize a model, and put it on the device specified.
# model = Classifier(input_dim=11, output_dim=1, hidden_layers=8, hidden_dim=256, dropout=0.1, m_type = 'lstm').to(device)
# model = torchvision.models.resnet152(pretrained=False).to(device)
model = LinearRegression(fit_intercept=True)
# model = LogisticRegression()
# For the classification task, we use cross-entropy as the measurement of performance.
# criterion = nn.CrossEntropyLoss()
# criterion = nn.MSELoss()

# Initialize optimizer, you may fine-tune some hyperparameters such as learning rate on your own.
# optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)


"""# Dataloader"""
label = 36
train_list = np.random.choice(list(set(np.arange(38))-set([label])), size=int(38*0.8),replace=False)
eval_list = np.array(list(set(np.arange(38)) - set([label]) - set(train_list)))
print(train_list,eval_list)
# Construct train and valid datasets.
# The argument "loader" tells how torchvision reads the data.
train_set = TyphoonDataset(data_path = './完整數據集.csv',typhoon_ids=[1,4,7,10,13,14,16,18,23,37],concat_n=args.concat_n,split='train')

# train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
# print('train_loader',len(train_loader))
# valid_set = TyphoonDataset(data_path = './完整數據集.csv',typhoon_ids=[2,5,8,28,31,34],concat_n=args.concat_n,split='valid')
# valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
# print('valid_loader',len(valid_loader))
test_set = TyphoonDataset(data_path = './完整數據集.csv',typhoon_ids=[36],concat_n=args.concat_n,split='test')
# test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
# print('test_loader',len(test_loader))
# scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, len(train_loader), T_mult=2, eta_min=0, last_epoch=- 1, verbose=False)

"""# Start Training"""
x,y = train_set.get_all_data()
nsamples, nx, ny = x.shape
x1 = x.reshape((nsamples,nx*ny))
# print(x.shape,y.shape)
model.fit(x1, y)
pred = model.predict(x1)
target = y
print(pred.shape,target.shape)
fig, ax = plt.subplots(1,figsize=(32,12))
ax.plot(np.arange(pred.shape[0]),pred,color='r',label='pred')
ax.plot(np.arange(target.shape[0]),target,color='b',label='true')
ax.legend()
plt.savefig('./pred.png')