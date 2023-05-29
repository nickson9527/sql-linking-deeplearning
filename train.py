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
model = Classifier(input_dim=11, output_dim=1, hidden_layers=8, hidden_dim=256, dropout=0.1, m_type = 'lstm').to(device)
# model = torchvision.models.resnet152(pretrained=False).to(device)

# For the classification task, we use cross-entropy as the measurement of performance.
# criterion = nn.CrossEntropyLoss()
criterion = nn.MSELoss()

# Initialize optimizer, you may fine-tune some hyperparameters such as learning rate on your own.
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)


"""# Dataloader"""

# Construct train and valid datasets.
# The argument "loader" tells how torchvision reads the data.
train_set = TyphoonDataset(data_path = './完整數據集.csv',typhoon_ids=[1,4,7,10,13,14,16,18,23,37],concat_n=args.concat_n,split='train')
train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
print('train_loader',len(train_loader))
valid_set = TyphoonDataset(data_path = './完整數據集.csv',typhoon_ids=[2,5,8,28,31,34],concat_n=args.concat_n,split='valid')
valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
print('valid_loader',len(valid_loader))
test_set = TyphoonDataset(data_path = './完整數據集.csv',typhoon_ids=[36],concat_n=args.concat_n,split='test')
test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
print('test_loader',len(test_loader))
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, len(train_loader), T_mult=2, eta_min=0, last_epoch=- 1, verbose=False)

"""# Start Training"""

# # Initialize trackers, these are not parameters and should not be changed
stale = 0
best_loss = 5e10

for epoch in range(args.n_epochs):

    # ---------- Training ----------
    # Make sure the model is in train mode before training.
    model.train()

    # These are used to record information in training.
    train_loss = []
    train_accs = []

    for batch in tqdm(train_loader, desc="Train"):
        
        # A batch consists of image data and corresponding labels.
        imgs, labels = batch
        if torch.isnan(imgs).any():
            print('error',torch.isnan(imgs).any())
            exit()
        #imgs = imgs.half()
        #print(imgs.shape,labels.shape)

        # Forward the data. (Make sure data and model are on the same device.)
        logits = model(imgs.to(device))

        # Calculate the cross-entropy loss.
        # We don't need to apply softmax before computing cross-entropy as it is done automatically.
        loss = criterion(logits, labels.to(device))

        # Gradients stored in the parameters in the previous step should be cleared out first.
        optimizer.zero_grad()

        # Compute the gradients for parameters.
        loss.backward()

        # Clip the gradient norms for stable training.
        # grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

        # Update the parameters with computed gradients.
        optimizer.step()
        scheduler.step()
        # Compute the accuracy for current batch.
        # acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

        # Record the loss and accuracy.
        train_loss.append(loss.item())
        # train_accs.append(acc)
    # exit()
    train_loss = sum(train_loss) / len(train_loss)
    # train_acc = sum(train_accs) / len(train_accs)

    # Print the information.
    # print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

    # ---------- Validation ----------
    # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
    model.eval()

    # These are used to record information in validation.
    valid_loss = []
    # valid_accs = []

    # Iterate the validation set by batches.
    for batch in tqdm(valid_loader, desc="Valid"):

        # A batch consists of image data and corresponding labels.
        imgs, labels = batch
        #imgs = imgs.half()

        # We don't need gradient in validation.
        # Using torch.no_grad() accelerates the forward process.
        with torch.no_grad():
            logits = model(imgs.to(device))

        # We can still compute the loss (but not the gradient).
        loss = criterion(logits, labels.to(device))

        # Compute the accuracy for current batch.
        # acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

        # Record the loss and accuracy.
        valid_loss.append(loss.item())
        # valid_accs.append(acc)
        #break

    # The average loss and accuracy for entire validation set is the average of the recorded values.
    valid_loss = sum(valid_loss) / len(valid_loss)
    # valid_acc = sum(valid_accs) / len(valid_accs)

    # Print the information.
    # print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")
    print(f"[ {epoch + 1:03d}/{args.n_epochs:03d} ] Train loss = {train_loss:.5f}| Valid loss = {valid_loss:.5f}")

    # update logs
    if valid_loss < best_loss:
        with open(f"./{args.exp_name}_log.txt","a") as f:
            f.write(f"[ {epoch + 1:03d}/{args.n_epochs:03d} ] Train loss = {train_loss:.5f}| Valid loss = {valid_loss:.5f}-> best\n")
    else:
        with open(f"./{args.exp_name}_log.txt","a") as f:
            f.write(f"[ {epoch + 1:03d}/{args.n_epochs:03d} ] Train loss = {train_loss:.5f}| Valid loss = {valid_loss:.5f}\n")


    # save models
    if valid_loss < best_loss:
        print(f"Best model found at epoch {epoch + 1} with {valid_loss:.5f}, saving model")
        torch.save(model.state_dict(), f"{args.exp_name}_best.ckpt") # only save best to prevent output memory exceed error
        best_loss = valid_loss
        stale = 0
    else:
        stale += 1
        if stale > args.patience:
            print(f"No improvment {args.patience} consecutive epochs, early stopping")
            print(f"acc = {best_loss:.5f} -> best")
            break

# model = Classifier(input_dim=input_dim, hidden_layers=hidden_layers, hidden_dim=hidden_dim).to(device)
model = Classifier(input_dim=11, output_dim=1, hidden_layers=8, hidden_dim=256, dropout=0.1, m_type = 'lstm').to(device)
model.load_state_dict(torch.load(f"{args.exp_name}_best.ckpt"))

"""Make prediction."""

pred = np.array([], dtype=np.int32)
target = np.array([], dtype=np.int32)
model.eval()
with torch.no_grad():
    for i, (imgs, labels) in enumerate(tqdm(test_loader)):
        # features = batch
        features = imgs.to(device)

        outputs = model(features)
        # print(outputs)

        # _, test_pred = torch.max(outputs, 1) # get the index of the class with the highest probability
        pred = np.concatenate((pred, outputs.cpu().numpy()), axis=0)
        target = np.concatenate((target, labels.numpy()), axis=0)
        

# plt.clf()
# plt.
print(pred.shape,target.shape)
fig, ax = plt.subplots(1,figsize=(32,12))
ax.plot(np.arange(pred.shape[0]),pred,color='r',label='pred')
ax.plot(np.arange(target.shape[0]),target,color='b',label='true')
ax.legend()
plt.savefig('./pred.png')