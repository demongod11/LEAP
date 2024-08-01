# PYTHONUNBUFFERED=1 nohup python train.py > train_out.log 2> train_err.log &

from model import *
from cs_dataset import *
from loss_function import *
import torch
import numpy as np
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os
import csv
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pickle
import time

def plotChart(x,y,xlabel,ylabel,leg_label,title):
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(1, 1, 1)
    plt.plot(x,y, label=leg_label)
    leg = plt.legend(loc='best', ncol=2, shadow=True, fancybox=True)
    leg.get_frame().set_alpha(0.5)
    plt.xlabel(xlabel, weight='bold')
    plt.ylabel(ylabel, weight='bold')
    plt.title(title,weight='bold')
    plt.savefig(title+'.png', format='png', bbox_inches='tight')
    

def main():
    device = torch.device('cuda:0')

    batch_size = 16
    num_epochs = 200
    learning_rate = 0.001
    hidden1_dim = 64
    hidden2_dim = 32

    folderPath = os.environ.get("HOME")+"/Priority-Cuts/Priority-Cuts-Filter/data/square/"
    cutstats_dataset = CutStatsDataset(folderPath=folderPath)

    training_validation_size = [int(0.9*len(cutstats_dataset)),len(cutstats_dataset) - int(0.9*len(cutstats_dataset))]
    train_DS,valid_DS = random_split(cutstats_dataset,training_validation_size)

    train_dl = DataLoader(train_DS,shuffle=True,batch_size=batch_size,pin_memory=True,num_workers=4)
    valid_dl = DataLoader(valid_DS,shuffle=True,batch_size=batch_size,pin_memory=True,num_workers=4)

    delay_model = DelayPredictor(hidden1_dim=hidden1_dim, hidden2_dim=hidden2_dim).to(device)
    optimizer = torch.optim.Adam(delay_model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, 'min',verbose=True)
    
    num_positive = cutstats_dataset.num_positive
    num_negative = len(cutstats_dataset) - num_positive
    weight_positive = num_negative / len(cutstats_dataset)
    weight_negative = num_positive / len(cutstats_dataset)
    class_weights = torch.tensor([[weight_negative, weight_positive]] * batch_size).to(device)
    loss_function = LossFunction(class_weights=class_weights,alpha=0.7)

    best_val_epoch = 1 
    valid_curve = []
    train_curve = []
    optim_valid_loss = 0

    for ep in range(num_epochs):
        start_time = time.time()
        print("\nEpoch [{}/{}]".format(ep+1, num_epochs))   
        print("\nTraining..")
        n_total_steps = len(train_dl)
        train_loss = 0
        for i, batch in enumerate(train_dl):
            batch_truths, batch_class_labels, batch_reg_labels = batch
            batch_truths = torch.tensor(np.array(batch_truths)).t().float().to(device)
            batch_class_labels = torch.tensor(np.array(batch_class_labels)).float().to(device)
            batch_reg_labels = torch.tensor(np.array(batch_reg_labels)).float().to(device)
            class_pred, reg_pred = delay_model(batch_truths)
            class_pred = class_pred.squeeze(1)
            reg_pred = reg_pred.squeeze(1)
            loss = loss_function(class_pred,reg_pred,batch_class_labels,batch_reg_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss = (train_loss*(i)+loss.item())/(i+1)
            
            if (i+1) % 100 == 0:
                print (f'Epoch [{ep+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {train_loss:.4f}')
                
        print("\nValidation..")
        n_total_steps = len(valid_dl)
        valid_loss = 0
        with torch.no_grad():
            for i, batch in enumerate(valid_dl):
                batch_truths, batch_class_labels, batch_reg_labels = batch
                batch_truths = torch.tensor(np.array(batch_truths)).t().float().to(device)
                batch_class_labels = torch.tensor(np.array(batch_class_labels)).float().to(device)
                batch_reg_labels = torch.tensor(np.array(batch_reg_labels)).float().to(device)
                class_pred, reg_pred = delay_model(batch_truths)
                class_pred = class_pred.squeeze(1)
                reg_pred = reg_pred.squeeze(1)
                loss = loss_function(class_pred,reg_pred,batch_class_labels,batch_reg_labels)
                valid_loss = (valid_loss*(i)+loss.item())/(i+1)
                
                if (i+1) % 100 == 0:
                    print (f'Epoch [{ep+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {valid_loss:.4f}')
        
        if ep == 0:
            optim_valid_loss = valid_loss
            torch.save(delay_model.state_dict(), "model_weights/epoch-{}-val_loss-{:.4f}.pt".format(ep+1,optim_valid_loss))
        else:
            if valid_loss < optim_valid_loss:
                optim_valid_loss = valid_loss
                best_val_epoch = ep+1
                torch.save(delay_model.state_dict(), "model_weights/epoch-{}-val_loss-{:.4f}.pt".format(ep+1,optim_valid_loss))
            
        print("Training loss for epoch {} is {:.4f}".format(ep+1,train_loss))
        print("Validation loss for epoch {} is {:.4f}".format(ep+1,valid_loss))
        train_curve.append(train_loss)
        valid_curve.append(valid_loss)
        scheduler.step(valid_loss)
        end_time = time.time()
        epoch_time = (end_time - start_time) / 60
        print(f"Epoch {ep+1} took {epoch_time:.2f} minutes")


    # Save training data for future plots
    with open('valid_curve.pkl','wb') as f:
        pickle.dump(valid_curve,f)

    with open('train_curve.pkl','wb') as f:
        pickle.dump(train_curve,f)
        
    ##### Plotting ######    
    plotChart([i+1 for i in range(len(valid_curve))],valid_curve,"# Epochs","Loss","valid_loss","Validation loss")
    plotChart([i+1 for i in range(len(train_curve))],train_curve,"# Epochs","Loss","train_loss","Training loss")


if __name__ == "__main__":
    main()