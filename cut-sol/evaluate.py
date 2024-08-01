# PYTHONUNBUFFERED=1 nohup python evaluate.py > ../data/adder/evaluate_out.log 2> ../data/adder/evaluate_err.log &

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
    

def main():
    device = torch.device('cuda:0')

    batch_size = 16
    hidden1_dim = 64
    hidden2_dim = 32

    folderPath = os.environ.get("HOME")+"/Priority-Cuts/Priority-Cuts-Filter/data/adder/"
    cutstats_dataset = CutClassTruths(folderPath=folderPath)

    test_dl = DataLoader(cutstats_dataset,shuffle=False,batch_size=batch_size,pin_memory=True,num_workers=4)

    delay_model = DelayPredictor(hidden1_dim=hidden1_dim, hidden2_dim=hidden2_dim).to(device)
        
    best_val_epoch = 1
    optim_valid_loss = 0.4634
    delay_model.load_state_dict(torch.load("model_weights/epoch-{}-val_loss-{:.4f}.pt".format(best_val_epoch, optim_valid_loss)))
    cut_delays = []   

    start_time = time.time()  
     
    print("Evaluating..")
    n_total_steps = len(test_dl)
    with torch.no_grad():
        for i, batch_truths in enumerate(test_dl):
            cur_batch_size = len(batch_truths[0])
            batch_truths = torch.tensor(np.array(batch_truths)).t().float().to(device)
            class_pred, reg_pred = delay_model(batch_truths)
            for j in range(0,cur_batch_size,2):
                tmp_class_pred1 = class_pred[j].item()
                tmp_class_pred2 = class_pred[j+1].item()
                tmp_reg_pred1 = reg_pred[j].item()
                tmp_reg_pred2 = reg_pred[j+1].item()
                if tmp_class_pred1 < 0.5 and tmp_class_pred2 < 0.5:
                    cut_delays.append(-1)
                elif tmp_class_pred1 >= 0.5 and tmp_class_pred2 < 0.5:
                    cut_delays.append(tmp_reg_pred1)
                elif tmp_class_pred1 < 0.5 and tmp_class_pred2 >= 0.5:
                    cut_delays.append(tmp_reg_pred2) 
                else:
                    cut_delays.append(min(tmp_reg_pred1, tmp_class_pred2))          
    
    end_time = time.time()
    epoch_time = (end_time - start_time) / 60
    print(f"Evaluation took {epoch_time:.2f} minutes")

    with open(folderPath+"cut_delays.csv",'w', newline='') as file:
        csv_writer = csv.writer(file)
        for i in range(len(cut_delays)):
            csv_writer.writerow([str(i+1),cut_delays[i]])
    
    print("Evaluation Completed!!")

if __name__ == "__main__":
    main()