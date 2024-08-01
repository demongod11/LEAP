# PYTHONUNBUFFERED=1 nohup python evaluate_class.py > ../data/c6288/evaluate_class_out.log 2> ../data/c6288/evaluate_class_err.log &

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
import time
    

def main():
    start_time = time.time()  
    device = torch.device('cuda:0')

    batch_size = 16
    hidden1_dim = 64
    hidden2_dim = 32

    folderPath = os.environ.get("HOME")+"/Priority-Cuts/Priority-Cuts-Filter/data/c6288/"
    cutstats_dataset = CutClassTruths(folderPath=folderPath)

    test_dl = DataLoader(cutstats_dataset,shuffle=False,batch_size=batch_size,pin_memory=True,num_workers=4)
    print(len(cutstats_dataset))
    delay_model = DelayClassPredictor(hidden1_dim=hidden1_dim, hidden2_dim=hidden2_dim).to(device)
        
    best_val_epoch = 168
    optim_valid_loss = 0.0143
    delay_model.load_state_dict(torch.load("model_weights/class_model/epoch-{}-val_loss-{:.4f}.pt".format(best_val_epoch, optim_valid_loss)))
    cut_delays = []   
     
    print("Evaluating..")
    n_total_steps = len(cutstats_dataset)
    with torch.no_grad():
        for i, batch_truths in enumerate(test_dl):
            cur_batch_size = len(batch_truths[0])
            batch_truths = torch.tensor(np.array(batch_truths)).t().float().to(device)
            class_pred = delay_model(batch_truths)
            for j in range(cur_batch_size):
                if class_pred[j].item() >= 0.5:
                    cut_delays.append(1)
                else:
                    cut_delays.append(0)
    
    tt_delay_map = {}
    for i in range(len(cutstats_dataset)):
        tt_delay_map[cutstats_dataset.decimal_cut_tt[i]] = cut_delays[i]

    with open(folderPath+"cuts.csv", 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)
        for row in csv_reader:
            tmp_cut_num = int(row[1])
            tmp_tt_0 = int(row[9])
            tmp_tt_1 = int(row[10])
            with open(folderPath+"interim_cut_delays.csv",'a+', newline='') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow([tmp_cut_num,0,tt_delay_map[tmp_tt_0]])
                csv_writer.writerow([tmp_cut_num,1,tt_delay_map[tmp_tt_1]])
    
    end_time = time.time()
    epoch_time = (end_time - start_time) / 60
    print(f"Class Evaluation took {epoch_time:.2f} minutes")
    print("Class Evaluation Completed!!")

if __name__ == "__main__":
    main()