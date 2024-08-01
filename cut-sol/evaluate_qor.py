# PYTHONUNBUFFERED=1 nohup python evaluate_qor.py > ../data/c6288/evaluate_qor_out.log 2> ../data/c6288/evaluate_qor_err.log &

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
    print("Evaluating..")
    start_time = time.time()  
    device = torch.device('cuda:0')

    batch_size = 16
    hidden1_dim = 64
    hidden2_dim = 32

    folderPath = os.environ.get("HOME")+"/Priority-Cuts/Priority-Cuts-Filter/data/c6288/"
    cutstats_dataset = CutQoRTruths(folderPath=folderPath)
    print(len(cutstats_dataset))
    test_dl = DataLoader(cutstats_dataset,shuffle=False,batch_size=batch_size,pin_memory=True,num_workers=4)

    delay_model = DelayQoRPredictor(hidden1_dim=hidden1_dim, hidden2_dim=hidden2_dim).to(device)
        
    best_val_epoch = 124
    optim_valid_loss = 17.2217
    delay_model.load_state_dict(torch.load("model_weights/qor_model/epoch-{}-val_loss-{:.4f}.pt".format(best_val_epoch, optim_valid_loss)))
    cut_delays = []   

    n_total_steps = len(test_dl)
    with torch.no_grad():
        for i, batch_truths in enumerate(test_dl):
            batch_truths = torch.tensor(np.array(batch_truths)).t().float().to(device)
            reg_pred = delay_model(batch_truths).squeeze(1)
            cut_delays.extend(reg_pred.tolist())
    
    tt_delay_map = {}
    for i in range(len(cutstats_dataset)):
        tt_delay_map[cutstats_dataset.decimal_cut_tt[i]] = cut_delays[i]

    with open(folderPath+"interim_cut_delays.csv", 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            tmp_cut_num = int(row[0])
            tmp_phase = int(row[1])
            if int(row[2]) == 1:
                tmp_cut_tt = cutstats_dataset.cut_truth_map[tmp_cut_num][tmp_phase]
                tmp_delay = tt_delay_map[tmp_cut_tt]
            else:
                tmp_delay = -1
            with open(folderPath+"cut_delays.csv", 'a+', newline='') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow([tmp_cut_num,tmp_phase,tmp_delay])
    
    end_time = time.time()
    epoch_time = (end_time - start_time) / 60
    print(f"QoR Evaluation took {epoch_time:.2f} minutes")
    print("QoR Evaluation Completed!!")

if __name__ == "__main__":
    main()