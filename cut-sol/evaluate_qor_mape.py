# PYTHONUNBUFFERED=1 nohup python evaluate_qor_mape.py > eval_qor_mape_out.log 2> eval_qor_mape_err.log &

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
    
def mape(y_true, y_pred):
    """
    Computes the mean absolute percentage error between y_true and y_pred.
    
    Args:
        y_true (torch.Tensor): Ground truth values.
        y_pred (torch.Tensor): Predicted values.
        
    Returns:
        torch.Tensor: Mean absolute percentage error.
    """
    error = torch.abs((y_true - y_pred) / torch.clamp(torch.abs(y_true), min=1e-6))
    return 100.0 * torch.mean(error)


def main():
    print("Evaluating..")
    start_time = time.time()  
    device = torch.device('cuda:0')

    batch_size = 16
    hidden1_dim = 64
    hidden2_dim = 32

    folderPath = os.environ.get("HOME")+"/Priority-Cuts/Priority-Cuts-Filter/data/square/"
    cutstats_dataset = CutQoR(folderPath=folderPath)
    print(len(cutstats_dataset))
    test_dl = DataLoader(cutstats_dataset,shuffle=False,batch_size=batch_size,pin_memory=True,num_workers=4)

    delay_model = DelayQoRPredictor(hidden1_dim=hidden1_dim, hidden2_dim=hidden2_dim).to(device)
        
    best_val_epoch = 124
    optim_valid_loss = 17.2217
    delay_model.load_state_dict(torch.load("model_weights/qor_model/epoch-{}-val_loss-{:.4f}.pt".format(best_val_epoch, optim_valid_loss)))
    
    valid_loss = 0
    n_total_steps = len(test_dl)
    with torch.no_grad():
        for i, batch in enumerate(test_dl):
            batch_truths, batch_reg_labels = batch
            batch_truths = torch.tensor(np.array(batch_truths)).t().float().to(device)
            batch_reg_labels = torch.tensor(np.array(batch_reg_labels)).float().to(device)
            reg_pred = delay_model(batch_truths)
            reg_pred = reg_pred.squeeze(1)
            loss = mape(reg_pred,batch_reg_labels)
            valid_loss = (valid_loss*(i)+loss.item())/(i+1)

    print(f"Validation loss is {valid_loss:.4f}")

    end_time = time.time()
    epoch_time = (end_time - start_time) / 60
    print(f"QoR Evaluation took {epoch_time:.2f} minutes")
    print("QoR Evaluation Completed!!")

if __name__ == "__main__":
    main()