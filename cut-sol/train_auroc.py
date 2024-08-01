# PYTHONUNBUFFERED=1 nohup python train_auroc.py > ../data/square/train_auroc_out.log 2> ../data/square/train_auroc_err.log &

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
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Subset

def main():
    # start_time = time.time()  
    device = torch.device('cuda:0')

    batch_size = 16
    hidden1_dim = 64
    hidden2_dim = 32

    folderPath = os.environ.get("HOME")+"/Priority-Cuts/Priority-Cuts-Filter/data/square/"
    cutstats_dataset = CutClass(folderPath=folderPath)

    X = cutstats_dataset.truths
    y = cutstats_dataset.class_labels
    
    validation_ratio = 0.1
    
    # Initialize StratifiedShuffleSplit
    stratified_splitter = StratifiedShuffleSplit(n_splits=1, test_size=validation_ratio, random_state=42)

    # Split the data
    for train_index, valid_index in stratified_splitter.split(X, y):
        train_DS = Subset(cutstats_dataset, train_index)
        valid_DS = Subset(cutstats_dataset, valid_index)
    
    test_dl = DataLoader(valid_DS,shuffle=False,batch_size=batch_size,pin_memory=True,num_workers=4)
    print(len(valid_DS))
    delay_model = DelayClassPredictor(hidden1_dim=hidden1_dim, hidden2_dim=hidden2_dim).to(device)
        
    best_val_epoch = 168
    optim_valid_loss = 0.0143
    delay_model.load_state_dict(torch.load("model_weights/class_model/epoch-{}-val_loss-{:.4f}.pt".format(best_val_epoch, optim_valid_loss)))
    cut_delays = []   
     
    print("Evaluating..")
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for i, batch in enumerate(test_dl):
            batch_truths, batch_class_labels = batch
            batch_truths = torch.tensor(np.array(batch_truths)).t().float().to(device)
            batch_class_labels = torch.tensor(np.array(batch_class_labels)).float().to(device)
            class_pred = delay_model(batch_truths)
            class_pred = class_pred.squeeze(1)
            all_preds.extend(class_pred.cpu().numpy())
            all_labels.extend(batch_class_labels.cpu().numpy())
    
    threshold_arr = [0,0.25,0.5,0.75,1]
    tpr_arr = []
    fpr_arr = []
    tp_arr = []
    fp_arr = []
    tn_arr = []
    fn_arr = []
    for thr in threshold_arr:
        tp=0
        fp=0
        tn=0
        fn=0
        for i in range(len(all_labels)):
            if (all_preds[i] > thr): 
                pred = 1
            else:
                pred = 0
            
            if pred == 1 and all_labels[i] == 1:
                tp+=1
            elif pred == 1 and all_labels[i] == 0:
                fp+=1
            elif pred == 0 and all_labels[i] == 1:
                fn+=1
            else:
                tn+=1
        tp_arr.append(tp)
        fp_arr.append(fp)
        tn_arr.append(tn)
        fn_arr.append(fn)
        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)
        acc = (tp+tn)/(tp+tn+fp+fn)
        tpr_arr.append(tpr)
        fpr_arr.append(fpr)
        print("Threshold - "+ str(thr) + " tp = " + str(tp) + ", fp = " + str(fp) + ", tn = " + str(tn) + ", fn = " + str(fn) + ", Accuracy = " + str(acc))
    
    
    # Plot ROC Curve
    for i, thr in enumerate(threshold_arr):
        plt.scatter(fpr_arr[i], tpr_arr[i], label=f'Threshold = {thr}', color=plt.cm.jet(i / len(threshold_arr)))

    # Connect the points with a line
    plt.plot(fpr_arr, tpr_arr, linestyle='-')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')

    # Add legend at bottom left
    plt.legend(loc='upper left', bbox_to_anchor=(0.7, 0.3), title='Threshold Values', fontsize=8)

    plt.grid(True)

    # Save the plot as an image file
    plt.savefig('roc_curve.png', bbox_inches='tight')

    
if __name__ == "__main__":
    main()