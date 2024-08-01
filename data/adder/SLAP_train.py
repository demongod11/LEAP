import pandas as pd
import numpy as np
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import sys

# Get the current time before running the code
start_time = datetime.datetime.now()

# Node Embedding

# Read the data from nodes.csv
node_df = pd.read_csv('/workspace/ckarfa/Priority-Cuts/Priority-Cuts-Filter/data/rc16b/nodes.csv')
labels = pd.read_csv('/workspace/ckarfa/Priority-Cuts/Priority-Cuts-Filter/data/rc16b/labels.csv')
# Merge with itself based on c1_idx and c2_idx
node_result = pd.merge(node_df, node_df, how='left', left_on='c1_idx', right_on='node_idx', suffixes=('', '_c1'))
node_result = pd.merge(node_result, node_df, how='left', left_on='c2_idx', right_on='node_idx', suffixes=('', '_c2'))


# Cut Embeddings

cut_df = pd.read_csv('/workspace/ckarfa/Priority-Cuts/Priority-Cuts-Filter/data/rc16b/cuts.csv')

num_cuts = cut_df.shape[0]


# Getting average delay for each cut amongst all mappings

sum_delays = np.empty(num_cuts, dtype=object)
# num_delays = np.empty((0,1))

# for cut in range(num_cuts):
#     # sum_delays[cut+1] = []
#     # num_delays[cut+1] = 0
#     num_delays = np.vstack((0,num_delays))

num_delays = np.zeros(num_cuts)

# for j in range(10000):
#     cur_map_dataframe = pd.read_csv("/workspace/ckarfa/Priority-Cuts/Priority-Cuts-Filter/data/adder/selective_maps/map_" + str(j+1) +".csv")
#     cut_1_list = cur_map_dataframe['cut1_idx'].tolist()
#     cut_2_list = cur_map_dataframe['cut2_idx'].tolist()
#     combined_values = list(set(cut_1_list+cut_2_list))
#     for cut in combined_values:
#         num_delays[cut-1] = num_delays[cut-1] +1
#         x = sum_delays[cut-1]
#         x = np.vstack((x,[labels.iloc[j,5]]))
#         sum_delays[cut-1] = x

for j in range(10000):
    cur_map_dataframe = pd.read_csv("/workspace/ckarfa/Priority-Cuts/Priority-Cuts-Filter/data/rc16b/selective_maps/map_" + str(j+1) +".csv")
    cut_1_list = cur_map_dataframe['cut1_idx'].to_numpy()
    cut_2_list = cur_map_dataframe['cut2_idx'].to_numpy()
    combined_values = np.unique(np.concatenate((cut_1_list, cut_2_list)))
    
    for cut in combined_values:
        cut_index = cut - 1
        num_delays[cut_index] += 1
        sum_delays[cut_index] = np.vstack((sum_delays[cut_index], [labels.iloc[j, 5]]))

# num_sum = 0

# for cut in range(num_cuts):
#     if sum_delays[cut+1] == 0:
#         num_sum = num_sum+1

# print(num_sum, "num sum")



# Select necessary columns and rename them
node_result = node_result[['node_idx', 'node_inv', 'num_fo', 'lvl', 'rev_lvl', 
                 'lvl_c1', 'num_fo_c1', 'node_inv_c1', 
                 'lvl_c2', 'num_fo_c2', 'node_inv_c2']]
node_result.columns = ['node_idx', 'node_inv', 'num_fo', 'lvl', 'rev_lvl', 
                  'c1_lvl', 'c1_fo', 'c1_inv', 
                  'c2_lvl', 'c2_fo', 'c2_inv']
node_result.fillna(0 ,inplace = True)

node_result.to_csv('/workspace/ckarfa/Priority-Cuts/node_embeddings.csv', index=False)


cut_result = {
    'inv_root':[],
    'num_lvs':[],
    'vol':[],
    'min_lvl':[],
    'max_lvl':[],
    'sum_lvl':[],
    'min_fo':[],
    'max_fo':[],
    'sum_fo':[]
}

train_result = {
    'root_emb':[],
    'ch1_emb':[],
    'ch2_emb':[],
    'ch3_emb':[],
    'ch4_emb':[],
    'ch5_emb':[],
    'inv_root':[],
    'num_lvs':[],
    'vol':[],
    'min_lvl':[],
    'max_lvl':[],
    'sum_lvl':[],
    'min_fo':[],
    'max_fo':[],
    'sum_fo':[]
}

# matrices = []
# matrices = np.array(matrices)
# labels_calc = []
# labels_calc = np.array(labels_calc)

matrices = np.empty((0, 150))
labels_calc = np.empty((0, 1))

cut_result = pd.DataFrame(cut_result)
train_result = pd.DataFrame(train_result)

num_of_nans = 0

for ijk in range(num_cuts):
    matrix_row = []
    cut_num = ijk
    curr_row = cut_df.loc[cut_num]
    inv_root = node_df[node_df['node_idx'] == curr_row['root_idx']]['node_inv']
    inv_root = inv_root.iloc[0]
    num_lvs = (0 if (curr_row['l1_idx'] == -1) else 1) + (0 if (curr_row['l2_idx'] == -1) else 1) + (0 if (curr_row['l3_idx'] == -1) else 1) + (0 if (curr_row['l4_idx'] == -1) else 1) + (0 if (curr_row['l5_idx'] == -1) else 1)
    vol = curr_row['vol_cut']
    lvl = [0,1,2,3,4,5]
    lvl[0] = -1 if (curr_row['l1_idx'] == -1) else node_df[node_df['node_idx'] == curr_row['l1_idx']]["lvl"].iloc[0]
    lvl[1] = -1 if (curr_row['l2_idx'] == -1) else node_df[node_df['node_idx'] == curr_row['l2_idx']]["lvl"].iloc[0]
    lvl[2] = -1 if (curr_row['l3_idx'] == -1) else node_df[node_df['node_idx'] == curr_row['l3_idx']]["lvl"].iloc[0]
    lvl[3] = -1 if (curr_row['l4_idx'] == -1) else node_df[node_df['node_idx'] == curr_row['l4_idx']]["lvl"].iloc[0]
    lvl[4] = -1 if (curr_row['l5_idx'] == -1) else node_df[node_df['node_idx'] == curr_row['l5_idx']]["lvl"].iloc[0]

    arr_added = []
    for k in node_result.iloc[curr_row['root_idx']]:
        arr_added.append(k)
    arr_added = arr_added[1:]
    for k in arr_added:
        matrix_row.append(k)
    # print (matrix_row)

    max_lvl = max(lvl)
    min_lvl = -1
    sum_lvl = 0
    for j in  range(5):
        if(min_lvl == -1):
            min_lvl = lvl[j]
        if(lvl[j] != -1):
            min_lvl = min(min_lvl,lvl[j])
            sum_lvl = sum_lvl + lvl[j]
    if(min_lvl == -1):
        min_lvl = 0
    

    fo = [0,1,2,3,4,5]
    blank_arr = [0,0,0,0,0,0,0,0,0,0]
    fo[0] = -1 if (curr_row['l1_idx'] == -1) else node_df[node_df['node_idx'] == curr_row['l1_idx']]["num_fo"].iloc[0]
    fo[1] = -1 if (curr_row['l2_idx'] == -1) else node_df[node_df['node_idx'] == curr_row['l2_idx']]["num_fo"].iloc[0]
    fo[2] = -1 if (curr_row['l3_idx'] == -1) else node_df[node_df['node_idx'] == curr_row['l3_idx']]["num_fo"].iloc[0]
    fo[3] = -1 if (curr_row['l4_idx'] == -1) else node_df[node_df['node_idx'] == curr_row['l4_idx']]["num_fo"].iloc[0]
    fo[4] = -1 if (curr_row['l5_idx'] == -1) else node_df[node_df['node_idx'] == curr_row['l5_idx']]["num_fo"].iloc[0]

    max_fo = max(fo)
    min_fo = -1
    sum_fo = 0
    for j in  range(5):
        if(min_fo == -1):
            min_fo = fo[j]
        if(fo[j] != -1):
            min_fo = min(min_fo,fo[j])
            sum_fo = sum_fo + fo[j]
    if(min_fo == -1):
        min_fo = 0

    row = {
        'inv_root': inv_root,
        'num_lvs': num_lvs,
        'vol': vol,
        'min_lvl':min_lvl,
        'max_lvl':max_lvl,
        'sum_lvl':sum_lvl,
        'min_fo':min_fo,
        'max_fo':max_fo,
        'sum_fo':sum_fo
    }
    for j in range(5):
        arr_added = []
        s = 'l' + str(j+1) + '_idx'
        if (curr_row[s] == -1):
            arr_added = blank_arr
        else:
            for k in node_result.loc[curr_row[s]]:
                arr_added.append(k)
            arr_added = arr_added[1:]
        for k in arr_added:
            matrix_row.append(k)

    for j in range(10):
        matrix_row.append(inv_root)
    for j in range(10):
        matrix_row.append(num_lvs)
    for j in range(10):
        matrix_row.append(vol)
    for j in range(10):
        matrix_row.append(min_lvl)
    for j in range(10):
        matrix_row.append(max_lvl)
    for j in range(10):
        matrix_row.append(sum_lvl)
    for j in range(10):
        matrix_row.append(min_fo)
    for j in range(10):
        matrix_row.append(max_fo)
    for j in range(10):
        matrix_row.append(sum_fo)
    
    # print(matrix_row)

    row_result = {
        'root_emb': node_result.loc[curr_row['root_idx']],
        'ch1_emb': blank_arr if (curr_row['l1_idx'] == -1) else node_result.loc[curr_row['l1_idx']],
        'ch2_emb': blank_arr if (curr_row['l2_idx'] == -1) else node_result.loc[curr_row['l2_idx']],
        'ch3_emb': blank_arr if (curr_row['l3_idx'] == -1) else node_result.loc[curr_row['l3_idx']],
        'ch4_emb': blank_arr if (curr_row['l4_idx'] == -1) else node_result.loc[curr_row['l4_idx']],
        'ch5_emb': blank_arr if (curr_row['l5_idx'] == -1) else node_result.loc[curr_row['l5_idx']],
        'inv_root': inv_root,
        'num_lvs': num_lvs,
        'vol': vol,
        'min_lvl':min_lvl,
        'max_lvl':max_lvl,
        'sum_lvl':sum_lvl,
        'min_fo':min_fo,
        'max_fo':max_fo,
        'sum_fo':sum_fo
    }
    # train_result = train_result._append(row_result,ignore_index=True)
    cut_result = cut_result._append(row,ignore_index=True)
    if num_delays[cut_num] == 0:
        # print("NAN added")
        num_of_nans = num_of_nans+1
        # print(num_of_nans)
        # labels_calc.append(float('nan'))
        # labels_calc = np.vstack([labels_calc,float('nan')])
        # matrices.append(matrix_row)
        # matrices = np.vstack([matrices, matrix_row])
    else:
        # for nums in range(num_delays[cut_num+1]):
        #     print(nums, ", " , cut_num)
        #     labels_calc = np.vstack([labels_calc,sum_delays[cut_num+1][nums]])
        #     matrices = np.vstack([matrices, matrix_row])
        print(cut_num, " this cut num started")
        indices_array = np.arange(num_delays[cut_num])[:, np.newaxis]  # Shape: (num_delays[cut_num+1], 1)

        # 2. Repeat sum_delays and matrix_row
        repeated_sum_delays = np.array(sum_delays[cut_num][1:])
        repeated_matrix_row = np.array([matrix_row] * indices_array.shape[0])
        labels_calc = np.concatenate((labels_calc, repeated_sum_delays), axis=0)
        matrices = np.concatenate((matrices, repeated_matrix_row), axis=0)

        # print(indices_array, cut_num)
        # break
        print(cut_num, " this cut num ended")
        # if matrices.shape[0] > 100000:
        #     break
# horizontally stack labels to matrices

matrices = np.hstack((matrices, labels_calc))

# remove all the rows with nan

# matrices = matrices[~np.isnan(matrices[:,-1])]

# Calculate the minimum and maximum values of the 151st column
min_val = matrices[:, -1].min()
max_val = matrices[:, -1].max()

# Normalize the values in the 151st column to the range [0, 9]
normalized_col_151 = (((matrices[:, -1] - min_val)*9) // (max_val - min_val)) 

# Round the normalized values to integers
# normalized_col_151_int = np.round(normalized_col_151).astype(int)

# Update the 151st column in the original array with the rounded integer values
matrices[:, -1] = normalized_col_151

matrices = matrices[np.logical_and(matrices[:,-1] >=0, matrices[:,-1] <= 9)]

# random_indices = np.random.choice(matrices.shape[0], 100000, replace=False)

# # matrices = pd.DataFrame(matrices)
# matrices = matrices[random_indices]
# Creating the model for training

print("training getting started")



# Get the current time after running the code
end_time = datetime.datetime.now()

# Calculate the execution time
execution_time = end_time - start_time

print("Execution time:", execution_time)


import pandas as pd
import datetime
import torch
from sklearn.model_selection import train_test_split
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
import random

start_time = datetime.datetime.now()


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(15, 1), stride=1)
        self.bn1 = nn.BatchNorm2d(128)  # Added Batch Normalization
        self.fc1 = nn.Linear(128*10, 10)

        # Xavier/He initialization for weights
        nn.init.xavier_uniform_(self.conv.weight)
        nn.init.kaiming_uniform_(self.fc1.weight)

    def forward(self, x):
        x = self.conv(x)
        # x = self.bn1(x)  # Apply batch normalization
        x = F.relu(x)    # Using ReLU activation
        x = x.view(-1, 1280)
        x = self.fc1(x)
        x = torch.relu(x)
        return x

class CustomDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_item = self.data[index]
        target_item = self.targets[index]
        
        # If data_item is a tuple or list with more than one element, return it as is
        if isinstance(data_item, (tuple, list)) and len(data_item) > 1:
            return data_item, target_item
        else:
            return data_item, target_item

# Set device
device = torch.device("cuda:0")

# Hyperparameters
num_epochs = 50
batch_size = 16


x_train = matrices[:, :-1]
y_train = matrices[:, -1]

# print(x_train.shape)
#  print the number of 0s, 1s, 2s, 3s, 4s, 5s in the labels

for i in range(10):
    print("Number of ", i, "s in the labels: ", len(y_train[y_train == i]))



x_train = x_train.reshape(-1, 1, 15, 10)
y_train = y_train.reshape(-1)

# Convert to PyTorch tensors
print(type(x_train))
print(type(y_train))

x_train = x_train.astype(str).astype(np.float32)
y_train = y_train.astype(str).astype(np.float32)
x_train = torch.tensor(x_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.long).to(device)

# Perform train-validation split
dataset = list(zip(x_train, y_train))
train_size = int(0.7 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# train_data, train_targets = train_dataset
# val_data, val_targets = val_dataset

# train_dataset = CustomDataset(train_data, train_targets)
# val_dataset = CustomDataset(val_dataset,val_targets)

# Create data loaders for training and validation
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Initialize network
model = CNN().to(device)

# Loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

optimal_accuracy = 0
best_epoch = -1


# Train Network

# with open("/workspace/ckarfa/Priority-Cuts/epoch_data.txt", "w") as file:
#     for epoch in range(num_epochs):
#         print('Training')
#         train_loss = 0
#         for x_batch, y_batch in tqdm(train_loader):
#             model.train()
#             x_batch = x_batch.to(device)
#             y_batch = y_batch.to(device)
#             y_pred = model(x_batch)
#             loss = criterion(y_pred, y_batch)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             train_loss += loss.item()
#         train_loss /= len(train_loader)
#         print(f"Epoch {epoch+1}, Training Loss: {train_loss:.4f}")
#         file.write(f"Train loss: {train_loss:.4f}\n")
#         print('Validating')
#         val_loss = 0
#         correct = 0
#         total = 0
#         with torch.no_grad():
#             model.eval()
#             for x_batch, y_batch in tqdm(val_loader):
#                 y_pred = model(x_batch)
#                 loss = criterion(y_pred, y_batch)
#                 val_loss += loss.item()
#                 _, predicted = y_pred.max(1)
#                 total += y_batch.size(0)
#                 correct += predicted.eq(y_batch).sum().item()
#         val_loss /= len(val_loader)
#         accuracy = correct / total
#         print(f"Epoch {epoch+1}, Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}")
#         file.write(f"validation loss: {val_loss:.4f}\n")
#         file.write(f"accuracy: {accuracy:.4f}\n")
#         if accuracy > optimal_accuracy:
#             optimal_accuracy = accuracy
#             best_epoch = epoch
#             torch.save(model.state_dict(), "/workspace/ckarfa/Priority-Cuts/Validation_accuracy_for_slap_dummy/" + str(epoch) + ".pt")
#         print("Optimal accuracy so far: ", optimal_accuracy)
#         print("Best epoch so far: ", best_epoch)
#         print()
        

# file.close()

def train_model(model, train_loader, val_loader, num_epochs):
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    best_epoch = 0
    best_accuracy = -1

    for epoch in range(num_epochs):
        # Training
        # model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # for name, param in model.named_parameters():
        #     if param.grad is not None:
        #         print(f"Parameter: {name}, Gradient: {param.grad}")
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Training'):
            # print(inputs)
            # print(labels)
            # optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping

            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_accuracy = 100. * correct / total

        # for name, param in model.named_parameters():
        #     if param.grad is not None:
        #         print(f"Parameter: {name}, Gradient: {param.grad}")

        # Validation
        # model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Validation'):
                # print(inputs)
                # print(labels)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_loss /= len(val_loader)
        val_accuracy = 100. * correct / total
        # for name, param in model.named_parameters():
        #     if param.grad is not None:
        #         print(f"Parameter: {name}, Gradient: {param.grad}")
        torch.save(model.state_dict(), "/workspace/ckarfa/Priority-Cuts/validation_accuracy_for_slap/" + str(epoch) + ".pt")
        # /workspace/ckarfa/Priority-Cuts/validation_accuracy_for_slap
        print(f'Epoch {epoch+1}/{num_epochs}, '
              f'Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%, '
              f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_epoch = epoch+1

    print('Finished Training')
    print(best_epoch," is the best epoch")


train_model(model=model, train_loader=train_loader, val_loader=val_loader, num_epochs=num_epochs)

# model.load_state_dict(torch.load("/workspace/ckarfa/Priority-Cuts/validation_accuracy_for_slap/" + str(best_epoch) + ".pt"))
end_time = datetime.datetime.now()
# print(f"Accuracy on training set: {check_accuracy(train_loader, model)*100:.2f}")
# print(f"Accuracy on validation set: {check_accuracy(val_loader, model)*100:.2f}")
print("best epoch is ", best_epoch)
execution_time = end_time - start_time
print("Execution time:", execution_time)

