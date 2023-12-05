import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Data Paths
data_path = "train_embeddings.csv"
label_path = "train_labels.csv"
test_data_path = "test_embeddings.csv"
test_label_path = "test_labels.csv"
valid_data_path = "valid_embeddings.csv"
valid_label_path = "valid_labels.csv"

# ------------ Aquire train data -------------

# Load filer_name-to-index dictionary
with open('train_dic.pkl', 'rb') as f:
    name_to_idx = pickle.load(f)

# Load and Training Data
X_train = np.loadtxt(data_path, delimiter=',')
y_train_mixed = pd.read_csv(label_path)

# Fix training labels
y_train = np.zeros(np.shape(y_train_mixed)[0])
for index, row in y_train_mixed.iterrows():
    y_train[name_to_idx[row["id"]]] = row["label"]

# Make data into tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)


# ----------- Aquire validation data ------------

# Load validation file names to index dictionary
with open('valid_dic.pkl', 'rb') as f:
    name_to_idx = pickle.load(f)

# Load Validation Data
X_valid = np.loadtxt(valid_data_path, delimiter=',')
y_valid_mixed = pd.read_csv(valid_label_path)

# Fix indexes of valid data
y_valid = np.zeros(np.shape(y_valid_mixed)[0])
for index, row in y_valid_mixed.iterrows():
    y_valid[name_to_idx[row["id"]]] = row["label"]


X_valid_t = torch.tensor(X_valid, dtype=torch.float32)
y_valid_t = torch.tensor(y_valid, dtype=torch.float32).reshape(-1, 1)
    


# ------------ Define 2 Layer NN ---------------

INPUT_SIZE = X_train.shape[1]

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # Input is output of resnet18 image classification model
        self.fc1 = nn.Linear(INPUT_SIZE, 100) 
        self.fc2 = nn.Linear(100, 1)
        self.act_out = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.act_out(self.fc2(x))
        return x



# ----------- Train model ---------------

model = Net()
loss_fn = nn.MSELoss()
optim = optim.Adam(model.parameters(), lr=0.001)

def train_loop(x_train, y_train, x_valid, y_valid, model, loss_fn, optimizer, batch_size, epoch_max):
    n_examples = x_train.shape[0]
    train_loss = np.zeros(epoch_max)
    valid_loss = np.zeros(epoch_max)
    for epoch in range(epoch_max):
        print(f"\033[1m-----------Epoch {epoch}------------\033[0m")
        for i in range(0, (n_examples // batch_size) + 1):
            x_slice = x_train[i:min(i+batch_size, n_examples), :]
            y_slice = y_train[i:min(i+batch_size, n_examples)]
            y_pred = model(x_slice)
            loss = loss_fn(y_pred, y_slice)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # ------- Graphing --------
        y_pred_train = model(X_train_t)
        y_pred_valid = model(X_valid_t)

        train_acc = (y_pred_train.round() == y_train_t).float().mean()
        valid_acc = (y_pred_valid.round() == y_valid_t).float().mean()

        train_loss[epoch] = train_acc
        valid_loss[epoch] = valid_acc
        # train_loss[epoch] = loss_fn(y_pred_train, y_train)
        # valid_loss[epoch] = loss_fn(y_pred_valid, y_valid)
        print(f"loss = {loss}")
    return train_loss, valid_loss

n_epochs = 40
batch_size = 4
train_loss, valid_loss = train_loop(X_train_t, y_train_t, X_valid_t, y_valid_t, model, loss_fn, optim, batch_size, n_epochs)

# ----------- Plotting data ---------------
epoch_nums = np.arange(n_epochs)
plt.plot(epoch_nums, train_loss, label="train_accuracy")
plt.plot(epoch_nums, valid_loss, label="valid_accuracy")
plt.legend()
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.show()


# -------- TEST -----------

# Load test file name to index dictionary
with open('test_dic.pkl', 'rb') as f:
    name_to_idx = pickle.load(f)

# Load Test Data
X_test = np.loadtxt(test_data_path, delimiter=',')
y_test_mixed = pd.read_csv(test_label_path)

# Fix indexes of test data
y_test = np.zeros(np.shape(y_test_mixed)[0])
for index, row in y_test_mixed.iterrows():
    y_test[name_to_idx[row["id"]]] = row["label"]

# Make data into tensors
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)



# Predict
y_pred_train = model(X_train_t)
y_pred_test = model(X_test_t)
y_pred_valid = model(X_valid_t)

train_acc = (y_pred_train.round() == y_train_t).float().mean()
valid_acc = (y_pred_valid.round() == y_valid_t).float().mean()
test_acc = (y_pred_test.round() == y_test_t).float().mean()

print("-------Accuracy------------")
print(f"Train Accuracy = {train_acc}")
print(f"Valid Accuracy = {valid_acc}")
print(f"Test Accuracy = {test_acc}")


                    