import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report

# Data Paths
data_path = "train_embeddings.csv"
label_path = "train_labels.csv"
test_data_path = "test_embeddings.csv"
test_label_path = "test_labels.csv"
valid_data_path = "valid_embeddings.csv"
valid_label_path = "valid_labels.csv"

# ------------ TRAIN -----------

# Load file name to index dictionary
with open('train_dic.pkl', 'rb') as f:
    name_to_idx = pickle.load(f)

# Load and Training Data
X_train = np.loadtxt(data_path, delimiter=',')
y_train_mixed = pd.read_csv(label_path)

# Fix training labels
y_train = np.zeros(np.shape(y_train_mixed)[0])
for index, row in y_train_mixed.iterrows():
    y_train[name_to_idx[row["id"]]] = row["label"]


# Train Model
logreg = LogisticRegression(random_state=16, max_iter=100000)
logreg.fit(X_train, y_train)
y_preds = logreg.predict(X_train)
print("---------TRAIN------------")
cnf_matrix = metrics.confusion_matrix(y_train, y_preds)
print(classification_report(y_train, y_preds))
print(cnf_matrix)

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

# Predict
y_pred = logreg.predict(X_test)
y_pred_valid = logreg.predict(X_valid)



print("---------TEST------------")
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))
print(cnf_matrix)

print("---------VALIDATION------------")
cnf_matrix = metrics.confusion_matrix(y_valid, y_pred_valid)
print(classification_report(y_valid, y_pred_valid))
print(cnf_matrix)
                    