import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
from model_test0 import *
import matplotlib.pyplot as plt
import os
import argparse

parser = argparse.ArgumentParser()




# 新增 drug_name / source_dir / target_dir
parser.add_argument('--drug_name', type=str, default='Afatinib',
                    help='Drug name to validate (e.g., AR-42, Gefitinib, Sorafenib, etc.)')
parser.add_argument('--source_dir', type=str, default='../datasets/single cell/Afatinib.csv',
                    help='Path to the cell line dataset for the specified drug')
parser.add_argument('--target_dir', type=str, default='../datasets/single cell/Target_expr_resp_z.Afatinib_tp4k.csv',
                    help='Path to the target expression dataset for the specified drug')

args = parser.parse_args()

# 在代码里使用

drug_name = args.drug_name
source_dir = args.source_dir
target_dir = args.target_dir



# ====================== Hyperparameter Configuration ===========================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 120
EPOCHS = 10
LEARNING_RATE = 8e-5
WEIGHT_DECAY = 1e-5


# ====================== Data Preprocessing Function ===========================
def load_and_preprocess_data(drug_name, source_dir, target_dir):
    # Read data
    source_data = pd.read_csv(source_dir)
    target_data = pd.read_csv(target_dir)
    # Get all unique labels and sort
    labels = sorted(source_data['label'].unique())
    CLASS_NUM = len(labels)  # Number of unique labels
    # Extract source data
    x_source, r_source, l_source = [], [], []
    label_idx1 = source_data.columns.get_loc('label')
    label_idx2 = target_data.columns.get_loc('label')
    for label in labels:
        temp_x = source_data[source_data['label'] == label].iloc[:, label_idx1+1:].values
        temp_r = source_data[source_data['label'] == label].iloc[:, 1].values
        x_source.append(temp_x)
        r_source.append(temp_r)
        one_hot = np.zeros((temp_x.shape[0], CLASS_NUM))
        one_hot[:, label - 1] = 1
        l_source.append(one_hot)

    # Extract target data
    last_label = labels[-1]
    x_target = target_data[target_data['label'] == last_label].iloc[:, label_idx2+1:].values
    r_target = target_data[target_data['label'] == last_label].iloc[:, 1].values
    one_hot_target = np.zeros((x_target.shape[0], CLASS_NUM))
    one_hot_target[:, last_label - 1] = 1

    # Standardization and NaN handling
    def standardize_and_fill(matrix):
        mean = np.mean(matrix, axis=0)
        std = np.std(matrix, axis=0)
        standardized = (matrix - mean) / std
        return np.nan_to_num(standardized, nan=1e-10)

    # Merge and process
    x_source_all = np.concatenate(x_source, axis=0)
    x_source_all = standardize_and_fill(x_source_all)
    l_source_all = np.concatenate(l_source, axis=0)
    r_source_all = np.concatenate(r_source, axis=0)

    x_target = standardize_and_fill(x_target)

    # Convert to tensor
    x_source_tensor = torch.tensor(x_source_all, dtype=torch.float32)
    l_source_tensor = torch.tensor(l_source_all, dtype=torch.float32)
    r_source_tensor = torch.tensor(r_source_all, dtype=torch.float32)
    x_target_tensor = torch.tensor(x_target, dtype=torch.float32)
    l_target_tensor = torch.tensor(one_hot_target, dtype=torch.float32)
    r_target_tensor = torch.tensor(r_target, dtype=torch.float32)

    # Data loader
    source_loader = DataLoader(TensorDataset(x_source_tensor, l_source_tensor, r_source_tensor),
                               batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    target_loader = DataLoader(TensorDataset(x_target_tensor, l_target_tensor, r_target_tensor),
                               batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    input_dim = x_source_tensor.shape[1]
    return source_loader, target_loader, input_dim, CLASS_NUM

# ====================== Training Function ===========================
def train_model(source_loader, target_loader, input_dim, CLASS_NUM):
    # Initialize model
    model = Model(input_dim, 1024, 740, CLASS_NUM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    CEloss = nn.BCEWithLogitsLoss()
    adversarial_loss = nn.CrossEntropyLoss()

    # Recorders
    train_losses, test_aucs = [], []
    total_feature = []

    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss, preds, labels = 0, [], []

        for x_batch, l_batch, r_batch in source_loader:
            x_batch, l_batch, r_batch = x_batch.to(device), l_batch.to(device), r_batch.to(device)
            c2, d3, trip, _ = model(x_batch, r_batch)
            c2 = c2.flatten()

            # Loss calculation
            loss_cls = CEloss(c2, r_batch)
            loss_adv = adversarial_loss(d3, l_batch)
            total_loss = loss_cls + loss_adv + trip

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Record losses and predictions
            epoch_loss += total_loss.item()
            preds.extend(c2.detach().cpu().numpy())
            labels.extend(r_batch.cpu().numpy())

        # Training AUC
        train_auc = roc_auc_score(labels, preds)
        train_losses.append(epoch_loss / len(source_loader))

        # Evaluation on target data
        model.eval()
        test_preds, test_labels = [], []

        with torch.no_grad():
            for x_batch, l_batch, r_batch in target_loader:
                x_batch, l_batch, r_batch = x_batch.to(device), l_batch.to(device), r_batch.to(device)
                c2, _, _, feature = model(x_batch, r_batch)
                test_preds.extend(c2.flatten().cpu().numpy())
                test_labels.extend(r_batch.cpu().numpy())
                total_feature.append(feature.cpu())

        test_auc = roc_auc_score(test_labels, test_preds)
        test_aucs.append(test_auc)

        print(f"Epoch {epoch + 1}/{EPOCHS} | Loss: {epoch_loss:.4f} | Train AUC: {train_auc:.4f} | Test AUC: {test_auc:.4f}")

    return train_losses, test_aucs, torch.cat(total_feature, dim=0).numpy()

# ====================== Plotting Function ===========================
def plot_results(train_losses, test_aucs, drug_name):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_aucs, label='Test AUC')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.legend()
    plt.title(f'{drug_name} Training Loss and Test AUC')
    plt.show()

# ====================== Main Pipeline ===========================
def run_pipeline(drug_name, source_dir, target_dir, save_feature_path=None):
    # Data preparation
    source_loader, target_loader, input_dim, CLASS_NUM = load_and_preprocess_data(drug_name, source_dir, target_dir)
    # Training
    train_losses, test_aucs, features = train_model(source_loader, target_loader, input_dim, CLASS_NUM)
    # Visualization
    plot_results(train_losses, test_aucs, drug_name)
    # Feature saving (optional)
    if save_feature_path:
        np.save(save_feature_path, features)
        print(f"Features saved to {save_feature_path}")
run_pipeline(drug_name, source_dir, target_dir, None)