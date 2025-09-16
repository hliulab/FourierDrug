import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
from model_test0 import *
import matplotlib.pyplot as plt
import os

source_path = '/data/sr/New_Folder/cell_Afatinib.csv' #cell_AR_42, cell_Docetaxel, cell_etoposide, cell_PLX4720
test_label = 3  # Choose label 3 as test set

# ====================== Hyperparameter Configuration ===========================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 8e-5
WEIGHT_DECAY = 1e-5


# ====================== Data Preprocessing Function ===========================
def load_and_preprocess_data(source_path, test_label):
    """
    Load data and split into training and test sets based on label.
    :param source_path: Path to the source CSV file.
    :param test_label: Label to be used as test set.
    :return: train_loader, test_loader, input_dim
    """
    # Load source data
    source_data = pd.read_csv(source_path)

    # Get all unique labels and sort
    labels = sorted(source_data['label'].unique())
    CLASS_NUM = len(labels)  # Number of unique labels

    # Store per-label data
    all_x, all_r, all_label_tensor = {}, {}, {}

    # Iterate over each label to collect data
    for label in labels:
        x_data = source_data[source_data['label'] == label].iloc[:, 6:].values
        r_data = source_data[source_data['label'] == label].iloc[:, 4].values
        one_hot_label = np.zeros((x_data.shape[0], CLASS_NUM))
        one_hot_label[:, label - 1] = 1  # One-hot encode the label
        all_x[label] = x_data
        all_r[label] = r_data
        all_label_tensor[label] = one_hot_label

    # ================= Standardization and NaN Handling =================
    def standardize_and_fill(matrix):
        mean = np.mean(matrix, axis=0)
        std = np.std(matrix, axis=0)
        standardized = (matrix - mean) / (std + 1e-8)  # Avoid division by zero
        return np.nan_to_num(standardized, nan=1e-10)  # Replace NaN with a small number

    # ======================= Split Data ================================
    # Prepare training data (exclude test_label)
    train_x, train_r, train_l = [], [], []
    for label in labels:
        if label != test_label:
            train_x.append(all_x[label])
            train_r.append(all_r[label])
            train_l.append(all_label_tensor[label])
    train_x = np.concatenate(train_x, axis=0)
    train_r = np.concatenate(train_r, axis=0)
    train_l = np.concatenate(train_l, axis=0)

    # Test data
    test_x = all_x[test_label]
    test_r = all_r[test_label]
    test_l = all_label_tensor[test_label]

    # ======================= Standardize ================================
    train_x = standardize_and_fill(train_x)
    test_x = standardize_and_fill(test_x)

    # ======================= Convert to Tensor ==========================
    train_x_tensor = torch.tensor(train_x, dtype=torch.float32)
    train_r_tensor = torch.tensor(train_r, dtype=torch.float32)
    train_l_tensor = torch.tensor(train_l, dtype=torch.float32)

    test_x_tensor = torch.tensor(test_x, dtype=torch.float32)
    test_r_tensor = torch.tensor(test_r, dtype=torch.float32)
    test_l_tensor = torch.tensor(test_l, dtype=torch.float32)

    # ======================= DataLoader ================================
    train_dataset = TensorDataset(train_x_tensor, train_l_tensor, train_r_tensor)
    test_dataset = TensorDataset(test_x_tensor, test_l_tensor, test_r_tensor)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    input_dim = train_x_tensor.shape[1]  # Feature dimension

    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}, Input dimension: {input_dim}")

    return train_loader, test_loader, input_dim, CLASS_NUM

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
train_loader, test_loader, input_dim, CLASS_NUM = load_and_preprocess_data(source_path, test_label)
train_losses, test_aucs, features = train_model(train_loader, test_loader, input_dim, CLASS_NUM)
