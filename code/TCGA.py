import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
from model_test import * 
import matplotlib.pyplot as plt
import os

# ====================== Execution Example ===========================
if __name__ == "__main__":
    drug_name = 'Cisplatin'  # Change drug name here: Temozolomide, Paclitaxel, Gemcitabine, Docetaxel
    source_dir = '/data/sr'
    target_dir = '/data/sr'
    save_feature_path = f'/data/sr/{drug_name}_final_features.npy'
    run_pipeline(drug_name, source_dir, target_dir, save_feature_path)


# ====================== Hyperparameter Configuration ===========================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 381
EPOCHS = 200
LEARNING_RATE = 8e-5
WEIGHT_DECAY = 1e-5
CLASS_NUM = 22  # Number of drug categories


# ====================== Data Preprocessing Function ===========================
def load_and_preprocess_data(drug_name, source_dir, target_dir):
    # Auto path joining
    source_path = os.path.join(source_dir, f'{drug_name}_matched_data.csv')
    target_path = os.path.join(target_dir, f'{drug_name}_patients.csv')

    # Read data
    source_data = pd.read_csv(source_path)
    target_data = pd.read_csv(target_path)

    # Extract source data
    x_source, r_source, l_source = [], [], []
    for label in range(1, CLASS_NUM + 1):
        temp_x = source_data[source_data['label'] == label].iloc[:, 5:10005].values
        temp_r = source_data[source_data['label'] == label].iloc[:, 2].values
        x_source.append(temp_x)
        r_source.append(temp_r)
        one_hot = np.zeros((temp_x.shape[0], CLASS_NUM))
        one_hot[:, label - 1] = 1
        l_source.append(one_hot)

    # Extract target data
    x_target = target_data[target_data['label'] == CLASS_NUM].iloc[:, 4:10004].values
    r_target = target_data[target_data['label'] == CLASS_NUM].iloc[:, 2].values
    one_hot_target = np.zeros((x_target.shape[0], CLASS_NUM))
    one_hot_target[:, CLASS_NUM - 1] = 1

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
    return source_loader, target_loader, input_dim


# ====================== Training Function ===========================
def train_model(source_loader, target_loader, input_dim):
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
    source_loader, target_loader, input_dim = load_and_preprocess_data(drug_name, source_dir, target_dir)
    # Training
    train_losses, test_aucs, features = train_model(source_loader, target_loader, input_dim)
    # Visualization
    plot_results(train_losses, test_aucs, drug_name)
    # Feature saving (optional)
    if save_feature_path:
        np.save(save_feature_path, features)
        print(f"Features saved to {save_feature_path}")



