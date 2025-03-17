import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
from sklearn.utils import shuffle
from torch.utils.data import TensorDataset
from model_test import *
import torch.nn as nn
from itertools import cycle
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
device = torch.device("cuda:0")
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
import umap
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x).float()
        self.y = torch.tensor(y).float()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return (self.x[idx], self.y[idx])



target_data = pd.read_csv('/data/sr/supplement/Bortezomib_target.csv')



t_1 = target_data[target_data['Label'] == 22].iloc[:, 3:].values
t_2 = target_data[target_data['Label'] == 23].iloc[:, 3:].values
t_3 = target_data[target_data['Label'] == 24].iloc[:, 3:].values
t_4 = target_data[target_data['Label'] == 25].iloc[:, 3:].values


t_r1 = target_data[target_data['Label'] == 22].iloc[:, 1].values
t_r2 = target_data[target_data['Label'] == 23].iloc[:, 1].values
t_r3 = target_data[target_data['Label'] == 24].iloc[:, 1].values
t_r4 = target_data[target_data['Label'] == 25].iloc[:, 1].values

def standardize_matrix(matrix):
    # 计算每一列的均值
    mean = np.mean(matrix, axis=0)
    # 计算每一列的标准差
    std = np.std(matrix, axis=0)
    # 标准化矩阵
    standardized_matrix = (matrix - mean) / std
    return standardized_matrix
def fill_nan_with_small_number(matrix, small_number=1e-10):
    # 使用 np.isnan 检查 NaN 值，并将 NaN 值替换为 small_number
    matrix = np.where(np.isnan(matrix), small_number, matrix)
    return matrix
result_list = []
l = [x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18, x_19, x_20, x_21, x_22, t_1, t_2, t_3, t_4]
# 循环从 x1 到 x11
for i in range(22):
    # 初始化一个全零张量
    zeros_tensor = torch.zeros((l[i].shape[0], 22))
    # 在第 i 列插入全一张量
    zeros_tensor[:, i] = 1
    # 将生成的张量加入到结果列表中
    l[i] = zeros_tensor
zeros_tensor1 = torch.zeros((l[22].shape[0], 22))
zeros_tensor2 = torch.zeros((l[23].shape[0], 22))
zeros_tensor3 = torch.zeros((l[24].shape[0], 22))
zeros_tensor4 = torch.zeros((l[25].shape[0], 22))
zeros_tensor1[:, 21] = 1
zeros_tensor2[:, 21] = 1
zeros_tensor3[:, 21] = 1
zeros_tensor4[:, 21] = 1
l[22] = zeros_tensor1
l[23] = zeros_tensor2
l[24] = zeros_tensor3
l[25] = zeros_tensor4



s1 = np.concatenate((x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18, x_19, x_20, x_21, x_22), axis=0)
s1 = standardize_matrix(s1)
s1 = fill_nan_with_small_number(s1)
l1 = torch.cat((l[0], l[1], l[2], l[3], l[4], l[5], l[6], l[7], l[8], l[9], l[10], l[11], l[12], l[13], l[14], l[15], l[16], l[17], l[18], l[19], l[20], l[21]), dim=0)
s2 = np.concatenate((t_1,), axis=0)
s2 = standardize_matrix(s2)
s2 = fill_nan_with_small_number(s2)
l2 = torch.cat((l[22],), dim=0)

bs = 500

r1 = np.concatenate((r_1, r_2, r_3, r_4, r_5, r_6, r_7, r_8, r_9, r_10, r_11, r_12, r_13, r_14, r_15, r_16, r_17, r_18, r_19, r_20, r_21, r_22), axis=0)
r1 = 1 - r1
r2 = np.concatenate((t_r1,), axis=0)

s1 = torch.Tensor(s1)
s2 = torch.Tensor(s2)

r1 = torch.Tensor(r1)
r2 = torch.Tensor(r2)

s_train = TensorDataset(s1, l1, r1,)
s_loader = DataLoader(s_train, batch_size=bs, shuffle=False, drop_last=False)
s_test = TensorDataset(s2, l2, r2,)
s_test = DataLoader(s_test, batch_size=bs, shuffle=False, drop_last=False)


CEloss = nn.BCEWithLogitsLoss()
adversarial_loss = nn.CrossEntropyLoss()
model = Model(s1.shape[1], 1024,740,).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=8e-5, weight_decay=1e-5)

losses = []
tlosses = []
auc_scores = []
tauc_scores = []
epochs = 100
best = 0

model.load_state_dict(torch.load('/data/sr/Bortezomib_0.pth'))
with torch.no_grad():
    tlabel_ = torch.Tensor().to(device)
    tpred_ = torch.Tensor().to(device)
    total_feature = torch.Tensor().to(device)

    for i, data_t in enumerate(s_test):
        t1_ = data_t[0].to(device)
        tl_ = data_t[1].to(device)
        tr_ = data_t[2].to(device)

        tc2, td3, t_trip, feature = model(t1_, tr_, )

        tlabel_ = torch.cat((tlabel_, tr_), dim=0)

        tpred_ = torch.cat((tpred_, tc2), dim=0)
        total_feature = torch.cat((total_feature, feature), dim=0)


tlabel_ = tlabel_.cpu().numpy()
tpred_ = tpred_.cpu().numpy()
x = range(1, epochs+1)

plt.plot(x, auc_scores, marker='.', linestyle='-', color='b')

plt.xlim(0, max(x))
plt.ylim(0, 1)

plt.title('AUC Scores over Epochs')
plt.xlabel('Epoch')
plt.ylabel('AUC Score')


plt.grid(True)


plt.show()


plt.plot(x, losses, marker='.', linestyle='-', color='b')


plt.title('losses over Epochs')
plt.xlabel('Epoch')
plt.ylabel('loss')


plt.grid(True)


plt.show()