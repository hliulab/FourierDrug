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
from sklearn.metrics import roc_auc_score, roc_curve
device = torch.device("cuda:0")
from sklearn.decomposition import PCA
import umap

class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x).float()
        self.y = torch.tensor(y).float()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return (self.x[idx], self.y[idx])


source_data = pd.read_csv('/data/sr/Cisplatin_matched_data.csv')
target_data = pd.read_csv('/data/sr/Cisplatin_patients.csv')
#Target_data = pd.read_csv('/data/sr/Target_expr_resp_z.Etoposide_tp4k.tsv', sep='\t')
#source2_data = pd.read_excel('/data/sr/Etoposide_tp4k_2.xlsx')

#print(source3_data)

x_1 = source_data[source_data['label'] == 1].iloc[:, 5:10005].values
x_2 = source_data[source_data['label'] == 2].iloc[:, 5:10005].values
x_3 = source_data[source_data['label'] == 3].iloc[:, 5:10005].values
x_4 = source_data[source_data['label'] == 4].iloc[:, 5:10005].values
x_5 = source_data[source_data['label'] == 5].iloc[:, 5:10005].values
x_6 = source_data[source_data['label'] == 6].iloc[:, 5:10005].values
x_7 = source_data[source_data['label'] == 7].iloc[:, 5:10005].values
x_8 = source_data[source_data['label'] == 8].iloc[:, 5:10005].values
x_9 = source_data[source_data['label'] == 9].iloc[:, 5:10005].values
x_10 = source_data[source_data['label'] == 10].iloc[:, 5:10005].values
x_11 = source_data[source_data['label'] == 11].iloc[:, 5:10005].values
x_12 = source_data[source_data['label'] == 12].iloc[:, 5:10005].values
x_13 = source_data[source_data['label'] == 13].iloc[:, 5:10005].values
x_14 = source_data[source_data['label'] == 14].iloc[:, 5:10005].values
x_15 = source_data[source_data['label'] == 15].iloc[:, 5:10005].values
x_16 = source_data[source_data['label'] == 16].iloc[:, 5:10005].values
x_17 = source_data[source_data['label'] == 17].iloc[:, 5:10005].values
x_18 = source_data[source_data['label'] == 18].iloc[:, 5:10005].values
x_19 = source_data[source_data['label'] == 19].iloc[:, 5:10005].values
x_20 = source_data[source_data['label'] == 20].iloc[:, 5:10005].values
x_21 = source_data[source_data['label'] == 21].iloc[:, 5:10005].values
x_22 = source_data[source_data['label'] == 22].iloc[:, 5:10005].values
# x_23 = source_data[source_data['label'] == 23].iloc[:, 5:15005].values
t_1 = target_data[target_data['label'] == 22].iloc[:, 4:10004].values
r_1 = source_data[source_data['label'] == 1].iloc[:, 2].values
r_2 = source_data[source_data['label'] == 2].iloc[:, 2].values
r_3 = source_data[source_data['label'] == 3].iloc[:, 2].values
r_4 = source_data[source_data['label'] == 4].iloc[:, 2].values
r_5 = source_data[source_data['label'] == 5].iloc[:, 2].values
r_6 = source_data[source_data['label'] == 6].iloc[:, 2].values
r_7 = source_data[source_data['label'] == 7].iloc[:, 2].values
r_8 = source_data[source_data['label'] == 8].iloc[:, 2].values
r_9 = source_data[source_data['label'] == 9].iloc[:, 2].values
r_10 = source_data[source_data['label'] == 10].iloc[:, 2].values
r_11 = source_data[source_data['label'] == 11].iloc[:, 2].values
r_12 = source_data[source_data['label'] == 12].iloc[:, 2].values
r_13 = source_data[source_data['label'] == 13].iloc[:, 2].values
r_14 = source_data[source_data['label'] == 14].iloc[:, 2].values
r_15 = source_data[source_data['label'] == 15].iloc[:, 2].values
r_16 = source_data[source_data['label'] == 16].iloc[:, 2].values
r_17 = source_data[source_data['label'] == 17].iloc[:, 2].values
r_18 = source_data[source_data['label'] == 18].iloc[:, 2].values
r_19 = source_data[source_data['label'] == 19].iloc[:, 2].values
r_20 = source_data[source_data['label'] == 20].iloc[:, 2].values
r_21 = source_data[source_data['label'] == 21].iloc[:, 2].values
r_22 = source_data[source_data['label'] == 22].iloc[:, 2].values
# r_23 = source_data[source_data['label'] == 23].iloc[:, 2].values
t_r = target_data[target_data['label'] == 22].iloc[:, 2].values
# ic_1 = source_data[source_data['label'] == 1].iloc[:, 1].values
# ic_2 = source_data[source_data['label'] == 2].iloc[:, 1].values
# ic_3 = source_data[source_data['label'] == 3].iloc[:, 1].values
# ic_4 = source_data[source_data['label'] == 4].iloc[:, 1].values
# ic_5 = source_data[source_data['label'] == 5].iloc[:, 1].values
# ic_6 = source_data[source_data['label'] == 6].iloc[:, 1].values
# ic_7 = source_data[source_data['label'] == 7].iloc[:, 1].values
# ic_8 = source_data[source_data['label'] == 8].iloc[:, 1].values
# ic_9 = source_data[source_data['label'] == 9].iloc[:, 1].values
# ic_10 = source_data[source_data['label'] == 10].iloc[:, 1].values
# ic_11 = source_data[source_data['label'] == 11].iloc[:, 1].values
# ic_12 = source_data[source_data['label'] == 12].iloc[:, 1].values
# ic_13 = source_data[source_data['label'] == 13].iloc[:, 1].values
# ic_14 = source_data[source_data['label'] == 14].iloc[:, 1].values
# ic_15 = source_data[source_data['label'] == 15].iloc[:, 1].values
# ic_16 = source_data[source_data['label'] == 16].iloc[:, 1].values
# ic_17 = source_data[source_data['label'] == 17].iloc[:, 1].values
# ic_18 = source_data[source_data['label'] == 18].iloc[:, 1].values
# ic_19 = source_data[source_data['label'] == 19].iloc[:, 1].values
# ic_20 = source_data[source_data['label'] == 20].iloc[:, 1].values
# ic_21 = source_data[source_data['label'] == 21].iloc[:, 1].values
# ic_22 = source_data[source_data['label'] == 22].iloc[:, 1].values
#x1_label = torch.cat((torch.ones((x_1.shape[0], 1)), torch.zeros((x_1.shape[0], 1)), torch.zeros((x_1.shape[0], 1)), torch.zeros((x_1.shape[0], 1)), torch.zeros((x_1.shape[0], 1)), torch.zeros((x_1.shape[0], 1)), torch.zeros((x_1.shape[0], 1)), torch.zeros((x_1.shape[0], 1)), torch.zeros((x_1.shape[0], 1)), torch.zeros((x_1.shape[0], 1)), torch.zeros((x_1.shape[0], 1))), dim=1)
#x1_1label = torch.cat((torch.ones((x1_1.shape[0],1)),torch.zeros((x1_1.shape[0],1))),dim=1)
#x2_0 = Target_data[Target_data['response'] == 0].iloc[:, 2:].values
#x2_0label = torch.cat((torch.zeros((x2_0.shape[0],1)),torch.ones((x2_0.shape[0],1))),dim=1)
#x2_1 = Target_data[Target_data['response'] == 1].iloc[:, 2:].values
#x2_1label = torch.cat((torch.zeros((x2_1.shape[0],1)),torch.ones((x2_1.shape[0],1))),dim=1)
# x3_0 = source3_data[source3_data['response'] == 0].iloc[:, 2:].values
# x3_0label = torch.cat((torch.zeros((x3_0.shape[0],1)),torch.zeros((x3_0.shape[0],1)),torch.ones((x3_0.shape[0],1))),dim=1)
# x3_1 = source3_data[source3_data['response'] == 1].iloc[:, 2:].values
# x3_1label = torch.cat((torch.zeros((x3_1.shape[0],1)),torch.zeros((x3_1.shape[0],1)),torch.ones((x3_1.shape[0],1))),dim=1)
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
# 循环从 x1 到 x11
result_list = []
l = [x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18, x_19, x_20, x_21, x_22, t_1]
# 循环从 x1 到 x11
for i in range(22):
    # 初始化一个全零张量
    zeros_tensor = torch.zeros((l[i].shape[0], 22))
    # 在第 i 列插入全一张量
    zeros_tensor[:, i] = 1
    # 将生成的张量加入到结果列表中
    l[i] = zeros_tensor
zeros_tensor = torch.zeros((l[22].shape[0], 22))
zeros_tensor[:, 21] = 1
l[22] = zeros_tensor
# print(x['x3_label'])
# 将列表中的张量连接起来，沿着列的方向
# print(x['x1_label'])

s1 = np.concatenate((x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18, x_19, x_20, x_21, x_22, ), axis=0)
s1 = standardize_matrix(s1)
s1 = fill_nan_with_small_number(s1)
l1 = torch.cat((l[0], l[1], l[2], l[3], l[4], l[5], l[6], l[7], l[8], l[9], l[10], l[11], l[12], l[13], l[14], l[15], l[16], l[17], l[18], l[19], l[20], l[21],), dim=0)
s2 = np.concatenate((t_1,), axis=0)
s2 = standardize_matrix(s2)
s2 = fill_nan_with_small_number(s2)
l2 = torch.cat((l[22],), dim=0)
# t1 = np.concatenate((x2_0,), axis=0)
# t2 = np.concatenate((x2_1,), axis=0)
# tl1 = torch.cat((x2_0label,), dim=0)
# tl2 = torch.cat((x2_1label,), dim=0)
# pca = PCA(n_components=800)  # 选择要降到的目标维度
# pca_s1 = pca.fit_transform(s1)
# pca_s2 = pca.fit_transform(s2)
bs = 381
# y1_0 = source_data[source_data['response'] == 0].iloc[:, 1:2].values
# y1_1 = source_data[source_data['response'] == 1].iloc[:, 1:2].values
# y2_0 = Target_data[Target_data['response'] == 0].iloc[:, 1:2].values
# y2_1 = Target_data[Target_data['response'] == 1].iloc[:, 1:2].values
#y3_0 = source3_data[source3_data['response'] == 0].iloc[:, 1:2].values
#y3_1 = source3_data[source3_data['response'] == 1].iloc[:, 1:2].values
r1 = np.concatenate((r_1, r_2, r_3, r_4, r_5, r_6, r_7, r_8, r_9, r_10, r_11, r_12, r_13, r_14, r_15, r_16, r_17, r_18, r_19, r_20, r_21, r_22,), axis=0)
r2 = np.concatenate((t_r,), axis=0)
# tr0 = np.concatenate((y2_0,), axis=0)
# tr1 = np.concatenate((y2_1,), axis=0)
s1 = torch.Tensor(s1)
s2 = torch.Tensor(s2)
# ic = torch.Tensor(ic_22)
# t1 = torch.Tensor(t1)
# t2 = torch.Tensor(t2)
r1 = torch.Tensor(r1)
r2 = torch.Tensor(r2)
# tr0 = torch.Tensor(tr0)
# tr1 = torch.Tensor(tr1)
#s1_train,l1_train,r0_train = train_test_split(s1,l1,r0,random_state=66)
#s2_train,l2_train,r1_train = train_test_split(s2,l2,r1,random_state=66)
s_train = TensorDataset(s1, l1, r1)
s_loader = DataLoader(s_train, batch_size=bs, shuffle=False, drop_last=False)
s_test = TensorDataset(s2, l2, r2)
s_test = DataLoader(s_test, batch_size=bs, shuffle=False, drop_last=False )
# s2_train = TensorDataset(s2, l2, r1)
# s2_loader = DataLoader(s2_train, batch_size=bs, shuffle=True, drop_last=True)
# s2_test = TensorDataset(t2, tl2, tr1)
# s2_test = DataLoader(s2_test,batch_size=bs,shuffle=True)


CEloss = nn.BCEWithLogitsLoss()
adversarial_loss = nn.CrossEntropyLoss()
model = Model(s1.shape[1], 1024,740, ).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=8e-5, weight_decay=1e-5)
#discriminator_optimizer = torch.optim.Adam(Model.parameters(), lr=0.0002, betas=(0.5, 0.999))




losses = []
tlosses = []
auc_scores = []
tauc_scores = []
epochs = 200
for epo in range(epochs):

    pred_ = []
    label_ = []

    for i, data in enumerate(s_loader):
        s1_ = data[0].to(device)
        l1_ = data[1].to(device)
        r1_ = data[2].to(device)
        c2, d3, trip, feature = model(s1_, r1_, )
        c2 = c2.flatten()
        loss2 = CEloss(c2, r1_)
        loss3 = adversarial_loss(d3, l1_)
        total_loss = loss2+loss3+trip
        # r1_ = r1_
        label_0 = r1_.cpu().numpy()
        c2_0 = c2
        c2_0 = c2_0.detach().cpu().numpy()
        label_.extend(label_0)
        pred_.extend(c2_0)
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
    loss_ = total_loss.item()
#     print(label_)
    # print(pred_)
    auc = roc_auc_score(label_, pred_)
    auc_scores.append(auc)
    losses.append(loss_)
    with torch.no_grad():
        tlabel_ = torch.Tensor().to(device)
        tpred_ = torch.Tensor().to(device)
        total_feature = torch.Tensor().to(device)
        # total_ic = torch.Tensor().to(device)
        # total_labels = torch.Tensor()
        for i, data_t in enumerate(s_test):
            t1_ = data_t[0].to(device)
            tl_ = data_t[1].to(device)
            tr_ = data_t[2].to(device)
            # tic = data_t[3].to(device)
            tc2, td3, t_trip, feature = model(t1_, tr_, )
            # total_ic = torch.cat((total_ic, tic), dim=0)
            tlabel_ = torch.cat((tlabel_, tr_), dim=0)

            tpred_ = torch.cat((tpred_, tc2), dim=0)
            total_feature = torch.cat((total_feature, feature), dim=0)

        # print(tlabel_)
        # print(tpred_)
    tlabel_ = tlabel_.cpu().numpy()
    tpred_ = tpred_.cpu().numpy()
    tauc = roc_auc_score(tlabel_, tpred_)
    print(tauc)
    tauc_scores.append(tauc)
# total_labels = total_labels.cpu().numpy()
total_feature = total_feature.cpu().numpy()
# total_ic = total_ic.cpu().numpy()
tpr, fpr, _ = roc_curve(tlabel_, tpred_)
pd.DataFrame(tlabel_).to_csv('Cisplatin_tlable.csv')
pd.DataFrame(tpred_).to_csv('Cisplatin_tprde.csv')
pd.DataFrame(total_feature).to_csv('Cisplatin_features.csv')
pd.DataFrame(auc_scores).to_csv('Cisplatin_auc.csv')
# pd.DataFrame(total_ic).to_csv('PLX4720_ic_x16.csv')
# print(tlabel_)
# print(tpred_)
plt.figure()
lw = 2
plt.plot(tpr, fpr, color='darkorange', lw=lw, label='ROC curve')
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()
x = range(1, 201)
#生成auc曲线
plt.plot(x, auc_scores, marker='.', linestyle='-', color='b')
plt.plot(x, tauc_scores, marker='.', linestyle='-', color='r')
plt.xlim(0, max(x))
plt.ylim(0, 1)
# 添加标题和轴标签
plt.title('AUC Scores over Epochs')
plt.xlabel('Epoch')
plt.ylabel('AUC Score')

# 显示网格
plt.grid(True)

# 显示图表
plt.show()