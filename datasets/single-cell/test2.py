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
from sklearn.metrics import roc_auc_score
device = torch.device("cuda:1")
from sklearn.decomposition import PCA


class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x).float()
        self.y = torch.tensor(y).float()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return (self.x[idx], self.y[idx])


source_data = pd.read_csv('/data/sr/updated_output3.csv')
#Target_data = pd.read_csv('/data/sr/Target_expr_resp_z.Etoposide_tp4k.tsv', sep='\t')
#source2_data = pd.read_excel('/data/sr/Etoposide_tp4k_2.xlsx')

#print(source3_data)

x_1 = source_data[source_data['label'] == 1].iloc[:, 5:].values
x_2 = source_data[source_data['label'] == 2].iloc[:, 5:].values
x_3 = source_data[source_data['label'] == 3].iloc[:, 5:].values
x_4 = source_data[source_data['label'] == 4].iloc[:, 5:].values
x_5 = source_data[source_data['label'] == 5].iloc[:, 5:].values
x_6 = source_data[source_data['label'] == 6].iloc[:, 5:].values
x_7 = source_data[source_data['label'] == 7].iloc[:, 5:].values
x_8 = source_data[source_data['label'] == 8].iloc[:, 5:].values
x_9 = source_data[source_data['label'] == 9].iloc[:, 5:].values
x_10 = source_data[source_data['label'] == 10].iloc[:, 5:].values
x_11 = source_data[source_data['label'] == 11].iloc[:, 5:].values
r_1 = source_data[source_data['label'] == 1].iloc[:, 1].values
r_2 = source_data[source_data['label'] == 2].iloc[:, 1].values
r_3 = source_data[source_data['label'] == 3].iloc[:, 1].values
r_4 = source_data[source_data['label'] == 4].iloc[:, 1].values
r_5 = source_data[source_data['label'] == 5].iloc[:, 1].values
r_6 = source_data[source_data['label'] == 6].iloc[:, 1].values
r_7 = source_data[source_data['label'] == 7].iloc[:, 1].values
r_8 = source_data[source_data['label'] == 8].iloc[:, 1].values
r_9 = source_data[source_data['label'] == 9].iloc[:, 1].values
r_10 = source_data[source_data['label'] == 10].iloc[:, 1].values
r_11 = source_data[source_data['label'] == 11].iloc[:, 1].values
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
result_list = []
x = {}
# 循环从 x1 到 x11
for i in range(11):
    # 初始化一个全零张量
    zeros_tensor = torch.zeros((source_data[source_data['label'] == i+1].shape[0], 11))
    # 在第 i 列插入全一张量
    zeros_tensor[:, i] = 1
    # 将生成的张量加入到结果列表中
    x[f'x{i + 1}_label'] = zeros_tensor
print(x['x3_label'])
# 将列表中的张量连接起来，沿着列的方向
print(x['x1_label'])


s1 = np.concatenate((x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10), axis=0)
l1 = torch.cat((x['x1_label'], x['x2_label'], x['x3_label'], x['x4_label'], x['x5_label'], x['x6_label'], x['x7_label'], x['x8_label'], x['x9_label'], x['x10_label']), dim=0)
s2 = np.concatenate((x_11,), axis=0)
l2 = torch.cat((x['x11_label'],), dim=0)
# t1 = np.concatenate((x2_0,), axis=0)
# t2 = np.concatenate((x2_1,), axis=0)
# tl1 = torch.cat((x2_0label,), dim=0)
# tl2 = torch.cat((x2_1label,), dim=0)
# pca = PCA(n_components=800)  # 选择要降到的目标维度
# pca_s1 = pca.fit_transform(s1)
# pca_s2 = pca.fit_transform(s2)
bs = 32
# y1_0 = source_data[source_data['response'] == 0].iloc[:, 1:2].values
# y1_1 = source_data[source_data['response'] == 1].iloc[:, 1:2].values
# y2_0 = Target_data[Target_data['response'] == 0].iloc[:, 1:2].values
# y2_1 = Target_data[Target_data['response'] == 1].iloc[:, 1:2].values
#y3_0 = source3_data[source3_data['response'] == 0].iloc[:, 1:2].values
#y3_1 = source3_data[source3_data['response'] == 1].iloc[:, 1:2].values
r1 = np.concatenate((r_1, r_2, r_3, r_4, r_5, r_6, r_7, r_8, r_9, r_10 ), axis=0)
r2 = np.concatenate((r_11,), axis=0)
# tr0 = np.concatenate((y2_0,), axis=0)
# tr1 = np.concatenate((y2_1,), axis=0)
s1 = torch.Tensor(s1)
s2 = torch.Tensor(s2)
# t1 = torch.Tensor(t1)
# t2 = torch.Tensor(t2)
r1 = torch.Tensor(r1)
r2 = torch.Tensor(r2)
# tr0 = torch.Tensor(tr0)
# tr1 = torch.Tensor(tr1)
#s1_train,l1_train,r0_train = train_test_split(s1,l1,r0,random_state=66)
#s2_train,l2_train,r1_train = train_test_split(s2,l2,r1,random_state=66)
s_train = TensorDataset(s1, l1, r1)
s_loader = DataLoader(s_train, batch_size=bs, shuffle=True, drop_last=True)
s_test = TensorDataset(s2, l2, r2)
s_test = DataLoader(s_test, batch_size=bs, shuffle=True)
# s2_train = TensorDataset(s2, l2, r1)
# s2_loader = DataLoader(s2_train, batch_size=bs, shuffle=True, drop_last=True)
# s2_test = TensorDataset(t2, tl2, tr1)
# s2_test = DataLoader(s2_test,batch_size=bs,shuffle=True)


CEloss = nn.BCEWithLogitsLoss()
adversarial_loss = nn.CrossEntropyLoss()
model = Model(s1.shape[1], 1024, 800).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-6, weight_decay=1e-6)
#discriminator_optimizer = torch.optim.Adam(Model.parameters(), lr=0.0002, betas=(0.5, 0.999))




losses = []
tlosses = []
auc_scores = []
tauc_scores = []
epochs = 600
for epo in range(epochs):

    pred_ = []
    label_ = []
    for i, data in enumerate(s_loader):
        s1_ = data[0].to(device)
        l1_ = data[1].to(device)
        r1_ = data[2].to(device)
        c2, d3, trip = model(s1_, r1_)
        c2 = c2.flatten()
        loss2 = CEloss(c2, r1_)
        loss3 = adversarial_loss(d3, l1_)
        total_loss = loss2+trip+loss3
        r1_ = r1_
        label_0 = r1_.cpu().numpy()
        c2_0 = c2
        c2_0 = c2_0.detach().cpu().numpy()
        label_.extend(label_0)
        pred_.extend(c2_0)
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
    loss_ = total_loss.item()
    auc = roc_auc_score(label_, pred_)
    auc_scores.append(auc)
    losses.append(loss_)
    with torch.no_grad():
        tlabel_ = torch.Tensor().to(device)
        tpred_ = torch.Tensor().to(device)

        for i, data_t in enumerate(s_test):
            t1_ = data_t[0].to(device)
            tl_ = data_t[1].to(device)
            tr_ = data_t[2].to(device)
            tc2, td3, t_trip = model(t1_, tr_)

            tlabel_ = torch.cat((tlabel_, tr_), dim=0)

            tpred_ = torch.cat((tpred_, tc2), dim=0)

        # print(tlabel_)
        # print(tpred_)
    tlabel_ = tlabel_.cpu().numpy()
    tpred_ = tpred_.cpu().numpy()
    tauc = roc_auc_score(tlabel_, tpred_)
    tauc_scores.append(tauc)
x = range(1, 601)
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

#生成loss曲线
plt.plot(x, losses, marker='.', linestyle='-', color='b')

# 添加标题和轴标签
plt.title('losses over Epochs')
plt.xlabel('Epoch')
plt.ylabel('loss')

# 显示网格
plt.grid(True)

# 显示图表
plt.show()
#
#
# with torch.no_grad():
#     tlabel_ = torch.Tensor().to(device)
#     tpred_ = torch.Tensor().to(device)
#     tauc_scores = []
#
#     for i, data_t in enumerate(s_test):
#         t1_ = data_t[0].to(device)
#         tl_ = data_t[1].to(device)
#         tr_ = data_t[2].to(device)
#         tc2, td3, t_trip = model(t1_, tr_)
#
#         tlabel_ = torch.cat((tlabel_, tr_),dim=0)
#
#         tpred_ = torch.cat((tpred_, tc2),dim=0)
#
#
#     print(tlabel_)
#     print(tpred_)
# tlabel_ = tlabel_.cpu().numpy()
# tpred_ = tpred_.cpu().numpy()
# tauc = roc_auc_score(tlabel_, tpred_)
# print(tauc)


    # print(auc_scores)
# # 合并特征和标签
# x1_test = np.concatenate([x1_0_test, x1_1_test], axis=0)
# test1_labels = np.concatenate([y1_0_test, y1_1_test], axis=0)
# x2_test = np.concatenate([x2_0_test, x2_1_test], axis=0)
# test2_labels = np.concatenate([y2_0_test, y2_1_test], axis=0)
# x3_test = np.concatenate([x3_0_test, x3_1_test], axis=0)
# test3_labels = np.concatenate([y3_0_test, y3_1_test], axis=0)
# # 随机打乱
# x1_test, test1_labels = shuffle(x1_test, test1_labels, random_state=42)
# x2_test, test2_labels = shuffle(x2_test, test2_labels, random_state=42)
# x3_test, test3_labels = shuffle(x3_test, test3_labels, random_state=42)


# train1_0_dataset = CustomDataset(x1_0_train, y1_0_train)
# train1_1_dataset = CustomDataset(x1_1_train, y1_1_train)
# train2_0_dataset = CustomDataset(x2_0_train, y2_0_train)
# train2_1_dataset = CustomDataset(x2_1_train, y2_1_train)
# train3_0_dataset = CustomDataset(x3_0_train, y3_0_train)
# train3_1_dataset = CustomDataset(x3_1_train, y3_1_train)
# train1_0_loader = DataLoader(train1_0_dataset, batch_size=32, shuffle=False)
# train1_1_loader = DataLoader(train1_1_dataset, batch_size=32, shuffle=False)
# train2_0_loader = DataLoader(train2_0_dataset, batch_size=32, shuffle=False)
# train2_1_loader = DataLoader(train2_1_dataset, batch_size=32, shuffle=False)
# train3_0_loader = DataLoader(train3_0_dataset, batch_size=32, shuffle=False)
# train3_1_loader = DataLoader(train3_1_dataset, batch_size=32, shuffle=False)
# test1_dataset = CustomDataset(x1_test, test1_labels)
# test2_dataset = CustomDataset(x2_test, test2_labels)
# test3_dataset = CustomDataset(x3_test, test3_labels)
# test1_loader = DataLoader(test1_dataset, batch_size=32, shuffle=False)
# test2_loader = DataLoader(test2_dataset, batch_size=32, shuffle=False)
# test3_loader = DataLoader(test3_dataset, batch_size=32, shuffle=False)
#
# feature10 = FeatureExtractor(train1_0_loader.dataset.x.shape[1],1024,256)
# feature1 = FeatureExtractor(train1_1_loader.dataset.x.shape[1],1024,256)
#
# SSID = features
# for j in enumerate(SSID):
#     SSID[j] = GRL(SSID[j])



"""my_model = Model(test1_loader.shape[0], 1024, 256)

optimizer = torch.optim.Adagrad(my_model.parameters(), lr=0.0001, weight_decay=0.001)
prediction_loss = nn.BCELoss()

losses = []
test_losses = []
loss = 0
for epoch in tqdm(range(100)):

    my_model.train()

    for i, data in enumerate(train_loader):
        # print(i)
        # print(data)
        exprs = data[0]
        label = data[1]

        pred = my_model(exprs)
        # print(pred)
        pred = pred.squeeze()
        # print(pred)
        # print(len(pred))

        loss = prediction_loss(pred, label)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    for i, data in enumerate(test_loader):
        # print(i)
        # print(data)
        exprs = data[0]
        label = data[1]

        pred = my_model(exprs)
        # print(pred)
        pred = pred.squeeze()
        # print(pred)
        # print(len(pred))

        test_loss = prediction_loss(pred, label)

    losses.append(loss.item())
    test_losses.append(test_loss.item())
print(test_losses)
print(losses)

# 绘制折线图
# plt.plot(losses)

# 设置图像标题和坐标轴标签
# plt.title('loss')
# plt.xlabel('epoch')
# plt.ylabel('loss')
#
# plt.savefig('loss_plot.png')


plt.plot(test_losses)

# 设置图像标题和坐标轴标签
plt.title('test_losses')
plt.xlabel('epoch')
plt.ylabel('test_loss')
# plt.show()
plt.savefig('test_loss_plot.png')
"""