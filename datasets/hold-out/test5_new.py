import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
from sklearn.utils import shuffle
from torch.utils.data import TensorDataset
from model_test_changed import *
import torch.nn as nn
from itertools import cycle
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

device = torch.device("cuda:0")
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
import umap
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression


class MyDataset(Dataset):
    def __init__(self, features, domain_labels, class_labels):
        self.features = torch.tensor(features).float()
        self.domain_labels = torch.tensor(domain_labels).float()
        self.class_labels = torch.tensor(class_labels).float()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.domain_labels[idx], self.class_labels[idx]


import itertools

source_data = pd.read_csv('/data/sr/updated_output4.csv')
target_data = pd.read_csv('/data/sr/Target_expr_resp_z.PLX4720_tp4k.csv')
# Target_data = pd.read_csv('/data/sr/Target_expr_resp_z.Etoposide_tp4k.tsv', sep='\t')
# source2_data = pd.read_excel('/data/sr/Etoposide_tp4k_2.xlsx')



# print(source3_data)
# 细胞系各个癌种的全部样本
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
x_12 = source_data[source_data['label'] == 12].iloc[:, 5:].values
x_13 = source_data[source_data['label'] == 13].iloc[:, 5:].values
x_14 = source_data[source_data['label'] == 14].iloc[:, 5:].values
x_15 = source_data[source_data['label'] == 15].iloc[:, 5:].values
x_16 = source_data[source_data['label'] == 16].iloc[:, 5:].values
t_1 = target_data[target_data['label'] == 17].iloc[:, 3:].values

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
r_12 = source_data[source_data['label'] == 12].iloc[:, 1].values
r_13 = source_data[source_data['label'] == 13].iloc[:, 1].values
r_14 = source_data[source_data['label'] == 14].iloc[:, 1].values
r_15 = source_data[source_data['label'] == 15].iloc[:, 1].values
r_16 = source_data[source_data['label'] == 16].iloc[:, 1].values
t_r = target_data[target_data['label'] == 17].iloc[:, 1].values

ic_1 = source_data[source_data['label'] == 1].iloc[:, 2].values
ic_2 = source_data[source_data['label'] == 2].iloc[:, 2].values
ic_3 = source_data[source_data['label'] == 3].iloc[:, 2].values
ic_4 = source_data[source_data['label'] == 4].iloc[:, 2].values
ic_5 = source_data[source_data['label'] == 5].iloc[:, 2].values
ic_6 = source_data[source_data['label'] == 6].iloc[:, 2].values
ic_7 = source_data[source_data['label'] == 7].iloc[:, 2].values
ic_8 = source_data[source_data['label'] == 8].iloc[:, 2].values
ic_9 = source_data[source_data['label'] == 9].iloc[:, 2].values
ic_10 = source_data[source_data['label'] == 10].iloc[:, 2].values
ic_11 = source_data[source_data['label'] == 11].iloc[:, 2].values
ic_12 = source_data[source_data['label'] == 12].iloc[:, 2].values
ic_13 = source_data[source_data['label'] == 13].iloc[:, 2].values
ic_14 = source_data[source_data['label'] == 14].iloc[:, 2].values
ic_15 = source_data[source_data['label'] == 15].iloc[:, 2].values
ic_16 = source_data[source_data['label'] == 16].iloc[:, 2].values
# 使用SMOTE进行采样
# smote = SMOTE(random_state=66)
# x_2, r_2 = smote.fit_resample(x_2, r_2)
# x_1, r_1 = smote.fit_resample(x_1, r_1)
# x_3, r_3 = smote.fit_resample(x_3, r_3)
# x_4, r_4 = smote.fit_resample(x_4, r_4)
# x_5, r_5 = smote.fit_resample(x_5, r_5)
# x_6, r_6 = smote.fit_resample(x_6, r_6)
# x_7, r_7 = smote.fit_resample(x_7, r_7)
# x_8, r_8 = smote.fit_resample(x_8, r_8)
# x_9, r_9 = smote.fit_resample(x_9, r_9)
# x_10, r_10 = smote.fit_resample(x_10, r_10)
# x_11, r_11 = smote.fit_resample(x_11, r_11)
# x_12, r_12 = smote.fit_resample(x_12, r_12)
# x_13, r_13 = smote.fit_resample(x_13, r_13)
# x_14, r_14 = smote.fit_resample(x_14, r_14)
# x_15, r_15 = smote.fit_resample(x_15, r_15)
# x_16, r_16 = smote.fit_resample(x_16, r_16)
# t_1, t_r = smote.fit_resample(t_1, t_r)


# x1_label = torch.cat((torch.ones((x_1.shape[0], 1)), torch.zeros((x_1.shape[0], 1)), torch.zeros((x_1.shape[0], 1)), torch.zeros((x_1.shape[0], 1)), torch.zeros((x_1.shape[0], 1)), torch.zeros((x_1.shape[0], 1)), torch.zeros((x_1.shape[0], 1)), torch.zeros((x_1.shape[0], 1)), torch.zeros((x_1.shape[0], 1)), torch.zeros((x_1.shape[0], 1)), torch.zeros((x_1.shape[0], 1))), dim=1)
# x1_1label = torch.cat((torch.ones((x1_1.shape[0],1)),torch.zeros((x1_1.shape[0],1))),dim=1)
# x2_0 = Target_data[Target_data['response'] == 0].iloc[:, 2:].values
# x2_0label = torch.cat((torch.zeros((x2_0.shape[0],1)),torch.ones((x2_0.shape[0],1))),dim=1)
# x2_1 = Target_data[Target_data['response'] == 1].iloc[:, 2:].values
# x2_1label = torch.cat((torch.zeros((x2_1.shape[0],1)),torch.ones((x2_1.shape[0],1))),dim=1)
# x3_0 = source3_data[source3_data['response'] == 0].iloc[:, 2:].values
# x3_0label = torch.cat((torch.zeros((x3_0.shape[0],1)),torch.zeros((x3_0.shape[0],1)),torch.ones((x3_0.shape[0],1))),dim=1)
# x3_1 = source3_data[source3_data['response'] == 1].iloc[:, 2:].values
# x3_1label = torch.cat((torch.zeros((x3_1.shape[0],1)),torch.zeros((x3_1.shape[0],1)),torch.ones((x3_1.shape[0],1))),dim=1)
result_list = []
l = [x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, t_1]
# 循环从 x1 到 x11
for i in range(16):
    # 初始化一个全零张量
    zeros_tensor = torch.zeros((l[i].shape[0], 17))
    # 在第 i 列插入全一张量
    zeros_tensor[:, i] = 1
    # 将生成的张量加入到结果列表中
    l[i] = zeros_tensor
zeros_tensor = torch.zeros((l[16].shape[0], 17))
zeros_tensor[:, 16] = 1
l[16] = zeros_tensor
# print(x['x3_label'])
# 将列表中的张量连接起来，沿着列的方向
# print(x['x1_label'])




s1 = np.concatenate((x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16), axis=0)

l1 = torch.cat((l[0], l[1], l[2], l[3], l[4], l[5], l[6], l[7], l[8], l[9], l[10], l[11], l[12], l[13], l[14], l[15]), dim=0)
s2 = np.concatenate((t_1,), axis=0)
l2 = torch.cat((l[16],), dim=0)
# ic = np.concatenate((ic_1, ic_2, ic_3, ic_4, ic_5, ic_6, ic_7, ic_8, ic_9, ic_10, ic_11, ic_12, ic_13, ic_14, ic_15,), axis=0)
# t1 = np.concatenate((x2_0,), axis=0)
# t2 = np.concatenate((x2_1,), axis=0)
# tl1 = torch.cat((x2_0label,), dim=0)
# tl2 = torch.cat((x2_1label,), dim=0)
# pca = PCA(n_components=800)  # 选择要降到的目标维度
# pca_s1 = pca.fit_transform(s1)
# pca_s2 = pca.fit_transform(s2)
bs = 1024
# y1_0 = source_data[source_data['response'] == 0].iloc[:, 1:2].values
# y1_1 = source_data[source_data['response'] == 1].iloc[:, 1:2].values
# y2_0 = Target_data[Target_data['response'] == 0].iloc[:, 1:2].values
# y2_1 = Target_data[Target_data['response'] == 1].iloc[:, 1:2].values
# y3_0 = source3_data[source3_data['response'] == 0].iloc[:, 1:2].values
# y3_1 = source3_data[source3_data['response'] == 1].iloc[:, 1:2].values
r1 = np.concatenate((r_1, r_2, r_3, r_4, r_5, r_6, r_7, r_8, r_9, r_10, r_11, r_12, r_13, r_14, r_15, r_16), axis=0)
r2 = np.concatenate((t_r,), axis=0)
# r16 = np.concatenate((r_16,), axis=0)
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
# s3 = torch.Tensor(x_16)
# r3 = torch.Tensor(r16)
ic_1 = torch.Tensor(ic_16)
# ic_16 = torch.Tensor(ic_16)
# s1_train,l1_train,r0_train = train_test_split(s1,l1,r0,random_state=66)
# s2_train,l2_train,r1_train = train_test_split(s2,l2,r1,random_state=66)
s_train = MyDataset(features=s1, domain_labels=l1, class_labels=r1)
s_loader = DataLoader(s_train, batch_size=bs, shuffle=False, drop_last=True)

s_test = MyDataset(features=s2, domain_labels=l2, class_labels=r2)
s_test = DataLoader(s_test, batch_size=bs, shuffle=False)
# s_test2 = TensorDataset(s3, l[15], r3, ic_16)
# s_test2 = DataLoader(s_test2, batch_size=bs, shuffle=True)
# s2_train = TensorDataset(s2, l2, r1)
# s2_loader = DataLoader(s2_train, batch_size=bs, shuffle=True, drop_last=True)
# s2_test = TensorDataset(t2, tl2, tr1)
# s2_test = DataLoader(s2_test,batch_size=bs,shuffle=True)


# # 使用RandomUnderSampler进行过采样
# ros = RandomOverSampler(random_state=42)
# source losses = []
# # tlosses_x_resampled, source_y_resampled = ros.fit_resample(source_x, source_y)

# CEloss = nn.BCEWithLogitsLoss()
CEloss = nn.BCEWithLogitsLoss()
adversarial_loss = nn.CrossEntropyLoss()
Trip = HardTripletLoss()
G = FeatureExtractor(s1.shape[1])
F = Classifier()
D = Discriminator()
G.cuda()
F.cuda()
D.cuda()
# model = Model(s1.shape[1]).to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-5)
optimizer=torch.optim.Adagrad(
    itertools.chain(G.parameters(), F.parameters(), D.parameters()), lr=1e-5, weight_decay=1e-5)
# discriminator_optimizer = torch.optim.Adam(Model.parameters(), lr=0.0002, betas=(0.5, 0.999))


losses = []
tlosses = []
auc_scores = []
tauc_scores = []
epochs = 600
for epo in range(epochs):

    pred_ = torch.Tensor()
    label_ = torch.Tensor()
    for features, domain_labels, class_labels in s_loader:
        G.train()
        F.train()
        D.train()
        # 特征提取
        features = features.cuda()
        feature_extracted = G(features)

        # 分类器训练
        class_logits = F(feature_extracted)
        class_logits=class_logits.to('cpu')
        class_labels_view = class_labels.view(-1,1)
        class_loss = CEloss(class_logits, class_labels_view)

        pred_ = torch.cat((pred_,class_logits),0)
        label_ = torch.cat((label_,class_labels_view),0)
        # 域鉴别器训练

        domain_logits = D(feature_extracted)
        domain_logits = domain_logits.to('cpu')
        domain_loss = adversarial_loss(domain_logits,domain_labels)
        # 三元组损失
        domain_class_label = torch.zeros(feature_extracted.size(0),1)
        domain_idx = np.argmax(domain_labels, axis=1) + 1
        for i,label in enumerate(class_labels):
            if label==1:
                domain_class_label[i]=0
            else:
                domain_class_label[i]=domain_idx[i]
        domain_class_label=domain_class_label.cuda()
        trip_loss = Trip(feature_extracted,domain_class_label)
        total_loss = class_loss+domain_loss+trip_loss
        total_loss.backward(retain_graph=True)
        optimizer.step()
        optimizer.zero_grad()
    loss_ = total_loss.item()
    #     print(label_)
    # print(pred_)
    # auc = roc_auc_score(label_, pred_)
    # auc_scores.append(auc)
    losses.append(loss_)
    with torch.no_grad():
        G.eval()
        F.eval()
        D.eval()
        tlabel_ = torch.Tensor()
        tpred_ = torch.Tensor()
        total_feature = torch.Tensor()
        total_ic = torch.Tensor()
        total_labels = torch.Tensor()
        for features, domain_labels, class_labels in s_test:
            # 特征提取
            features = features.cuda()
            feature_extracted = G(features)

            # 分类器训练
            class_logits = F(feature_extracted)
            class_logits = class_logits.to('cpu')
            class_labels_view = class_labels.view(-1, 1)
            class_loss = CEloss(class_logits, class_labels_view)
            tpred_ = torch.cat((tpred_, class_logits), 0)
            tlabel_ = torch.cat((tlabel_, class_labels_view), 0)
            feature_extracted = feature_extracted.to('cpu')
            total_feature = torch.cat((total_feature, feature_extracted), dim=0)

        # print(tlabel_)
        # print(tpred_)
        tlabel_ = tlabel_.cpu().numpy()
        tpred_ = tpred_.cpu().numpy()
        tauc = roc_auc_score(tlabel_, tpred_)
        print(tauc)
        tauc_scores.append(tauc)
total_labels = total_labels.cpu().numpy()
total_feature = total_feature.cpu().numpy()
# total_ic = total_ic.cpu().numpy()
tpr, fpr, _ = roc_curve(tlabel_, tpred_)
# pd.DataFrame(tlabel_).to_csv('PLX4720_tlable_x16.csv')
# pd.DataFrame(tpred_).to_csv('PLX470_tprde_x16.csv')
# pd.DataFrame(total_feature).to_csv('PLX4720_feature_x16.csv')
# pd.DataFrame(total_ic).to_csv('PLX4720_ic_x16.csv')
# print(tlabel_)
# print(tpred_)
# plt.figure()
# lw = 2
# plt.plot(tpr, fpr, color='darkorange', lw=lw, label='ROC curve')
# plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic (ROC) Curve')
# plt.legend(loc='lower right')
# plt.show()
# x = range(1, 601)
# # 生成auc曲线
# plt.plot(x, auc_scores, marker='.', linestyle='-', color='b')
# plt.plot(x, tauc_scores, marker='.', linestyle='-', color='r')
# plt.xlim(0, max(x))
# plt.ylim(0, 1)
# # 添加标题和轴标签
# plt.title('AUC Scores over Epochs')
# plt.xlabel('Epoch')
# plt.ylabel('AUC Score')
# #
# # # 显示网格
# plt.grid(True)
# #
# # # 显示图表
# plt.show()
#
# # 生成loss曲线
# plt.plot(x, losses, marker='.', linestyle='-', color='b')
#
# # 添加标题和轴标签
# plt.title('losses over Epochs')
# plt.xlabel('Epoch')
# plt.ylabel('loss')
#
# # 显示网格
# plt.grid(True)
#
# # 显示图表
# plt.show()

total_feature = torch.Tensor().to(device)
total_response = torch.Tensor().to(device)
total_labels = torch.Tensor()
total_c2 = torch.Tensor().to(device)
total_ic = torch.Tensor().to(device)
# with torch.no_grad():
#     for i, data in enumerate(s_test2):
#         s1_ = data[0].to(device)
#         l1_ = data[1].to(device)
#         r1_ = data[2].to(device)
#         ic_t = data[3].to(device)
#         # ic = MinMaxScaler(feature_range=(-1, 1))
#         c2, d3, trip, feature = model(s1_, r1_)
#         total_feature = torch.cat((total_feature, feature), dim=0)
#         total_response = torch.cat((total_response, r1_), dim=0)
#         total_c2 = torch.cat((total_c2, c2), dim=0)
#         total_ic = torch.cat((total_ic, ic_t), dim=0)
#
#         l1_ = l1_.to('cpu')
#         labels = np.argmax(l1_, axis=1)
#         total_labels = torch.cat((total_labels, labels), dim=0)
# total_feature = torch.Tensor().to(device)
# total_response = torch.Tensor().to(device)
# total_labels = torch.Tensor()
with torch.no_grad():
    for features, domain_labels, class_labels in s_loader:
        # 特征提取
        features = torch.Tensor.to(device)
        feature_extracted = G(features)

        # 分类器训练
        class_logits = F(feature_extracted)
        class_loss = CEloss(class_logits, class_labels)
        # pred_ = torch.cat((pred_, class_logits), 0)
        # label_ = torch.cat((label_, class_labels), 0)
        total_feature = torch.cat((total_feature, feature_extracted), dim=0)
        total_response = torch.cat((total_response, class_labels), dim=0)
        l1_ = l1_.to('cpu')
        labels = np.argmax(l1_, axis=1)
        total_labels = torch.cat((total_labels, labels), dim=0)
umap_after = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='correlation')
total_feature = total_feature.cpu()
total_response = total_response.cpu()

X1_2d = umap_after.fit_transform(total_feature)
print(total_labels)

print(X1_2d[total_response == 0])
print(total_feature.size())
print(total_response.size())
print(total_labels.size())
# X2_2d = tsne.fit_transform(single_feature_l)
# X3_2d = umap_after.fit_transform(total_single)
colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10']  # 用于三个数据集的颜色
markers = ['x', '.']  # 圆形和方形标记
# labels = ['bulk', 'single_unlabel']
plt.figure(figsize=(25, 25))
for i, color in enumerate(zip(colors)):
    # 根据标签绘制不同的点

    for j, marker in enumerate(markers):
        subset = X1_2d[(total_response == j) & (total_labels == i)]  # 这里假设y1, y2, y3都是数值型标签，0对应不敏感，1对应敏感
        plt.scatter(subset[:, 0], subset[:, 1], c=color, marker=marker, s=100)

plt.legend()
plt.title('t-SNE Visualization of Three Datasets')
plt.savefig('PLX4720_exp_ex16.jpg', dpi=500)
total_labels = total_labels.cpu().numpy()
total_feature = total_feature.numpy()
total_response = total_response.numpy()
# pd.DataFrame(total_labels).to_csv('PLX4720_total_labels_ex16.csv')
# pd.DataFrame(total_feature).to_csv('PLX4720_total_feature_ex16.csv')
# pd.DataFrame(total_response).to_csv('PLX4720_total_response_ex16.csv')
# # model_linear = LinearRegression()
# total_ic = total_ic.cpu()
# total_c2 = total_c2.cpu()
# total_ic = total_ic.unsqueeze(1)
# # total_ic = total_ic.exp()
# total_c2 = total_c2.numpy()
# total_ic = total_ic.numpy()
# pd.DataFrame(total_c2).to_csv('c2_ic.csv')
# pd.DataFrame(total_ic).to_csv('c2_ic.csv', mode='a', header=False)
# total_c2 = 3 + (total_c2 - np.min(total_c2)) * (4 - 3) / (np.max(total_c2) - np.min(total_c2))
# plt.scatter(total_ic, total_c2)
# plt.xlabel('c2')
# plt.ylabel('ic')
# plt.title('Scatter plot of ic and c2')
# plt.show()

# total_c2 = total_c2.reshape(-1)
# total_ic = total_c2.reshape(-1)
# scaler_ic = MinMaxScaler(feature_range=(0, 6))
# scaler_c2 = MinMaxScaler(feature_range=(0, 6))
# ic_scaled = scaler_ic.fit_transform(total_ic)
# c2_scaled = scaler_c2.fit_transform(total_c2)
# total_ic = torch.sigmoid(total_ic)
# total_c2 = torch.exp(total_c2)
# print(total_c2)
# model_linear.fit(total_ic, total_c2)
# 计算残差
# predict = model_linear.predict(total_c2)
# residuals = total_ic - predict
# threshold = 2 * np.std(residuals)
# outlier_indices = np.where(np.abs(residuals) > threshold)[0]
# filtered_ic = np.delete(total_ic, outlier_indices)
# filtered_c2 = np.delete(total_c2, outlier_indices)
# min_ic = np.min(filtered_ic)
# max_ic = np.max(filtered_ic)
# min_c2 = np.min(filtered_c2)
# max_c2 = np.max(filtered_c2)
#
# # 最小-最大缩放到0到1的区间
# filtered_ic = (filtered_ic - min_ic) / (max_ic - min_ic)
# filtered_c2 = (filtered_c2 - min_c2) / (max_c2 - min_c2)
# filtered_ic = filtered_ic.reshape(-1, 1)
# filtered_c2 = filtered_c2.reshape(-1, 1)
# model_linear.fit(total_c2, total_ic)
# confidence = model_linear.score(total_ic, total_c2)
# print("R^2:", confidence)
# plt.scatter(total_c2, total_ic, label='Data Points')
# plt.plot(total_c2, model_linear.predict(total_ic), color='red', label='Linear Regression')
# plt.xlabel('c2')
# plt.ylabel('ic')
# plt.title('Linear Regression')
# plt.legend()
# plt.show()
#
# umap_after = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='correlation')
# total_feature = total_feature.cpu()
# total_response = total_response.cpu()
# X1_2d = umap_after.fit_transform(total_feature)
# print(total_labels)
#
# print(X1_2d[total_response==0])
# print(total_feature.size())
# print(total_response.size())
# print(total_labels.size())
# #X2_2d = tsne.fit_transform(single_feature_l)
# # X3_2d = umap_after.fit_transform(total_single)
# colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9'] # 用于三个数据集的颜色
# markers = ['x', '.'] # 圆形和方形标记
# # labels = ['bulk', 'single_unlabel']
# plt.figure(figsize=(25,25))
# for i,color in enumerate(zip(colors)) :
# # 根据标签绘制不同的点
#
#     for j, marker in enumerate(markers):
#         subset = X1_2d[(total_response == j)&(total_labels == i)] # 这里假设y1, y2, y3都是数值型标签，0对应不敏感，1对应敏感
#         plt.scatter(subset[:, 0], subset[:, 1], c=color,  marker=marker, s=20)
#
# plt.legend()
# plt.title('t-SNE Visualization of Three Datasets')
# plt.savefig('new_exp.jpg',dpi = 500)
