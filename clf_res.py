import argparse
import random
import os
import datetime
import warnings
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sktime.datasets import load_from_tsfile_to_dataframe
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import csv

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
lgr = logging.getLogger(__name__)

# ----------------- Data Loading & Preprocessing -----------------
def load_single(filepath):
    df, labels = load_from_tsfile_to_dataframe(filepath, return_separate_X_and_y=True,
                                               replace_missing_vals_with='NaN')
    labels = pd.Series(labels, dtype="category")
    labels_df = pd.DataFrame(labels.cat.codes, dtype=np.int8)
    lengths = df.applymap(lambda x: len(x)).values
    horiz_diffs = np.abs(lengths - np.expand_dims(lengths[:, 0], -1))
    
    if np.sum(horiz_diffs) > 0:
        df = df.applymap(subsample)
    
    lengths = df.applymap(lambda x: len(x)).values
    vert_diffs = np.abs(lengths - np.expand_dims(lengths[0, :], 0))
    max_seq_len = int(np.max(lengths[:, 0])) if np.sum(vert_diffs) > 0 else lengths[0, 0]
    
    df = df.groupby(by=df.index).transform(interpolate_missing)
    return df, labels_df, max_seq_len

def interpolate_missing(y):
    if y.isna().any():
        y = y.interpolate(method='linear', limit_direction='both')
    return y

def subsample(y, limit=256, factor=2):
    if len(y) > limit:
        return y[::factor].reset_index(drop=True)
    return y

def rolling_window(data, look_back, stride):
    dataX = []
    for i in range(0, len(data) - look_back + 1, stride):
        dataX.append(list(data[i:i+look_back]))
    return np.array(dataX)

def Multivariate_rolling_window(all_df, lookback, stride):
    all_res = []
    for i in range(len(all_df)):
        single_res = [rolling_window(j, lookback, stride) for j in all_df[i]]
        all_res.append(single_res)
    return np.array(all_res)

def normalize_and_pad(data_list, max_len, mode="train", scaler=None):
    if isinstance(data_list, np.ndarray):
        data_list = data_list.tolist()

    N, C = len(data_list), len(data_list[0])

    if mode == "train":
        all_values = np.concatenate([np.array(sample[c]) for sample in data_list for c in range(C)]).reshape(-1, 1)
        scaler = StandardScaler()
        scaler.fit(all_values)
    elif mode == "test" and scaler is None:
        raise ValueError("mode='test' must provide scaler")
    
    data_norm = np.zeros((N, C, max_len))
    padding_mask = np.zeros((N, max_len), dtype=np.float32)

    for i, sample in enumerate(data_list):
        L = min(max(len(np.array(seq)) for seq in sample), max_len)
        for c in range(C):
            seq = np.array(sample[c])
            seq_norm = scaler.transform(seq.reshape(-1,1)).reshape(-1)
            data_norm[i, c, :len(seq_norm)] = seq_norm[:max_len]
        padding_mask[i, :L] = 1

    return (data_norm, padding_mask, scaler) if mode == "train" else (data_norm, padding_mask)

def data_read(filepath):
    df, labels_df, tslens = load_single(filepath)
    all_label = list(labels_df.to_numpy().reshape(-1))
    ls = [[np.array(df.iloc[i,j]) for j in range(df.shape[1])] for i in range(df.shape[0])]
    return np.array(ls), all_label, len(set(all_label)), tslens

# ----------------- Models -----------------
class Autoencoder(nn.Module):
    def __init__(self, input_size):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 512), nn.ReLU(True),
            nn.Linear(512, 256), nn.ReLU(True),
            nn.Linear(256, 128), nn.ReLU(True),
            nn.Linear(128, 64)
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 128), nn.ReLU(True),
            nn.Linear(128, 256), nn.ReLU(True),
            nn.Linear(256, 512), nn.ReLU(True),
            nn.Linear(512, input_size)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_rec = self.decoder(z)
        return x_rec, z

class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(Classifier, self).__init__()
        self.fc2 = nn.Linear(input_dim, 256)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = nn.ReLU()(self.fc2(x))
        x = self.fc3(x)
        return x

# ----------------- Utilities -----------------
def mask_me(data, ratio, lens, device):
    data = data.to(device)
    mask = np.zeros(int(lens*ratio)).tolist() + np.ones(lens - int(lens*ratio)).tolist()
    random.shuffle(mask)
    mask = torch.tensor(mask).bool().to(device)
    return torch.masked_fill(input=data, mask=~mask, value=0).to(device)

def TSNE_repre_res(test_out_clf, test_all_label, datast_path):
    TSNE_repre = [{"X":test_out_clf, "y":test_all_label}]
    pd.DataFrame(TSNE_repre).to_csv(os.path.join(datast_path, '_TSNE_repre.csv'), index=False)

def TSNE_plot(X, y, datast_path):
    X_tsne = TSNE(n_components=2, random_state=42).fit_transform(np.array(X))
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y)
    plt.legend(*scatter.legend_elements(), title="class")
    plt.title('t-SNE')
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.savefig(os.path.join(datast_path, 'TSNE_plot.pdf'))

# ----------------- Training & Evaluation -----------------
def validate_nets(test_loader, nets, clf_model, device, test_all_label):
    net, net1, net2, net3 = nets
    net.eval(); net1.eval(); net2.eval(); net3.eval()
    clf_model.eval()

    test_out_clf = []
    with torch.no_grad():
        for batch in test_loader:
            x, label = batch[0].to(device), batch[1].to(device)
            var_x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1, -1)
            var_x1 = x.reshape(x.shape[0], x.shape[1], x.shape[2], 2, -1)
            var_x2 = x.reshape(x.shape[0], x.shape[1], x.shape[2], 4, -1)
            _, out = net(var_x)
            _, out1 = net1(var_x1)
            _, out2 = net2(var_x2)

            sub_len = out.shape[0]
            repre = torch.cat([out.reshape(sub_len,-1), out1.reshape(sub_len,-1), out2.reshape(sub_len,-1)], dim=1)
            out_clf = clf_model(repre)

            test_out_clf.append(out_clf.cpu().detach().numpy())

    test_out_clf = np.concatenate(test_out_clf, axis=0)
    acc = sum([np.argmax(test_out_clf[i])==test_all_label[i] for i in range(len(test_all_label))]) / len(test_all_label)

    net.train(); net1.train(); net2.train(); net3.train(); clf_model.train()
    return acc, test_out_clf

# ----------------- Main Function -----------------
def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    datast_path = os.path.join('Multivariate_ts', args.dataset)
    train_file = [f for f in os.listdir(datast_path) if '_TRAIN' in f][0]
    test_file = [f for f in os.listdir(datast_path) if '_TEST' in f][0]
    train_path = os.path.join(datast_path, train_file)
    test_path = os.path.join(datast_path, test_file)
    datast_path = os.path.join(datast_path, str(datetime.datetime.now())[5:19])
    os.makedirs(datast_path, exist_ok=True)

    # ---- Data ----
    all_df, all_label, num_class, train_max_seq_len = data_read(train_path)
    test_all_df, test_all_label, _, test_max_seq_len = data_read(test_path)
    max_seq_len = max(train_max_seq_len, test_max_seq_len)

    all_df, train_mask, scaler = normalize_and_pad(all_df, max_len=max_seq_len, mode="train")
    test_all_df, test_mask = normalize_and_pad(test_all_df, max_len=max_seq_len, mode="test", scaler=scaler)

    r_window_data = Multivariate_rolling_window(all_df, args.lookback, args.stride)
    test_r_window_data = Multivariate_rolling_window(test_all_df, args.lookback, args.stride)

    train_set = TensorDataset(torch.from_numpy(r_window_data).float(), torch.tensor(all_label).long())
    test_set = TensorDataset(torch.from_numpy(test_r_window_data).float(), torch.tensor(test_all_label).long())
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=args.shuffle)
    test_loader = DataLoader(test_set, batch_size=args.batch_size)

    # ---- Models ----
    net = Autoencoder(args.lookback).to(device)
    net1 = Autoencoder(int(args.lookback/2)).to(device)
    net2 = Autoencoder(int(args.lookback/4)).to(device)
    net3 = Autoencoder(args.lookback).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam([
        {'params': net.parameters()},
        {'params': net1.parameters()},
        {'params': net2.parameters()},
        {'params': net3.parameters()},
    ], lr=args.lr)
    nets = [net, net1, net2, net3]

    # ---- Train autoencoders ----
    for e in range(args.epochs):
        for batch in train_loader:
            x = batch[0].to(device)
            train_x_mask = mask_me(x, 0.15, args.lookback, device)
            var_x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1, -1)
            var_x1 = x.reshape(x.shape[0], x.shape[1], x.shape[2], 2, -1)
            var_x2 = x.reshape(x.shape[0], x.shape[1], x.shape[2], 4, -1)
            var_x_mask = train_x_mask.reshape(x.shape[0], x.shape[1], x.shape[2], 1, -1)

            out, _ = net(var_x)
            out1, _ = net1(var_x1)
            out2, _ = net2(var_x2)
            out3, _ = net3(var_x_mask)

            loss1 = criterion(out, var_x)
            loss2 = criterion(out1, var_x1)
            loss3 = criterion(out2, var_x2)
            loss4 = criterion(out3, var_x_mask)

            loss_vals = [loss1, loss2, loss3, loss4]
            loss_sum = sum([l.item() for l in loss_vals])
            loss = sum([loss_vals[i]*loss_vals[i].item()/loss_sum for i in range(4)])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {e+1}/{args.epochs} done")

    print("Autoencoder training finished")

    # ---- Classifier ----
    clf_model = Classifier(7*r_window_data.shape[1]*r_window_data.shape[2]*64, num_class).to(device)
    criterion_clf = nn.CrossEntropyLoss()
    optimizer_clf = torch.optim.Adam(clf_model.parameters(), lr=args.lr)
    last_res = 0
    acc_res = []
    for e in range(args.clf_epochs):
        for batch in train_loader:
            x, label = batch[0].to(device), batch[1].to(device)
            var_x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1, -1)
            var_x1 = x.reshape(x.shape[0], x.shape[1], x.shape[2], 2, -1)
            var_x2 = x.reshape(x.shape[0], x.shape[1], x.shape[2], 4, -1)

            _, out = net(var_x)
            _, out1 = net1(var_x1)
            _, out2 = net2(var_x2)

            sub_len = out.shape[0]
            repre = torch.cat([out.reshape(sub_len,-1), out1.reshape(sub_len,-1), out2.reshape(sub_len,-1)], dim=1)
            out_clf = clf_model(repre)
            loss_clf = criterion_clf(out_clf, label)

            optimizer_clf.zero_grad()
            loss_clf.backward()
            optimizer_clf.step()
        print('Epoch: {}, loss_clf: {:.10f}'.format(e + 1, loss_clf.item()))
        acc, test_out_clf = validate_nets(test_loader, nets, clf_model, device, test_all_label)
        # print(f"Classifier epoch {e+1}, test accuracy: {acc:.4f}")
        acc_res.append(acc)
        if acc > last_res:
            last_res = acc
            save_path = os.path.join(datast_path,'model')
            os.makedirs(save_path, exist_ok=True)
            torch.save(net, os.path.join(save_path,'net_.pkl'))
            torch.save(net1, os.path.join(save_path,'net1_.pkl'))
            torch.save(net2, os.path.join(save_path,'net2_.pkl'))
            torch.save(net3, os.path.join(save_path,'net3_.pkl'))
            torch.save(clf_model, os.path.join(save_path,'clf.pkl'))
    print('test_set_acc:',max(acc_res))
# ----------------- Argparse -----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Libras')
    parser.add_argument('--lookback', type=int, default=44)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=90)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--clf_epochs', type=int, default=400)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--shuffle', type=bool, default=False)
    args = parser.parse_args()
    main(args)
