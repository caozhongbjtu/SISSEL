import argparse
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import warnings
import logging
import os
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import datetime

device = 'cuda'
os.environ['PYTORCH_ENABLE_TENSOR_CORE'] = '1'


def col_Multivariate_rolling_window(all_df, start, end, lookback, stride):
    all_res = []
    lens_data = all_df.shape[1]
    for i in range(start, lens_data - end + 1, stride):
        res = []
        for series in all_df:
            res.append(series[i:(i + lookback)])
        all_res.append(res)
    return np.array(all_res)


def load_forecast_csv(name, lookback_, pred_lens):
    data = pd.read_csv(name, index_col='date', parse_dates=True)
    print("len_data:", len(data))

    data = data.to_numpy()
    if 'ETTh1' in name or 'ETTh2' in name:
        train_slice = slice(None, 8545)
        test_slice = slice(16 * 30 * 24 - (lookback_ + pred_lens), 20 * 30 * 24)
        n_slice = 8545
        s_slice = 2881

    elif 'ETTm1' in name or 'ETTm2' in name:
        train_slice = slice(None, 34465)
        test_slice = slice(16 * 30 * 24 * 4, 20 * 30 * 24 * 4)
        n_slice = 34465
        s_slice = 11521

    elif 'weather' in name:
        train_slice = slice(None, 36792)
        test_slice = slice(16 * 30 * 24 * 4, 20 * 30 * 24 * 4)
        n_slice = 36792
        s_slice = 10540

    elif 'traffic' in name:
        train_slice = slice(None, 12185)
        test_slice = slice(16 * 30 * 24 * 4, 20 * 30 * 24 * 4)
        n_slice = 12185
        s_slice = 3509

    elif 'electricity' in name:
        train_slice = slice(None, 18317)
        test_slice = slice(16 * 30 * 24 * 4, 20 * 30 * 24 * 4)
        n_slice = 18317
        s_slice = 5261

    elif 'exchange_rate' in name:
        train_slice = slice(None, 5120)
        test_slice = slice(16 * 30 * 24 * 4, 20 * 30 * 24 * 4)
        n_slice = 5120
        s_slice = 1422

    scaler = StandardScaler().fit(data[train_slice])
    data = scaler.transform(data)
    return data, n_slice, s_slice, scaler


class autoencoder(nn.Module):
    def __init__(self, input_size):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, input_size)
        )

    def forward(self, x):
        _x = self.encoder(x)
        x = self.decoder(_x)
        return x, _x


def mask_me(data, ratio, lens):
    data = data.to(device)
    mask = np.zeros(int(lens * ratio)).tolist() + np.ones(lens - int(lens * ratio)).tolist()
    random.shuffle(mask)
    mask = torch.tensor(mask).bool().to(device)
    out = torch.masked_fill(input=data, mask=~mask, value=0).to(device)
    return out


class Lineras(nn.Module):
    def __init__(self, input_dim, xxx):
        super(Lineras, self).__init__()
        self.fc2 = nn.Linear(input_dim, 512)
        self.fc3 = nn.Linear(512, xxx)

    def forward(self, x):
        x = nn.ReLU()(self.fc2(x))
        x = self.fc3(x)
        return x


def vail(test_loader, net, net1, net2, net_mask, forecast_model,
          criterion_forecast, criterion_MAE,
          lookback, pred_lens, stride, avg_pool_2, avg_pool_4):

    net.eval(); net1.eval(); net2.eval(); net_mask.eval(); forecast_model.eval()

    iter_num = 0
    res_MAE = 0
    res_MSE = 0

    with torch.no_grad():
        for batch in test_loader:
            batch_data = col_Multivariate_rolling_window(
                np.array(batch[0]).T,
                0, lookback + pred_lens,
                lookback + pred_lens,
                stride
            )
            x = torch.tensor(batch_data[..., :lookback]).to(device)
            label = torch.tensor(batch_data[..., lookback:]).to(device)

            var_x = x.reshape(x.shape[0], x.shape[1], 1, -1)
            var_x1 = x.reshape(x.shape[0], x.shape[1], 2, -1)
            var_x2 = x.reshape(x.shape[0], x.shape[1], 4, -1)

            _, out = net(var_x)
            _, out1 = net1(var_x1)
            _, out2 = net2(var_x2)

            sub_len = out.shape[0]
            repre = (out.reshape(sub_len, out.shape[1], -1)
                     + avg_pool_2(out1.reshape(sub_len, out1.shape[1], -1))
                     + avg_pool_4(out2.reshape(sub_len, out2.shape[1], -1))) / 3

            out_forecast = forecast_model(repre)

            MSE_loss = criterion_forecast(out_forecast, label)
            MAE_loss = criterion_MAE(out_forecast, label)

            iter_num += 1
            res_MAE += MAE_loss
            res_MSE += MSE_loss

    res_MAE /= iter_num
    res_MSE /= iter_num


    net.train(); net1.train(); net2.train(); net_mask.train(); forecast_model.train()

    return res_MSE.item(), res_MAE.item(), out_forecast, label



# ----------------------------------------------------------
# ----------------------- MAIN 函数 -------------------------
# ----------------------------------------------------------
def main(args):

    # ---------------- Load Data ----------------
    if 'ETT' in args.dataset:
        dataset_path = 'all_datasets/long_term_forecast/ETT-small/'
        file = dataset_path + args.dataset + '.csv'
    else:
        dataset_path = 'all_datasets/long_term_forecast/' +args.dataset+ '/'
        file = dataset_path + args.dataset + '.csv'
    data, n_slice, s_slice, scaler = load_forecast_csv(
        file, args.lookback, args.pred_lens
    )

    # Train & test slicing
    train_data = [data[:n_slice]]
    test_data = [data[-(s_slice + args.lookback):]]

    # 选择 batch_size
    for x in range(args.batch_sizes, 0, -1):
        if test_data[0].shape[0] % x > args.lookback + args.pred_lens and train_data[0].shape[0] % x > args.lookback + args.pred_lens:
            batch_size = x
            print("batch_size:", x)
            break

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_data[0]).float()),
        batch_size=batch_size, shuffle=False
    )

    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(test_data[0]).float()),
        batch_size=batch_size, shuffle=False
    )

    # ---------------- Setup Models ----------------
    net = autoencoder(args.lookback).to(device)
    net1 = autoencoder(args.lookback // 2).to(device)
    net2 = autoencoder(args.lookback // 4).to(device)
    net_mask = autoencoder(args.lookback).to(device)

    net = torch.compile(net)
    net1 = torch.compile(net1)
    net2 = torch.compile(net2)
    net_mask = torch.compile(net_mask)

    criterion = nn.MSELoss().to(device)
    criterion1 = nn.MSELoss().to(device)
    criterion2 = nn.MSELoss().to(device)
    criterion3 = nn.MSELoss().to(device)

    optimizer = torch.optim.Adam([
        {'params': net.parameters(), 'lr': args.AE_lr_rate},
        {'params': net1.parameters(), 'lr': args.AE_lr_rate},
        {'params': net2.parameters(), 'lr': args.AE_lr_rate},
        {'params': net_mask.parameters(), 'lr': args.AE_lr_rate},
    ])

    # ---------------- AE Training ----------------
    net_loss, net_loss1, net_loss2, net_loss3, net_loss4 = [], [], [], [], []

    for e in range(args.epoch):
        iter_num = 0
        loss_epoch = 0

        for batch in train_loader:

            batch_data = col_Multivariate_rolling_window(
                np.array(batch[0]).T,
                0, args.lookback + args.pred_lens,
                args.lookback + args.pred_lens,
                args.stride
            )

            x = torch.tensor(batch_data[..., :args.lookback]).to(device)
            train_x_mask = mask_me(x, 0.15, args.lookback)

            var_x = x.reshape(x.shape[0], x.shape[1], 1, -1)
            var_x1 = x.reshape(x.shape[0], x.shape[1], 2, -1)
            var_x2 = x.reshape(x.shape[0], x.shape[1], 4, -1)
            var_x_mask = train_x_mask.reshape(x.shape[0], x.shape[1], 1, -1)

            out, _ = net(var_x)
            out1, _ = net1(var_x1)
            out2, _ = net2(var_x2)
            out3, _ = net_mask(var_x_mask)

            loss1 = criterion(out, var_x)
            loss2 = criterion1(out1, var_x1)
            loss3 = criterion2(out2, var_x2)
            loss4 = criterion3(out3, var_x)

            sum_loss = loss1.item() + loss2.item() + loss3.item() + loss4.item()

            if e == 0 and iter_num == 0:
                loss = 0.25 * (loss1 + loss2 + loss3 + loss4)
                loss_ = np.array([loss.item(), loss1.item(), loss2.item(), loss3.item(), loss4.item()])
                loss_epoch = loss_
            else:
                loss = (loss1 * loss_[1] + loss2 * loss_[2] + loss3 * loss_[3] + loss4 * loss_[4]) / sum_loss
                loss_ = np.array([loss.item(), loss1.item(), loss2.item(), loss3.item(), loss4.item()])
                loss_epoch += loss_

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num += 1

        loss_epoch = loss_epoch / iter_num
        net_loss.append(loss_epoch[0])
        net_loss1.append(loss_epoch[1])
        net_loss2.append(loss_epoch[2])
        net_loss3.append(loss_epoch[3])
        net_loss4.append(loss_epoch[4])

        print(f'Epoch: {e+1}, batch: {iter_num}, Loss: {loss_epoch[0]:.4f}, '
              f'Loss1: {loss_epoch[1]:.4f}, Loss2: {loss_epoch[2]:.4f}, '
              f'Loss3: {loss_epoch[3]:.4f}, Loss4: {loss_epoch[4]:.4f}')

    # -------- Save AE loss curves --------
    os.makedirs("loss_result", exist_ok=True)
    plt.suptitle("mse")
    plt.plot(net_loss1, label="$AE_0$")
    plt.plot(net_loss2, label="$AE_1$")
    plt.plot(net_loss3, label="$AE_2$")
    plt.plot(net_loss4, label="$AE_{mask}$")
    plt.legend()
    plt.savefig('loss_result/' + str(datetime.datetime.now())[5:19] + 'training.pdf')
    np.save('loss_result/' + str(datetime.datetime.now())[5:19] + "net_loss_.npy",
           [net_loss, net_loss1, net_loss2, net_loss3, net_loss4])

    # ---------------- forecast ----------------
    forecast_model = Lineras(64, args.pred_lens).to(device)
    criterion_forecast = nn.MSELoss().to(device)
    criterion_MAE = nn.L1Loss().to(device)
    optimizer = torch.optim.Adam(forecast_model.parameters(), lr=args.forecast_lr_rate)
    forecast_model = torch.compile(forecast_model)

    avg_pool_2 = nn.AvgPool1d(kernel_size=2, stride=2)
    avg_pool_4 = nn.AvgPool1d(kernel_size=4, stride=4)

    save_path = dataset_path + args.dataset[:-4] + "_" + str(args.pred_lens) + "_" + str(datetime.datetime.now())[5:19]
    os.makedirs(save_path, exist_ok=True)

    f = open(save_path + "/config.txt", 'a')
    f.write("batch_size:{}, epoch:{}, clf_epoch:{}, stride:{}, lookback:{}, pred_lens:{}\n".format(
        batch_size, args.epoch, args.clf_epoch, args.stride, args.lookback, args.pred_lens
    ))
    f.close()

    last_res = 2
    last_mae = 0

    # ----------- Train forecast --------
    for e in range(args.clf_epoch):
        iter_num = 0
        holder = 0

        for batch in train_loader:
            batch_data = col_Multivariate_rolling_window(
                np.array(batch[0]).T,
                0, args.lookback + args.pred_lens,
                args.lookback + args.pred_lens,
                args.stride
            )

            x = torch.tensor(batch_data[..., :args.lookback]).to(device)
            label = torch.tensor(batch_data[..., args.lookback:]).to(device)

            var_x = x.reshape(x.shape[0], x.shape[1], 1, -1)
            var_x1 = x.reshape(x.shape[0], x.shape[1], 2, -1)
            var_x2 = x.reshape(x.shape[0], x.shape[1], 4, -1)

            _, out = net(var_x)
            _, out1 = net1(var_x1)
            _, out2 = net2(var_x2)

            sub_len = out.shape[0]
            repre = (out.reshape(sub_len, out.shape[1], -1)
                     + avg_pool_2(out1.reshape(sub_len, out1.shape[1], -1))
                     + avg_pool_4(out2.reshape(sub_len, out2.shape[1], -1))) / 3

            out_forecast = forecast_model(repre)

            loss_forecast = criterion_forecast(out_forecast, label)

            optimizer.zero_grad()
            loss_forecast.backward()
            optimizer.step()

            iter_num += 1
            holder += loss_forecast.item()

        print(f'Epoch: {e+1}, batch:{iter_num}, loss_clf:{holder/iter_num:.10f}')

        res_MSE, res_MAE, _, _ = vail(
            test_loader, net, net1, net2, net_mask,
            forecast_model, criterion_forecast, criterion_MAE,
            args.lookback, args.pred_lens, args.stride,
            avg_pool_2, avg_pool_4
        )

        if res_MSE < last_res:
            last_res = res_MSE
            last_mae = res_MAE
            os.makedirs(save_path + '/model', exist_ok=True)
            torch.save(net.state_dict(), save_path + '/model/net_.pkl')
            torch.save(net1.state_dict(), save_path + '/model/net1_.pkl')
            torch.save(net2.state_dict(), save_path + '/model/net2_.pkl')
            torch.save(net_mask.state_dict(), save_path + '/model/net_mask_.pkl')
            torch.save(forecast_model.state_dict(), save_path + '/model/clf.pkl')

    print("test_MSE:", last_res)
    print("test_MAE:", last_mae)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_sizes', type=int, default=2000)
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--clf_epoch', type=int, default=1)
    parser.add_argument('--lookback', type=int, default=200)
    parser.add_argument('--pred_lens', type=int, default=96)
    parser.add_argument('--dataset', type=str, default='ETTm2')
    # parser.add_argument('--dataset_path', type=str, default='all_datasets/long_term_forecast/ETT-small/')
    parser.add_argument('--stride', type=int, default=1)

    parser.add_argument('--forecast_lr_rate', type=float, default=0.001)
    parser.add_argument('--AE_lr_rate', type=float, default=0.001)

    args = parser.parse_args()

    print("Args:", args)

    main(args)
