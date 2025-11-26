#!/bin/bash

# =======================
#   ETTm2
# =======================
python forecast_res.py --dataset ETTm2 --pred_lens 96 --batch_sizes 2000 --epoch 600 --clf_epoch 500 --stride 1 --lookback 200
python forecast_res.py --dataset ETTm2 --pred_lens 192 --batch_sizes 2000 --epoch 600 --clf_epoch 500 --stride 1 --lookback 200
python forecast_res.py --dataset ETTm2 --pred_lens 336 --batch_sizes 2000 --epoch 600 --clf_epoch 500 --stride 1 --lookback 200
python forecast_res.py --dataset ETTm2 --pred_lens 720 --batch_sizes 2000 --epoch 600 --clf_epoch 500 --stride 1 --lookback 200

# =======================
#   ETTh2
# =======================
python forecast_res.py --dataset ETTh2 --pred_lens 96 --batch_sizes 6000 --epoch 200 --clf_epoch 500 --stride 1 --lookback 128
python forecast_res.py --dataset ETTh2 --pred_lens 192 --batch_sizes 6000 --epoch 200 --clf_epoch 500 --stride 1 --lookback 128
python forecast_res.py --dataset ETTh2 --pred_lens 336 --batch_sizes 6000 --epoch 200 --clf_epoch 500 --stride 1 --lookback 128
python forecast_res.py --dataset ETTh2 --pred_lens 720 --batch_sizes 6000 --epoch 200 --clf_epoch 500 --stride 1 --lookback 128

# =======================
#   ETTm1
# =======================
python forecast_res.py --dataset ETTm1 --pred_lens 96 --batch_sizes 20000 --epoch 100 --clf_epoch 500 --stride 1 --lookback 200
python forecast_res.py --dataset ETTm1 --pred_lens 192 --batch_sizes 20000 --epoch 100 --clf_epoch 500 --stride 1 --lookback 200
python forecast_res.py --dataset ETTm1 --pred_lens 336 --batch_sizes 20000 --epoch 100 --clf_epoch 500 --stride 1 --lookback 200
python forecast_res.py --dataset ETTm1 --pred_lens 720 --batch_sizes 20000 --epoch 100 --clf_epoch 500 --stride 1 --lookback 200

# =======================
#   ETTh1           #####
# =======================
python forecast_res.py --dataset ETTh1 --pred_lens 96 --batch_sizes 6000 --epoch 200 --clf_epoch 500 --stride 1 --lookback 128
python forecast_res.py --dataset ETTh1 --pred_lens 192 --batch_sizes 6000 --epoch 200 --clf_epoch 500 --stride 1 --lookback 128
python forecast_res.py --dataset ETTh1 --pred_lens 336 --batch_sizes 6000 --epoch 200 --clf_epoch 500 --stride 1 --lookback 128
python forecast_res.py --dataset ETTh1 --pred_lens 720 --batch_sizes 6000 --epoch 200 --clf_epoch 500 --stride 1 --lookback 128

# =======================
#   Traffic
# =======================
python forecast_res.py --dataset traffic --pred_lens 96 --batch_sizes 478 --epoch 100 --clf_epoch 500 --stride 1 --lookback 128
python forecast_res.py --dataset traffic --pred_lens 192 --batch_sizes 552 --epoch 100 --clf_epoch 500 --stride 1 --lookback 128
python forecast_res.py --dataset traffic --pred_lens 336 --batch_sizes 750 --epoch 100 --clf_epoch 500 --stride 1 --lookback 128
python forecast_res.py --dataset traffic --pred_lens 720 --batch_sizes 929 --epoch 100 --clf_epoch 500 --stride 1 --lookback 128

# =======================
#   Electricity
# =======================
python forecast_res.py --dataset electricity --pred_lens 96 --batch_sizes 573 --epoch 100 --clf_epoch 500 --stride 1 --lookback 200
python forecast_res.py --dataset electricity --pred_lens 192 --batch_sizes 600 --epoch 90 --clf_epoch 500 --stride 1 --lookback 200
python forecast_res.py --dataset electricity --pred_lens 336 --batch_sizes 984 --epoch 100 --clf_epoch 500 --stride 1 --lookback 200
python forecast_res.py --dataset electricity --pred_lens 720 --batch_sizes 1135 --epoch 100 --clf_epoch 500 --stride 1 --lookback 200

# =======================
#   Exchange Rate
# =======================
python forecast_res.py --dataset exchange_rate --pred_lens 96 --batch_sizes 3100 --epoch 500 --clf_epoch 500 --stride 1 --lookback 32
python forecast_res.py --dataset exchange_rate --pred_lens 192 --batch_sizes 3100 --epoch 500 --clf_epoch 500 --stride 1 --lookback 64
python forecast_res.py --dataset exchange_rate --pred_lens 336 --batch_sizes 3100 --epoch 300 --clf_epoch 500 --stride 1 --lookback 64
python forecast_res.py --dataset exchange_rate --pred_lens 720 --batch_sizes 3100 --epoch 200 --clf_epoch 100 --stride 1 --lookback 100
