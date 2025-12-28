# SISSEL


# Environment  
python 3.9.16  
torch==2.0.0  
numpy==1.26.4  
pandas==1.5.3  
scikit-learn==1.2.2  
sktime==0.33.0  

# Data  


Time series forecasting datasets can be downloaded from: https://github.com/thuml/Time-Series-Library  
After downloading, the data structure should be: all_datasets/long_term_forecast/* where * represents the dataset names (ETT-small, exchange_rate, traffic, electricity).

Time series classification datasets can be downloaded from: https://www.timeseriesclassification.com/index.php  


Note: make sure to download the ‘.ts’ format files.  
After downloading, the data structure should be: Multivariate_ts/* where * is the name of each dataset.

# Running Scripts  


## Time Series Forecasting
The script_forecast.sh file contains all running commands. You can directly execute:
```bash
./script_forecast.sh
```
Alternatively, you can run an individual script, for example: 
```bash 
python forecast_res.py --dataset electricity --pred_lens 96 --batch_sizes 573 --epoch 100 --clf_epoch 500 --stride 1 --lookback 200
```
## Time series classification
The script_UEA.sh file contains all running commands. You can directly execute:

```bash
./script_UEA.sh
```
Alternatively, you can run an individual script, for example:  
```bash
python clf_res.py --dataset AtrialFibrillation --lookback 640 --stride 1 --batch_size 15 --epochs 300 --clf_epochs 100 --shuffle False
```
