# Forecast Evaluation Metrics
# ---------------------------
# This file includes the mathematical formulation and Python + MATLAB implementation
# for four key metrics used to compare forecasted sequences against actual values.

# ---------------------------
# Python Implementations
# ---------------------------

import numpy as np
import pandas as pd
import os
from typing import List
from dtaidistance import dtw
from properscoring import crps_ensemble
from dtaidistance import dtw


# 1. MAE - Mean Absolute Error
def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean(np.abs(y_true - y_pred))

def mae_std(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.std(np.abs(y_true - y_pred))

# 2. DTW - Dynamic Time Warping Distance
def dtw_distance(y_true: List[float], y_pred: List[float]) -> float:
    return dtw.distance(y_true, y_pred)

# 3. Rolling RMSE
def rolling_rmse(y_true: np.ndarray, y_pred: np.ndarray, window: int = 5) -> np.ndarray:
    errors = (y_true - y_pred)**2
    rolling = np.convolve(errors, np.ones(window)/window, mode='valid')
    return np.sqrt(rolling)

def rolling_rmse_stats(y_true: np.ndarray, y_pred: np.ndarray, window: int = 5):
    rmse_vals = []
    for i in range(y_true.shape[0]):
        rr = rolling_rmse(y_true[i], y_pred[i], window)
        rmse_vals.append(rr)
    rmse_vals = np.concatenate(rmse_vals)
    return np.mean(rmse_vals), np.std(rmse_vals)

# 4. CRPS - Continuous Ranked Probability Score
def crps_score(y_true: np.ndarray, y_samples: np.ndarray) -> float:
    return np.mean(crps_ensemble(y_true, y_samples))

def crps_std(y_true: np.ndarray, y_samples: np.ndarray) -> float:
    return np.std(crps_ensemble(y_true, y_samples))


def evaluate_forecast(Y_TEST, model_name):
    results = []
    for trial in range (0, 49):
        test_path = f"/home/chiara/Energy/PRED/{model_name}_predictions_{trial}.csv"
        Y_PREDICT = pd.read_csv(test_path, header=0, sep=',', index_col="index")
        print(f'The predicted test set (trial {trial})  has size of: {Y_PREDICT.shape}')
        for row in range(1,127):

            # Ensure the row index is valid for Y_PREDICT.index
            if row >= len(Y_PREDICT.index):
                continue

            # Split the index string and ensure the split list has enough elements
            split_index = Y_PREDICT.index[row].split("_")
            if len(split_index) <= 1:
                continue

            key = split_index[3]  # Use the second element of the split list as the key
            pred_filtered_rows = Y_PREDICT[Y_PREDICT.index.str.contains(f"_{key}_")].to_numpy()
            y_pred = np.concatenate(pred_filtered_rows, axis=0)
            true_filtered_rows = Y_TEST[Y_TEST.index.str.contains(f"_{key}_")].to_numpy()
            y_true = np.concatenate(true_filtered_rows, axis=0)
            
            mae = mean_absolute_error(y_true, y_pred)
            mae_s = mae_std(y_true, y_pred)
            # Ensure dtw_distance is defined


            

            # Compute DTW distance
            
            dtw_dist =dtw.distance(y_true, y_pred)
            rmse_m, rmse_s = rolling_rmse_stats(y_true, y_pred)
            crps = crps_score(y_true, y_pred)
            crps_s = crps_std(y_true, y_pred)
            results.append({
                "trial": trial,
                "key": key,
                "MAE": mae,
                "MAE_std": mae_s,
                "DTW": dtw_dist,
                "Rolling_RMSE": rmse_m,
                "Rolling_RMSE_std": rmse_s,
                "CRPS": crps,
                "CRPS_std": crps_s
            })

    return pd.DataFrame(results)
 



for model_name in ["VAE", "gan"]:
    res = []
    test_path = "/home/chiara/Energy/Data/Test_2m_72.csv" 
    Y_TEST = pd.read_csv(test_path, header=0, sep=';', decimal=",", index_col="index")
    Y_TEST = Y_TEST[[col for col in Y_TEST.columns if 'y' in col]]
    #print(f'The actual test set has size of: {Y_TEST.shape}')
    res = evaluate_forecast(Y_TEST, 'VAE')
    res.to_csv(f"/home/chiara/Energy/METRICS/{model_name}_evaluation_summary.csv", index=False)
