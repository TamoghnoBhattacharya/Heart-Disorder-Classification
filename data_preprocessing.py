import os
import sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import scipy
import sklearn
import scipy.io as sio

def data_clean_combined():
    df = pd.read_csv('data/dataset.csv')
    df_ecg = df.drop('3600',axis=1)
    avg = df_ecg.mean(axis=1)
    diff = df_ecg.max(axis=1) - df_ecg.min(axis=1)
    df_ecg=df_ecg.subtract(avg,axis=0)
    df_ecg=df_ecg.div(diff,axis=0)
    df_ecg['3600'] = df['3600']
    return df_ecg.to_csv('data/dataset.csv', index=False)

def data_clean_single():
    df_ecg = pd.read_csv('data/data.csv')
    avg = df_ecg.mean(axis=1)
    diff = df_ecg.max(axis=1) - df_ecg.min(axis=1)
    df_ecg=df_ecg.subtract(avg,axis=0)
    df_ecg=df_ecg.div(diff,axis=0)
    return df_ecg.to_csv('data/data.csv', index=False)

data_clean_combined()
