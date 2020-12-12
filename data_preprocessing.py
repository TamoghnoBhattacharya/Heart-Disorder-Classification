import os
import sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import scipy
import sklearn
import scipy.io as sio

df = pd.read_csv('data/dataset.csv')
df_ecg = df.drop('3600',axis=1)
avg = df_ecg.mean(axis=1)
stddev = df_ecg.std(axis=1)
df_ecg=df_ecg.subtract(avg,axis=0)
df_ecg=df_ecg.div(stddev,axis=0)
df_ecg.round(4)
df_ecg['3600'] = df['3600']
df_ecg.to_csv('data/preprocessed_dataset.csv', index=False)
