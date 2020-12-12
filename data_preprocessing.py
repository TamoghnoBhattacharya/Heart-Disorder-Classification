import os
import sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline
import scipy
import sklearn
import scipy.io as sio

df = pd.read_csv('data/dataset.csv')
df=df.drop('Unnamed: 0', axis=1)
train=df
test = df.drop('3600',axis=1)
df_ecg = test
avg = df_ecg.mean(axis=1)
stddev = df_ecg.std(axis=1)
df_ecg=df_ecg.subtract(avg,axis=0)
df_ecg=df_ecg.div(stddev,axis=0)
df_ecg.round(4)