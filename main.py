from data_to_csv import new_mat_to_csv
from data_preprocessing import data_clean_single
import os
import sys
import numpy as np
import pandas as pd
import keras
from keras import models
from keras.models import load_model


PATH = 'MLII/1 NSR/100m (0).mat'
names = ['1 NSR', '2 APB', '3 AFL', '4 AFIB', '5 SVTA', '6 WPW', '7 PVC', '8 Bigeminy', '9 Trigeminy', '10 VT', '11 IVR', '12 VFL', '13 Fusion', '14 LBBBB', '15 RBBBB', '16 SDHB', '17 PR']


new_mat_to_csv(PATH)
data_clean_single()
data = pd.read_csv('data/data.csv')
data=np.asarray(data).reshape(-1,3600,1)

model = load_model('models/cnn_model_1.h5')
pred = model.predict_classes(data)
print(names[pred[0]])
