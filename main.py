import data_to_csv
import data_preprocessing
import train
import os
import sys
import numpy as np
import pandas as pd
import keras
from keras import models
from keras.models import load_model


PATH = 'PATH of MAT file'
NEW_PATH = 'PATH of CSV file'
names = ['1 NSR', '2 APB', '3 AFL', '4 AFIB', '5 SVTA', '6 WPW', '7 PVC', '8 Bigeminy', '9 Trigeminy', '10 VT', '11 IVR', '12 VFL', '13 Fusion', '14 LBBBB', '15 RBBBB', '16 SDHB', '17 PR']


data_to_csv.new_mat_to_csv(PATH, NEW_PATH)
data_preprocessing.data_clean_single(NEW_PATH, NEW_PATH)
data = pd.read_csv(NEW_PATH)

model = load_model('models/cnn_model_1.h5')
# train.training_process_multiple(PATH)
# train.cnn_model()

data=np.asarray(data).reshape(-1,3600,1)
pred = model.predict_classes(data)
print(names[pred[0]])
