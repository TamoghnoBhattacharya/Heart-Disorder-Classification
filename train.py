import numpy as np
import pandas as pd
from keras.layers import Dense, Conv1D, MaxPooling1D, BatchNormalization, Dropout, Activation, MaxPooling2D, Flatten
from keras.models import Sequential, load_model
from keras.losses import categorical_crossentropy, binary_crossentropy
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

def training_process_multiple(PATH):
    data = pd.read_csv(PATH)
    train_y = data['3600'] 
    train_X = data.drop('3600', axis=1)
    test = data.sample(frac=0.2)
    test_y = test['3600'] 
    test_X = test.drop('3600', axis=1)
    train_y = to_categorical(train_y, 17)
    test_y = to_categorical(test_y, 17)

    train_X = np.asarray(train_X).reshape(-1,3600,1)
    train_y = np.asarray(train_y)
    test_X = np.asarray(test_X).reshape(-1,3600,1)
    test_y = np.asarray(test_y)
    


def cnn_model():
    model = Sequential()
    model.add(Conv1D(128,50,activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=(2)))

    model.add(Conv1D(32,7,activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=(2)))
    model.add(Conv1D(32,10,activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv1D(128,5,activation='relu'))
    model.add(MaxPooling1D(pool_size=(2)))

    model.add(Conv1D(256,15,activation='relu'))
    model.add(MaxPooling1D(pool_size=(2)))
    model.add(Conv1D(512,5,activation='relu'))
    model.add(Conv1D(128,3,activation='relu'))   
    model.add(Flatten())

    model.add(Dense(512,activation='relu'))
    model.add(Dense(17,activation='softmax'))

    return model

def model_fit(model):
    training_process_multiple('data/preprocessed_dataset.csv')
    model = cnn_model()
    model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
    model.fit(x=train_X, y=train_y, epochs=10, batch_size=16, verbose = 2, validation_data=(test_X, test_y))
    model.save('models/cnn_model_1.h5')

model_fit(model)


