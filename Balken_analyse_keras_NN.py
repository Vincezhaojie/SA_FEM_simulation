import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import pickle

now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "balken_logs"
logdir = "{}/baklen-{}".format(root_logdir, now)

n_inputs = 4

#read data
df1 = pd.read_excel('balken_simulationDaten_1552684302213.xlsx')

df = pd.concat([df1])

X_train = df.drop(columns=['maxStress(MPa)', 'maxDisplacement(mm)'])
y_train = df['maxDisplacement(mm)']

scaler = MinMaxScaler()  # or MaxAbsScaler(), MinMaxScaler()
X_train_nor = pd.DataFrame(scaler.fit_transform(X_train.values), index=X_train.index, columns=X_train.columns)

#save the scaler, prepare to Test
pickle_out = open("balken_MinMaxScaler.pickle", "wb")
pickle.dump(scaler, pickle_out)
pickle_out.close()

NN_model = Sequential()
#input layer
NN_model.add(Dense(n_inputs, kernel_initializer='normal', input_shape=X_train_nor.shape[1:], activation='relu'))

#hidden layers
NN_model.add(Dense(512, kernel_initializer='normal', activation='relu'))
NN_model.add(Dense(512, kernel_initializer='normal', activation='relu'))
NN_model.add(Dense(512, kernel_initializer='normal', activation='relu'))


#output layer
NN_model.add(Dense(1, kernel_initializer='normal', activation='linear'))

tensorboard = TensorBoard(log_dir=logdir)

NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])

NN_model.fit(X_train_nor, y_train, epochs=300, validation_split=0.2, callbacks=[tensorboard])

NN_model.save('balken_NN_model.h5')




















