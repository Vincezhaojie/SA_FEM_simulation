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
root_logdir = "tf_logs"
logdir = "{}/L_Form-{}".format(root_logdir, now)

n_inputs = 7

#read data
df1 = pd.read_excel('simulationDaten_1552308049255.xlsx')
df2 = pd.read_excel('simulationDaten_1552313754763.xlsx')
df3 = pd.read_excel('simulationDaten_1552320706273.xlsx')
df4 = pd.read_excel('simulationDaten_1552330987438.xlsx')
df5 = pd.read_excel('simulationDaten_1552335987567.xlsx')
df6 = pd.read_excel('simulationDaten_1552396072773.xlsx')
df7 = pd.read_excel('simulationDaten_1552409799089.xlsx')
df8 = pd.read_excel('simulationDaten_1552509408173.xlsx')
df = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8])

X_train = df.drop(columns=['maxDisp(mm)'])
y_train = df['maxDisp(mm)']

scaler = MinMaxScaler()  # or MaxAbsScaler(), MinMaxScaler()
X_train_nor = pd.DataFrame(scaler.fit_transform(X_train.values), index=X_train.index, columns=X_train.columns)

#save the scaler, prepare to Test
pickle_out = open("MinMaxScaler.pickle", "wb")
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

NN_model.fit(X_train_nor, y_train, epochs=800, validation_split=0.2, callbacks=[tensorboard])

NN_model.save('NN_model.h5')













