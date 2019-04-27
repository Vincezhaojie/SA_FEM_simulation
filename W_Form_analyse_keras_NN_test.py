import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras import losses
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import matplotlib.pyplot as plt
import pickle

up = 3
down = 1
model_name = 'W_Form_NN_modelB.h5'
scaler_name = "W_Form_MinMaxScalerB.pickle"


NN_model = load_model(model_name)

df_test = pd.read_excel('W_Form_SimulationDaten_test.xlsx')

df_test = df_test[df_test['maxDisp(mm)'] > down]
df_test = df_test[df_test['maxDisp(mm)'] <= up]
df_test = df_test[df_test['maxStress(MPa)'] < 351.6]


X_test = df_test.drop(columns=['maxDisp(mm)', 'maxStress(MPa)', 'class', 'out'])
y_test = df_test['maxDisp(mm)']

scaler = MinMaxScaler()
pickle_in = open(scaler_name, "rb")
scaler = pickle.load(pickle_in)

X_test_nor = pd.DataFrame(scaler.transform(X_test.values), index=X_test.index, columns=X_test.columns)
predictions = NN_model.predict(X_test_nor)
with tf.Session() as sess:
    mae = tf.keras.metrics.mean_absolute_error(y_test, np.squeeze(predictions)).eval()
    print(mae)

fig = plt.figure(figsize=(9, 8))
plt.plot(y_test, np.squeeze(predictions), 'o', alpha=0.4)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', c=(0, 0, 0))
plt.plot([y_test.min(), y_test.max()], [y_test.min() + 0.2, y_test.max() + 0.2], '--', c=(0, 0, 0))
plt.plot([y_test.min(), y_test.max()], [y_test.min() - 0.2, y_test.max() - 0.2], '--', c=(0, 0, 0))
plt.xlabel('realer Wert')
plt.ylabel('Vorhersage')
plt.title('Modellgüte')
plt.show()



