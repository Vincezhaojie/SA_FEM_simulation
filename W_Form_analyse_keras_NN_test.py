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

#########################################################
up = 50
down = 15
model_name = 'W_Form_F3=0_NN_modelE.h5'
scaler_name = "W_Form_F3=0_MinMaxScalerE.pickle"
#########################################################


NN_model = load_model(model_name)

#df_test = pd.read_excel('W_Form_SimulationDaten_test.xlsx')
#df_test = pd.read_excel('W_Form_partE_as_test.xlsx')
df_test = pd.read_excel('W_Form_F3=0_test.xlsx')

df_test = df_test[df_test['maxDisp(mm)'] > down]
df_test = df_test[df_test['maxDisp(mm)'] <= up]
df_test = df_test[df_test['maxStress(MPa)'] < 351.6]


#X_test = df_test.drop(columns=['maxDisp(mm)', 'maxStress(MPa)', 'class', 'out'])
X_test = df_test.drop(columns=['maxDisp(mm)', 'maxStress(MPa)'])
y_test = df_test['maxDisp(mm)']

scaler = MinMaxScaler()
pickle_in = open(scaler_name, "rb")
scaler = pickle.load(pickle_in)

X_test_nor = pd.DataFrame(scaler.transform(X_test.values), index=X_test.index, columns=X_test.columns)
predictions = NN_model.predict(X_test_nor)
with tf.Session() as sess:
    mape = tf.keras.metrics.MAPE(y_test.values, np.squeeze(predictions)).eval()
    mae = tf.keras.metrics.MAE(y_test.values, np.squeeze(predictions)).eval()
    print(mape)

tol = round(mae, 2)
fig = plt.figure(figsize=(9, 8))
plt.plot(y_test, np.squeeze(predictions), 'o', alpha=0.4)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', c=(0, 0, 0))
plus = plt.plot([y_test.min(), y_test.max()], [y_test.min() + tol, y_test.max() + tol], '--', c=(0, 0, 0), label="+" + str(tol), color='green')
minus = plt.plot([y_test.min(), y_test.max()], [y_test.min() - tol, y_test.max() - tol], '--', c=(0, 0, 0), label="-" + str(tol), color='green')
plt.legend(loc='upper left')
plt.xlabel('realer Wert')
plt.ylabel('Vorhersage')
plt.title('Modellgüte    ' + 'MAPE=' + str(round(mape, 2)) + '%')
plt.show()



