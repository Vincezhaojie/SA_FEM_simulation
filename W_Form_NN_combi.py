import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.utils import shuffle
from datetime import datetime
import pickle
import matplotlib.pyplot as plt

#load all modells
W_Form_yieled_classifier = load_model('W_Form_yieled_classifier.h5')
W_Form_class_classifier = load_model('W_Form_class_classifier.h5')
W_Form_NN_modelA = load_model('W_Form_NN_modelA.h5')
W_Form_NN_modelB = load_model('W_Form_NN_modelB.h5')
W_Form_NN_modelC = load_model('W_Form_NN_modelC.h5')
W_Form_NN_modelD = load_model('W_Form_NN_modelD.h5')
W_Form_NN_modelE = load_model('W_Form_NN_modelE.h5')

#load all scalers
W_Form_yieled_MinMaxScaler = MinMaxScaler()
pickle_in = open("W_Form_yieled_MinMaxScaler.pickle", "rb")
W_Form_yieled_MinMaxScaler = pickle.load(pickle_in)

W_Form_class_MinMaxScaler = MinMaxScaler()
pickle_in = open("W_Form_class_MinMaxScaler.pickle", "rb")
W_Form_class_MinMaxScaler = pickle.load(pickle_in)

W_Form_MinMaxScalerA = MinMaxScaler()
pickle_in = open("W_Form_MinMaxScalerA.pickle", "rb")
W_Form_MinMaxScalerA = pickle.load(pickle_in)

W_Form_MinMaxScalerB = MinMaxScaler()
pickle_in = open("W_Form_MinMaxScalerB.pickle", "rb")
W_Form_MinMaxScalerB = pickle.load(pickle_in)

W_Form_MinMaxScalerC = MinMaxScaler()
pickle_in = open("W_Form_MinMaxScalerC.pickle", "rb")
W_Form_MinMaxScalerC = pickle.load(pickle_in)

W_Form_MinMaxScalerD = MinMaxScaler()
pickle_in = open("W_Form_MinMaxScalerD.pickle", "rb")
W_Form_MinMaxScalerD = pickle.load(pickle_in)

W_Form_MinMaxScalerE = MinMaxScaler()
pickle_in = open("W_Form_MinMaxScalerE.pickle", "rb")
W_Form_MinMaxScalerE = pickle.load(pickle_in)


n_inputs = 9

#read data
df = pd.read_excel('W_Form_simulationDaten_test.xlsx')
X = df.drop(columns=['maxDisp(mm)', 'maxStress(MPa)', 'class', 'out'])
y = df['maxDisp(mm)']
print(len(df))

#1,yieled_classifier, work as a filter
X_nor = pd.DataFrame(W_Form_yieled_MinMaxScaler.transform(X.values), index=X.index, columns=X.columns)
is_yieled = W_Form_yieled_classifier.predict(X_nor)
is_yieled = np.argmax(is_yieled, axis=1)
is_yieled = is_yieled.astype(bool)
for i in range(len(is_yieled)):
    is_yieled[i] = not is_yieled[i]
X = X.iloc[is_yieled] #now the Daten in X are all not(!!!!!) out of yieled.
y = y.iloc[is_yieled]

#2, classfication in class 0 1 ... 4
X_nor = pd.DataFrame(W_Form_class_MinMaxScaler.transform(X.values), index=X.index, columns=X.columns)
predictions = W_Form_class_classifier.predict(X_nor)
predictions = np.argmax(predictions, axis=1)
predictions = predictions.flatten()
index_0 = np.argwhere(predictions == 0).flatten()
index_1 = np.argwhere(predictions == 1).flatten()
index_2 = np.argwhere(predictions == 2).flatten()
index_3 = np.argwhere(predictions == 3).flatten()
index_4 = np.argwhere(predictions == 4).flatten()

X_values = X.values
y_values = y.values

X_0_val, X_1_val, X_2_val, X_3_val, X_4_val = [], [], [], [], []
y_0_val, y_1_val, y_2_val, y_3_val, y_4_val = [], [], [], [], []
for index in index_0:
    X_0_val.append(X_values[index])
    y_0_val.append(y_values[index])
X_0 = pd.DataFrame(X_0_val, columns=X.columns)
for index in index_1:
    X_1_val.append(X_values[index])
    y_1_val.append(y_values[index])
X_1 = pd.DataFrame(X_1_val, columns=X.columns)
for index in index_2:
    X_2_val.append(X_values[index])
    y_2_val.append(y_values[index])
X_2 = pd.DataFrame(X_2_val, columns=X.columns)
for index in index_3:
    X_3_val.append(X_values[index])
    y_3_val.append(y_values[index])
X_3 = pd.DataFrame(X_3_val, columns=X.columns)
for index in index_4:
    X_4_val.append(X_values[index])
    y_4_val.append(y_values[index])
X_4 = pd.DataFrame(X_4_val, columns=X.columns)

#3, regression
#model A
X_0_nor = pd.DataFrame(W_Form_MinMaxScalerA.transform(X_0.values), index=X_0.index, columns=X_0.columns)
predictions = W_Form_NN_modelA.predict(X_0_nor)
y_min = min(y_0_val)

fig = plt.figure(figsize=(9, 8))
plt.plot(y_0_val, np.squeeze(predictions), 'o', alpha=0.4)
plt.xlabel('realer Wert')
plt.ylabel('Vorhersage')
plt.title('Modellgüte')

#model B
X_1_nor = pd.DataFrame(W_Form_MinMaxScalerB.transform(X_1.values), index=X_1.index, columns=X_1.columns)
predictions = W_Form_NN_modelB.predict(X_1_nor)
plt.plot(y_1_val, np.squeeze(predictions), 'o', alpha=0.4)

#model C
X_2_nor = pd.DataFrame(W_Form_MinMaxScalerC.transform(X_2.values), index=X_2.index, columns=X_2.columns)
predictions = W_Form_NN_modelC.predict(X_2_nor)
plt.plot(y_2_val, np.squeeze(predictions), 'o', alpha=0.4)

#model D
X_3_nor = pd.DataFrame(W_Form_MinMaxScalerD.transform(X_3.values), index=X_3.index, columns=X_3.columns)
predictions = W_Form_NN_modelD.predict(X_3_nor)
plt.plot(y_3_val, np.squeeze(predictions), 'o', alpha=0.4)

#model E
X_4_nor = pd.DataFrame(W_Form_MinMaxScalerE.transform(X_4.values), index=X_4.index, columns=X_4.columns)
predictions = W_Form_NN_modelE.predict(X_4_nor)
plt.plot(y_4_val, np.squeeze(predictions), 'o', alpha=0.4)
y_max = max(y_4_val)

plt.plot([1, 1], [0, y_max], '--', c=(0, 0, 0))
plt.plot([3, 3], [0, y_max], '--', c=(0, 0, 0))
plt.plot([7, 7], [0, y_max], '--', c=(0, 0, 0))
plt.plot([15, 15], [0, y_max], '--', c=(0, 0, 0))
plt.plot([y_min, y_max], [y_min, y_max], '--', c=(0, 0, 0))
plt.show()






















