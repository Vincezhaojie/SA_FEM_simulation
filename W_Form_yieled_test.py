import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import confusion_matrix; import itertools
from tensorflow.keras.utils import to_categorical

num_classes = 2

NN_model = load_model('W_Form_yieled_classifier.h5')

df_test = pd.read_excel('W_Form_simulationDaten_test.xlsx')

X_test = df_test.drop(columns=['maxDisp(mm)', 'maxStress(MPa)', 'out', 'class'])
y_test = df_test['out']

scaler = MinMaxScaler()
pickle_in = open("W_Form_yieled_MinMaxScaler.pickle", "rb")
scaler = pickle.load(pickle_in)

X_test_nor = pd.DataFrame(scaler.transform(X_test.values), index=X_test.index, columns=X_test.columns)


predictions = NN_model.predict(X_test_nor)
predictions = np.argmax(predictions, axis=1)

y_now = y_test
X_now = X_test_nor


matrix = confusion_matrix(y_now, predictions).transpose()
labels = y_now.unique()
plt.figure(figsize=(10, 6))
plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Konfusionsmatrix auf Testdaten')
plt.ylabel('Vorhersage')
plt.xlabel('Realer Wert')
tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels, rotation=45)
plt.yticks(tick_marks, labels)
fmt = 'd'; thresh = matrix.max() / 2.
for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
    plt.text(j, i, format(matrix[i, j], fmt),horizontalalignment="center",color="white" if matrix[i, j] > thresh else "black")
plt.show()
