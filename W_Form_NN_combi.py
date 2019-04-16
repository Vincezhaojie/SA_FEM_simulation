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

#load all modells
W_Form_yieled_classifier = load_model('W_Form_yieled_classifier.h5')
W_Form_NN_modelA = load_model('W_Form_NN_modelA.h5')
W_Form_NN_modelB = load_model('W_Form_NN_modelB.h5')
W_Form_NN_modelC = load_model('W_Form_NN_modelC.h5')
W_Form_NN_modelD = load_model('W_Form_NN_modelD.h5')
W_Form_NN_modelE = load_model('W_Form_NN_modelE.h5')

#load all scalers
W_Form_yieled_MinMaxScaler = MinMaxScaler()
pickle_in = open("W_Form_yieled_MinMaxScaler.pickle", "rb")
W_Form_yieled_MinMaxScaler = pickle.load(pickle_in)

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
df = pd.read_excel('W_Form_Daten_withClass.xlsx')
#df = shuffle(df)
X = df.drop(columns=['maxDisp(mm)', 'maxStress(MPa)', 'class', 'out'])
y = df['maxDiap(mm)']
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

#2, classfication in class A B ... E




































