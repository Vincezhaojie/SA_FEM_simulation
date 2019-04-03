import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from datetime import datetime
import pickle
import matplotlib.pyplot as plt
from plot_learning_curve import plot_learning_curve

#help function
def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ''
        frozen_graph = tf.compat.v1.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph

now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "W_Form_logs"
logdir = "{}/W_Form_partB-{}".format(root_logdir, now)

n_inputs = 9

#read data
df1 = pd.read_excel('W_Form_simulationDaten_1553758068644_clean.xlsx')
df2 = pd.read_excel('W_Form_simulationDaten_1553765907570_clean.xlsx')
df3 = pd.read_excel('W_Form_simulationDaten_1553765909540_clean.xlsx')
df4 = pd.read_excel('W_Form_simulationDaten_1553902548685_double_clean.xlsx')
df5 = pd.read_excel('W_Form_big_data_complecated_part1.xlsx')
df6 = pd.read_excel('W_Form_partB_completed.xlsx')
df = pd.concat([df1, df2, df3, df4, df5, df6])

df = df[df['maxDisp(mm)'] > 1]
df = df[df['maxDisp(mm)'] <= 3]


df = shuffle(df)

print(len(df))

X_train = df.drop(columns=['maxDisp(mm)', 'maxStress(MPa)'])
y_train = df['maxDisp(mm)']

scaler = MinMaxScaler()  # or MaxAbsScaler(), MinMaxScaler()
X_train_nor = pd.DataFrame(scaler.fit_transform(X_train.values), index=X_train.index, columns=X_train.columns)

#save the scaler, prepare to Test
pickle_out = open("W_Form_MinMaxScalerB.pickle", "wb")
pickle.dump(scaler, pickle_out)
pickle_out.close()

NN_model = Sequential()
#input layer
NN_model.add(Dense(n_inputs, kernel_initializer='normal', input_shape=X_train_nor.shape[1:], activation='relu'))

#hidden layers
NN_model.add(Dense(512, kernel_initializer='normal', activation='relu'))
NN_model.add(Dense(512, kernel_initializer='normal', activation='relu'))
NN_model.add(Dense(512, kernel_initializer='normal', activation='relu'))
NN_model.add(Dense(512, kernel_initializer='normal', activation='relu'))
NN_model.add(Dense(512, kernel_initializer='normal', activation='relu'))
NN_model.add(Dropout(0.5))
NN_model.add(Dense(512, kernel_initializer='normal', activation='relu'))
NN_model.add(Dropout(0.5))
NN_model.add(Dense(512, kernel_initializer='normal', activation='relu'))
NN_model.add(Dropout(0.5))

#output layer
NN_model.add(Dense(1, kernel_initializer='normal', activation='linear'))

tensorboard = TensorBoard(log_dir=logdir)

NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])

NN_model.fit(X_train_nor, y_train, epochs=2000, validation_split=0.2, callbacks=[tensorboard])

NN_model.save('W_Form_NN_modelB.h5')

frozen_graph = freeze_session(tf.keras.backend.get_session(), output_names=[out.op.name for out in NN_model.outputs])
output_names = [out.op.name for out in NN_model.outputs]
print(output_names)
# tf.train.write_graph(frozen_graph, './', 'W_Form_NN_model.pbtxt', as_text=True)
tf.train.write_graph(frozen_graph, './', 'W_Form_NN_modelB.pb', as_text=False)

# plot_learning_curve(NN_model, X_train_nor, y_train, 15)


























