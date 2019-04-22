import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.utils import shuffle
from datetime import datetime
import pickle

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

def create_model(neurons=10, hidden_layers=1):
    model = Sequential()
    #input layer
    model.add(Dense(n_inputs, kernel_initializer='normal', input_shape=X_train_nor.shape[1:], activation='relu'))

    #hidden layers
    for i in range(hidden_layers):
        model.add(Dense(neurons, kernel_initializer='normal', activation='relu'))

    #output layer
    model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model




now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "W_Form_logs"
logdir = "{}/W_Form_yieled-{}".format(root_logdir, now)

n_inputs = 9
num_classes = 2

#read data
df = pd.read_excel('W_Form_Daten_withClass.xlsx')
df = shuffle(df)
print(len(df))

X_train = df.drop(columns=['maxDisp(mm)', 'maxStress(MPa)', 'class', 'out'])
y_train = df['out']
y_train = to_categorical(y_train, num_classes)
scaler = MinMaxScaler()  # or MaxAbsScaler(), MinMaxScaler()
X_train_nor = pd.DataFrame(scaler.fit_transform(X_train.values), index=X_train.index, columns=X_train.columns)

#save the scaler, prepare to Test
pickle_out = open("W_Form_yieled_MinMaxScaler.pickle", "wb")
pickle.dump(scaler, pickle_out)
pickle_out.close()


# model = KerasClassifier(build_fn=create_model, verbose=0, epochs=50)
# neurons = [n for n in range(60, 120)]
# hidden_layers = [1, 2, 3, 4, 5]
# param_grid = dict(neurons=neurons, hidden_layers=hidden_layers)
# grid = RandomizedSearchCV(estimator=model, param_distributions =param_grid, n_jobs=-1, n_iter=20)
# grid_result = grid.fit(X_train_nor, y_train)
#
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
#     print("%f (%f) with: %r" % (mean, stdev, param))



NN_model = create_model(99, 5)
tensorboard = TensorBoard(log_dir=logdir)

NN_model.fit(X_train_nor, y_train, epochs=50, validation_split=0.2, callbacks=[tensorboard])

NN_model.save('W_Form_yieled_classifier.h5')

frozen_graph = freeze_session(tf.keras.backend.get_session(), output_names=[out.op.name for out in NN_model.outputs])
output_names = [out.op.name for out in NN_model.outputs]
print(output_names)
# tf.train.write_graph(frozen_graph, './', 'W_Form_NN_model.pbtxt', as_text=True)
tf.train.write_graph(frozen_graph, './', 'W_Form_yieled_classifier.pb', as_text=False)

# plot_learning_curve(NN_model, X_train_nor, y_train, 15)




















