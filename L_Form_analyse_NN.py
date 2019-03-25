import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from datetime import datetime


n_inputs = 10
n_hidden1 = 500
n_hidden2 = 400
n_hidden3 = 200
n_outputs = 1
learning_rate = 0.01
n_epoch = 150
batch_size = 100

now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/L_Form-{}".format(root_logdir, now)

#read data
df1 = pd.read_excel('simulationDaten_1552308049255.xlsx')
df2 = pd.read_excel('simulationDaten_1552313754763.xlsx')
df3 = pd.read_excel('simulationDaten_1552320706273.xlsx')
df = pd.concat([df1, df2, df3])
df['Elastic Modulus(N/mm^2)'] = 200000
df['Poissons Ration'] = 0.29
df['Shear Modulus(N/mm^2)'] = 77000

train_size = 0.8
X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['maxDisp(mm)']),
                                                    df['maxDisp(mm)'],
                                                    test_size=1-train_size,
                                                    random_state=88)

scaler = StandardScaler()  # or MaxAbsScaler(), MinMaxScaler()
X_train_nor = pd.DataFrame(scaler.fit_transform(X_train.values), index=X_train.index, columns=X_train.columns)
X_test_nor = pd.DataFrame(scaler.transform(X_test.values), index=X_test.index, columns=X_test.columns)

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X')
y = tf.placeholder(tf.float32, shape=(None), name='y')

activation = tf.nn.relu
he_init = tf.contrib.layers.variance_scaling_initializer()
hidden1 = tf.contrib.layers.fully_connected(X, n_hidden1, activation_fn=activation, weights_initializer=he_init)
hidden2 = tf.contrib.layers.fully_connected(hidden1, n_hidden2, activation_fn=activation, weights_initializer=he_init)
hidden3 = tf.contrib.layers.fully_connected(hidden2, n_hidden3, activation_fn=activation, weights_initializer=he_init)
predict = tf.contrib.layers.fully_connected(hidden3, n_outputs, activation_fn=None, weights_initializer=he_init)


def fetch_batch(epoch, batch_index, batch_size):
    X_batch = X_train_nor[batch_index * batch_size: (batch_index + 1) * batch_size]
    y_batch = y_train[batch_index * batch_size: (batch_index + 1) * batch_size]
    return X_batch, y_batch


loss = tf.losses.mean_squared_error(y, predict) ** 0.5  #RMSE
optimizer = tf.train.AdamOptimizer(learning_rate)
train_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epoch):
        for batch_index in range(len(X_train_nor) // batch_size):
            X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
            sess.run(train_op, feed_dict={X: X_batch, y: y_batch})
        sess.run(train_op, feed_dict={X: X_batch, y: y_batch})
        RMSE_train = loss.eval(feed_dict={X: X_batch, y: y_batch})
        RMSE_test = loss.eval(feed_dict={X: X_test_nor, y: y_test})
        print(epoch, "RMSE_train:", RMSE_train, "  RMSE_test:", RMSE_test)




























