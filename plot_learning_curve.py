import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#plot learning curve
def plot_learning_curve(model, X, y, hm_splice):
    X_val = X[0: int(0.2 * len(X))]
    y_val = y[0: int(0.2 * len(X))]
    X_tra = X[int(0.2 * len(X)):]
    y_tra = y[int(0.2 * len(X)):]

    all_mae = []
    percentage = np.linspace(0.1, 1, hm_splice)
    for i in range(hm_splice):
        X_tra_copy = X_tra.copy()
        y_tra_copy = y_tra.copy()
        X_tra_copy = X_tra_copy[0: int(len(X) * percentage[i])]
        y_tra_copy = y_tra_copy[0: int(len(X) * percentage[i])]
        model.fit(X_tra_copy, y_tra_copy, epochs=200)

        #cacalate MAE of validation daten
        y_pred = model.predict(X_val)
        y_val_tensor = tf.constant(y_val.as_matrix().reshape(-1, 1), dtype=tf.float32)
        _, mae = tf.metrics.mean_absolute_error(y_val_tensor, y_pred)
        init_g = tf.global_variables_initializer()
        init_l = tf.local_variables_initializer()
        with tf.Session() as sess:
            sess.run(init_g)
            sess.run(init_l)
            all_mae.append(mae.eval())

    learning_curve_x = [len(X) * percent for percent in np.linspace(0.1, 1, 10)]

    plt.figure(figsize=(12, 9))
    plt.plot(learning_curve_x, all_mae)
    plt.title('learning curve')
    plt.xlabel('size of trainingdaten')
    plt.ylabel('MAE')
    plt.show()

