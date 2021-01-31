import matplotlib.pyplot as plt
import numpy as np
import keras

"""

If you find either the codes or the results are helpful to your work, please kindly cite our paper

[Time Series Classification from Scratch with Deep Neural Networks: A Strong Baseline] (https://arxiv.org/abs/1611.06455)

[Imaging Time-Series to Improve Classification and Imputation] (https://arxiv.org/abs/1506.00327)
"""


def draw_cam(model, x_test, y_test):
    get_last_conv = keras.backend.function([model.layers[0].input, keras.backend.learning_phase()],
                                           [model.layers[-3].output])
    last_conv = get_last_conv([x_test[:100], 1])[0]

    get_softmax = keras.backend.function([model.layers[0].input, keras.backend.learning_phase()], [model.layers[-1].output])
    softmax = get_softmax(([x_test[:100], 1]))[0]
    softmax_weight = model.get_weights()[-2]
    CAM = np.dot(last_conv, softmax_weight)

    for k in range(20):
        CAM = (CAM - CAM.min(axis=1, keepdims=True)) / (CAM.max(axis=1, keepdims=True) - CAM.min(axis=1, keepdims=True))
        c = np.exp(CAM) / np.sum(np.exp(CAM), axis=1, keepdims=True)
        plt.figure(figsize=(13, 7));
        plt.plot(x_test[k].squeeze());
        plt.scatter(np.arange(len(x_test[k])), x_test[k].squeeze(), cmap='hot_r', c=c[k, :, :, int(y_test[k])].squeeze(),
                    s=100);
        plt.title(
            'True label:' + str(y_test[k]) + '   likelihood of label ' + str(y_test[k]) + ': ' + str(
                softmax[k][int(y_test[k])]))
        plt.colorbar();
