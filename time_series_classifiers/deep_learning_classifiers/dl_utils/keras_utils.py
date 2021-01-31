import os
import warnings

import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
warnings.simplefilter('ignore', category=DeprecationWarning)

from keras.models import Model
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K


def multi_label_log_loss(y_pred, y_true):
    return K.sum(K.binary_crossentropy(y_pred, y_true), axis=-1)


def train_model(model: Model, exercise, x_train, y_train, x_test, y_test, epochs=50, batch_size=128, learning_rate=1e-3,
                monitor='loss', optimization_mode='auto', compile_model=True):
    is_timeseries = True
    classes = np.unique(y_train)
    le = LabelEncoder()
    y_ind = le.fit_transform(y_train.ravel())
    recip_freq = len(y_train) / (len(le.classes_) * np.bincount(y_ind).astype(np.float64))
    class_weight = recip_freq[le.transform(classes)]

    print("Class weights : ", class_weight)

    y_train = to_categorical(y_train, len(np.unique(y_train)))
    y_test = to_categorical(y_test, len(np.unique(y_test)))

    if is_timeseries:
        factor = 1. / np.cbrt(2)
    else:
        factor = 1. / np.sqrt(2)
    print(os.getcwd())
    weight_fn = "./weights/hpe_weights_{}.h5".format(exercise)

    model_checkpoint = ModelCheckpoint(weight_fn, verbose=1, mode=optimization_mode,
                                       monitor=monitor, save_best_only=True, save_weights_only=True)
    reduce_lr = ReduceLROnPlateau(monitor=monitor, patience=100, mode=optimization_mode,
                                  factor=factor, cooldown=0, min_lr=1e-4, verbose=2)
    callback_list = [model_checkpoint, reduce_lr]

    optm = Adam(lr=learning_rate)

    if compile_model:
        model.compile(optimizer=optm, loss='categorical_crossentropy', metrics=['accuracy'])
    # with tf.device('/gpu:0'):
    #     model.fit(X, y, epochs=20, batch_size=128, callbacks=callbacks_list)
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=callback_list,
              class_weight=class_weight, verbose=2, validation_data=(x_test, y_test))


def evaluate_model(model: Model, exercise, x_test, y_test, batch_size=128):
    y_test = to_categorical(y_test, len(np.unique(y_test)))

    optm = Adam(lr=1e-3)
    model.compile(optimizer=optm, loss='categorical_crossentropy', metrics=['accuracy'])

    weight_fn = "./weights/hpe_weights_{}.h5".format(exercise)
    model.load_weights(weight_fn)

    print("\nEvaluating : ")
    loss, accuracy = model.evaluate(x_test, y_test, batch_size=batch_size)
    predictions = model.predict(x_test, batch_size=batch_size)
    # print("predictions ", predictions, y_test)
    confusion_matrix = metrics.confusion_matrix(y_test.argmax(axis=1), predictions.argmax(axis=1))
    classification_report = metrics.classification_report(y_test.argmax(axis=1), predictions.argmax(axis=1),
                                                          target_names=["a", "arch", "n", "r"])
    print("Confusion Matrix: \n", confusion_matrix)
    print("Classification report: \n", classification_report)
    print("Final Accuracy : ", accuracy)

    return accuracy, loss
