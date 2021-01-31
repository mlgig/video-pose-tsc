# FCN model
# when tuning start with learning rate->mini_batch_size ->
# momentum-> #hidden_units -> # learning_rate_decay -> #layers
"""
References
@article{IsmailFawaz2018deep,
  Title                    = {Deep learning for time series classification: a review},
  Author                   = {Ismail Fawaz, Hassan and Forestier, Germain and Weber, Jonathan and Idoumghar, Lhassane and Muller, Pierre-Alain},
  journal                  = {Data Mining and Knowledge Discovery},
  Year                     = {2019},
  volume                   = {33},
  number                   = {4},
  pages                    = {917--963},
}

"""
import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import time
from sklearn import metrics

from time_series_classifiers.deep_learning_classifiers.utils_functions import save_logs
from time_series_classifiers.deep_learning_classifiers.utils_functions import calculate_metrics
from utils.program_stats import timeit


class Classifier_CNN:

    def __init__(self, output_directory, exercise, input_shape, nb_classes, verbose=False,build=True):
        self.output_directory = output_directory

        if build == True:
            self.model = self.build_model(input_shape, nb_classes)
            if verbose:
                self.model.summary()
            self.verbose = verbose
            self.model.save_weights(self.output_directory + 'model_init.hdf5')
        self.model_name = "cnn"
        self.exercise = exercise
        return

    def build_model(self, input_shape, nb_classes):
        padding = 'valid'
        input_layer = keras.layers.Input(input_shape)

        if input_shape[0] < 60:  # for italypowerondemand dataset
            padding = 'same'

        conv1 = keras.layers.Conv1D(filters=6, kernel_size=7, padding=padding, activation='sigmoid')(input_layer)
        # kernel shape = 7 * 8
        # output shape = 218 * 6
        # number of parameter   s = filters * kernel size + biases
        # 6 * 7 * 8 + 6
        conv1 = keras.layers.AveragePooling1D(pool_size=3)(conv1)
        # output shape = 72 * 6

        conv2 = keras.layers.Conv1D(filters=12, kernel_size=7, padding=padding, activation='sigmoid')(conv1)
        # output shape = 66 * 12
        # parameters  = 12 * 7 * 6 + 12
        conv2 = keras.layers.AveragePooling1D(pool_size=3)(conv2)
        # output shape = 22 * 12

        flatten_layer = keras.layers.Flatten()(conv2)

        # output shape = 264

        output_layer = keras.layers.Dense(units=nb_classes, activation='sigmoid')(flatten_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(),
                      metrics=['accuracy'])

        file_path = self.output_directory + '/best_model.hdf5'

        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss',
                                                           save_best_only=True)

        self.callbacks = [model_checkpoint]

        return model

    @timeit
    def fit(self, x_train, y_train, x_val, y_val, y_true,  nb_epochs, mini_batch_size=16):
        if not tf.test.is_gpu_available:
            print('error')
            exit()

        # x_val and y_val are only used to monitor the test loss and NOT for training
        start_time = time.time()

        hist = self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=nb_epochs,
                              verbose=self.verbose, validation_data=(x_val, y_val), callbacks=self.callbacks)

        duration = time.time() - start_time

        self.model.save(self.output_directory+'/last_model.hdf5')

        # y_pred = model.predict(x_val)
        #
        # # convert the predicted from binary to integer
        # y_pred = np.argmax(y_pred, axis=1)

        # save_logs(self.output_directory, hist, y_pred, y_true, duration,lr=False)

        keras.backend.clear_session()

    def predict(self, x_test, y_true, enc, return_df_metrics=True):
        model_path = self.output_directory + '/best_model.hdf5'
        model = keras.models.load_model(model_path)
        y_pred = model.predict(x_test)
        y_pred = np.argmax(y_pred, axis=1)
        y_pred_label = np.array([enc.categories_[0][i] for i in y_pred])
        y_true_label = np.array([enc.categories_[0][i] for i in y_true])
        confusion_matrix = metrics.confusion_matrix(y_true_label, y_pred_label)
        classification_report = metrics.classification_report(y_true_label, y_pred_label)
        print(confusion_matrix)
        print(classification_report)
        df_metrics = calculate_metrics(y_true, y_pred, 0.0)
        return confusion_matrix, classification_report, df_metrics
