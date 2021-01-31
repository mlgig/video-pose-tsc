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


class Classifier_FCN:

    def __init__(self, output_directory, exercise, input_shape, nb_classes, verbose=True, build=True):
        self.output_directory = output_directory
        if build == True:
            self.model = self.build_model(input_shape, nb_classes)
            if verbose:
                self.model.summary()
            self.verbose = verbose
            self.model.save_weights(self.output_directory + 'model_init.hdf5')
        self.model_name = "fcn"
        self.exercise = exercise
        return

    def build_model(self, input_shape, nb_classes):
        input_layer = keras.layers.Input(input_shape)

        conv1 = keras.layers.Conv1D(filters=128, kernel_size=8, padding='same')(input_layer)
        conv1 = keras.layers.BatchNormalization()(conv1)
        conv1 = keras.layers.Activation(activation='relu')(conv1)

        conv2 = keras.layers.Conv1D(filters=256, kernel_size=5, padding='same')(conv1)
        conv2 = keras.layers.BatchNormalization()(conv2)
        conv2 = keras.layers.Activation('relu')(conv2)

        conv3 = keras.layers.Conv1D(128, kernel_size=3, padding='same')(conv2)
        conv3 = keras.layers.BatchNormalization()(conv3)
        conv3 = keras.layers.Activation('relu')(conv3)

        gap_layer = keras.layers.GlobalAveragePooling1D()(conv3)

        output_layer = keras.layers.Dense(nb_classes, activation='softmax')(gap_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(),
                      metrics=['accuracy'])

        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50,
                                                      min_lr=0.0001)

        file_path = self.output_directory + '/best_model.hdf5'

        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss',
                                                           save_best_only=True)

        self.callbacks = [reduce_lr, model_checkpoint]

        return model

    @timeit
    def fit(self, x_train, y_train, x_val, y_val, y_true, nb_epochs, batch_size=16):
        if not tf.test.is_gpu_available:
            print('error gpu not available')
            exit()
        # x_val and y_val are only used to monitor the test loss and NOT for training

        mini_batch_size = int(min(x_train.shape[0] / 10, batch_size))

        start_time = time.time()

        hist = self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=nb_epochs,
                              verbose=self.verbose, validation_data=(x_val, y_val), callbacks=self.callbacks)

        duration = time.time() - start_time

        self.model.save(self.output_directory + '/last_model.hdf5')

        keras.backend.clear_session()
        return self.model

    def predict(self, x_test, y_true, enc):
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
