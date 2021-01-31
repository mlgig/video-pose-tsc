import argparse
import configparser
import os
from pathlib import Path
import logging
import numpy as np
import getpass

from configobj import ConfigObj
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf

from time_series_classifiers import TRAIN_DATASET_Y, TRAIN_DATASET_X, TEST_DATASET_X, TEST_DATASET_Y
from utils.math_funtions import get_combinations
from utils.sklearn_utils import report_average, plot_confusion_matrix
from utils.util_functions import create_directory_if_not_exists

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
DL_METHODS = ["fcn", "cnn", "resnet"]


def read_dataset(data_path, dataset_name):
    datasets_dict = {}
    x_train = np.load(os.path.join(data_path, "{}.npy".format(TRAIN_DATASET_X)), allow_pickle=True)
    y_train = np.load(os.path.join(data_path, "{}.npy".format(TRAIN_DATASET_Y)), allow_pickle=True)
    x_test = np.load(os.path.join(data_path, "{}.npy".format(TEST_DATASET_X)), allow_pickle=True)
    y_test = np.load(os.path.join(data_path, "{}.npy".format(TEST_DATASET_Y)), allow_pickle=True)
    datasets_dict[dataset_name] = (x_train.copy(), y_train.copy(), x_test.copy(), y_test.copy())
    logger.info("Data shape is: ")
    logger.info("{} {} {} {}".format(x_train.shape, y_train.shape, x_test.shape, y_test.shape))
    return datasets_dict


def fit_classifier(classifier_name, datasets_dict, dataset_name):
    x_train = datasets_dict[dataset_name][0]
    y_train = datasets_dict[dataset_name][1]
    y_train_orig = y_train
    x_test = datasets_dict[dataset_name][2]
    y_test = datasets_dict[dataset_name][3]

    nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))

    # transform the labels from integers to one hot vectors
    enc = OneHotEncoder(categories='auto')
    enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
    y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
    y_test = enc.transform(y_test.reshape(-1, 1)).toarray()

    # save orignal y because later we will use binary
    y_true = np.argmax(y_test, axis=1)

    if len(x_train.shape) == 2:  # if univariate
        # add a dimension to make it multivariate with one dimension
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    input_shape = x_train.shape[1:]
    classifier = create_classifier(classifier_name, input_shape, nb_classes, weights_directory)
    # with tf.device('/gpu:1'):
    classifier.fit(x_train, y_train, x_test, y_test, y_true, nb_epochs)
    # vizualize_weights(model, enc, x_train, y_train_orig, y_test)
    confusion_matrix, classification_report, df_metrics = classifier.predict(x_test, y_true, enc)

    classification_report_list.append(classification_report)
    labels = list(enc.categories_[0])

    plot_confusion_matrix(weights_directory, seed_value, confusion_matrix, labels)

    # output_results_path = os.path.join(output_path, "DeepLearning")
    # create_directory_if_not_exists(output_results_path)
    # df_metrics.to_csv(output_results_path + "/{}_{}.csv".format(exercise, classifier_name), index=False)
    print(df_metrics)


def create_classifier(classifier_name, input_shape, nb_classes, output_directory, verbose=False):
    if classifier_name == 'fcn':
        from time_series_classifiers.deep_learning_classifiers.classifiers import fcn
        return fcn.Classifier_FCN(output_directory, exercise, input_shape, nb_classes, verbose)
    if classifier_name == 'resnet':
        from time_series_classifiers.deep_learning_classifiers.classifiers import resnet
        return resnet.Classifier_RESNET(output_directory, exercise, input_shape, nb_classes, verbose)
    if classifier_name == 'cnn':
        from time_series_classifiers.deep_learning_classifiers.classifiers import cnn
        return cnn.Classifier_CNN(output_directory, exercise, input_shape, nb_classes, verbose)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dl_config", required=True, help="path of the config file")
    parser.add_argument("--exercise_config", required=True, help="path of the config file")
    args = parser.parse_args()
    dl_config = ConfigObj(args.dl_config)

    home_path = str(Path.home())
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    input_data_path = os.path.join(base_path, dl_config["INPUT_DATA_PATH"])

    exercise = dl_config["EXERCISE"]
    classifier_type = dl_config["CLASSIFIER_TYPE"]
    nb_epochs = dl_config.as_int("EPOCHS")
    multiclass_dir = dl_config["MULTICLASS_DIR"]
    output_path = os.path.join(base_path, dl_config["OUTPUT_PATH"])
    local_prefix = dl_config["LOCAL_PREFIX"]

    seed_values = dl_config.as_list("SEED_VALUES")

    config_parser = configparser.RawConfigParser()
    config_parser.read(args.exercise_config)
    valid_classes = config_parser.get(exercise, "valid_classes").split(",")
    label_index_mapping = {i + 1: value for i, value in enumerate(valid_classes)}
    index_label_mapping = {value: i + 1 for i, value in enumerate(valid_classes)}

    classification_report_list = []
    logger.info("Fitting classifier: {} : ".format(classifier_type))
    for seed_value in seed_values:
        logger.info("Seed value: {}".format(seed_value))
        data_path = os.path.join(input_data_path, exercise, seed_value, multiclass_dir)
        weights_directory = os.path.join(output_path, 'dl_weights', exercise, classifier_type, multiclass_dir,
                                         seed_value)
        create_directory_if_not_exists(weights_directory)
        datasets_dict = read_dataset(data_path, multiclass_dir)
        fit_classifier(classifier_type, datasets_dict, multiclass_dir)
        tf.keras.backend.clear_session()
        logger.info('DONE')

    logger.info("Average classification report")
    logger.info(report_average(*classification_report_list))
