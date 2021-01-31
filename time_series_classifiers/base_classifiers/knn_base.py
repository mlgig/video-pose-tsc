import argparse
import configparser
import os
import getpass
from pathlib import Path
import logging

from configobj import ConfigObj
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
from tslearn.utils import load_time_series_txt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn import metrics
import pandas as pd

from time_series_classifiers import TRAIN_DATASET_X, TEST_DATASET_X, TRAIN_DATASET_Y, TEST_DATASET_Y
from utils.math_funtions import get_combinations
from utils.program_stats import timeit
from utils.sklearn_utils import report_average
from utils.util_functions import create_directory_if_not_exists

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def read_dataset(path):
    X_train = load_time_series_txt(os.path.join(tslearn_format_path, "{}.txt".format(TRAIN_DATASET_X)))
    y_train = np.load(os.path.join(tslearn_format_path, "{}.npy".format(TRAIN_DATASET_Y)))
    X_test = load_time_series_txt(os.path.join(tslearn_format_path, "{}.txt".format(TEST_DATASET_X)))
    y_test = np.load(os.path.join(tslearn_format_path, "{}.npy".format(TEST_DATASET_Y)))
    return X_train, y_train, X_test, y_test


@timeit
def fit_kneighbors(k=1):
    knn = KNeighborsTimeSeriesClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    logger.info("Accuracy is: {}".format(accuracy))
    confusion_matrix = metrics.confusion_matrix(y_test, predictions)
    classification_report = metrics.classification_report(y_test, predictions)
    logger.info("Confusion Matrix: {}\n".format(confusion_matrix))
    logger.info("Classification report: {}\n".format(classification_report))
    classification_report_list.append(classification_report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--knn_config", required=True, help="path of the config file")
    parser.add_argument("--exercise_config", required=True, help="path of the config file")
    args = parser.parse_args()
    knn_config = ConfigObj(args.knn_config)

    home_path = str(Path.home())
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    input_data_path = os.path.join(base_path, knn_config["INPUT_DATA_PATH"])

    exercise = knn_config["EXERCISE"]
    seed_values = knn_config["SEED_VALUES"]
    multiclass_classification = knn_config.as_bool("MULTICLASS_CLASSIFICATION")
    multiclass_dir = knn_config["MULTICLASS_DIR"]
    neighbors = knn_config.as_int("NEIGHBORS")
    output_results = os.path.join(base_path, knn_config["OUTPUT_RESULTS"])

    config_parser = configparser.RawConfigParser()
    config_parser.read(args.exercise_config)
    valid_classes = config_parser.get(exercise, "valid_classes").split(",")
    label_index_mapping = {i + 1: value for i, value in enumerate(valid_classes)}
    index_label_mapping = {value: i + 1 for i, value in enumerate(valid_classes)}

    classification_report_list = []
    for seed_value in seed_values:
        logger.info("----------------------------------------------------")
        logger.info("Fitting KNN for seed value: {}".format(seed_value))
        tslearn_format_path = os.path.join(input_data_path, exercise, seed_value, multiclass_dir)
        X_train, y_train, X_test, y_test = read_dataset(tslearn_format_path)
        fit_kneighbors(neighbors)
        logger.info("----------------------------------------------------")

    output_results_directory = os.path.join(output_results, "KNN")
    create_directory_if_not_exists(output_results_directory)

    logger.info("Average classification report")
    logger.info(report_average(*classification_report_list))
