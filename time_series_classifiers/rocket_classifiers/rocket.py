import argparse
import configparser
import os
import logging
from pathlib import Path

from configobj import ConfigObj
from sklearn import metrics
import numpy as np
from sktime.transformers.series_as_features.rocket import Rocket
from sklearn.linear_model import RidgeClassifierCV
from sktime.utils.load_data import load_from_tsfile_to_dataframe
import sktime

from time_series_classifiers import TRAIN_DATASET_X, TEST_DATASET_X, VAL_DATASET_X
from utils.math_funtions import get_combinations
from utils.program_stats import timeit
from utils.sklearn_utils import report_average, plot_confusion_matrix
from utils.util_functions import create_directory_if_not_exists

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def read_dataset(path):
    x_train, y_train = load_from_tsfile_to_dataframe(os.path.join(path,
                                                                  "{}.ts".format(TRAIN_DATASET_X)))
    logger.info("Training data shape {} {} {}".format(x_train.shape, len(x_train.iloc[0, 0]), y_train.shape))

    x_test, y_test = load_from_tsfile_to_dataframe(os.path.join(path,
                                                                "{}.ts".format(TEST_DATASET_X)))
    logger.info("Testing data shape: {} {}".format(x_test.shape, y_test.shape))

    try:
        x_val, y_val = load_from_tsfile_to_dataframe(os.path.join(input_data_path,
                                                                  "{}.ts".format(VAL_DATASET_X)))
        logger.info("Validation data shape: {} {}".format(x_val.shape, y_val.shape))
    except (sktime.utils.load_data.TsFileParseException, FileNotFoundError):
        logger.info("Validation data is empty:")
        x_val, y_val = None, None

    return x_train, y_train, x_test, y_test, x_val, y_val


class RocketTransformerClassifier:
    def __init__(self, exercise):
        self.exercise = exercise
        self.classifiers_mapping = {}

    @timeit
    def fit_rocket(self, x_train, y_train, kernels=10000):
        rocket = Rocket(num_kernels=kernels, normalise=False)
        rocket.fit(x_train)
        x_training_transform = rocket.transform(x_train)
        self.classifiers_mapping["transformer"] = rocket
        classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), normalize=True)
        classifier.fit(x_training_transform, y_train)
        self.classifiers_mapping["classifier"] = classifier

    @timeit
    def predict_rocket(self, x_test, y_test, x_val=None, y_val=None):
        rocket = self.classifiers_mapping["transformer"]
        classifier = self.classifiers_mapping["classifier"]
        x_test_transform = rocket.transform(x_test)
        predictions = classifier.predict(x_test_transform)
        labels = list(np.sort(np.unique(y_test)))
        confusion_matrix = metrics.confusion_matrix(y_test, predictions)
        classification_report = metrics.classification_report(y_test, predictions)
        logger.info("-----------------------------------------------")
        logger.info("Metrics on testing data")
        logger.info("Accuracy {}".format(metrics.accuracy_score(y_test, predictions)))
        logger.info("\n Confusion Matrix: \n {}".format(confusion_matrix))
        logger.info("\n Classification report: \n{}".format(classification_report))

        classification_report_list.append(classification_report)

        plot_confusion_matrix(output_results_path, seed_value, confusion_matrix, labels)

        if x_val:
            logger.info("-----------------------------------------------")
            logger.info("Metrics on validation data")
            x_val_transform = rocket.transform(x_val)
            predictions = classifier.predict(x_val_transform)
            confusion_matrix = metrics.confusion_matrix(y_val, predictions)
            classification_report = metrics.classification_report(y_val, predictions)
            logger.info("Accuracy {}".format(metrics.accuracy_score(y_test, predictions)))
            logger.info("\n Confusion Matrix: \n {}".format(confusion_matrix))
            logger.info("\n Classification report: \n{}".format(classification_report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rocket_config", required=True, help="path of the config file")
    parser.add_argument("--exercise_config", required=True, help="path of the config file")
    args = parser.parse_args()
    rocket_config = ConfigObj(args.rocket_config)

    home_path = str(Path.home())
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    seed_values = rocket_config["SEED_VALUES"]
    input_data_path = os.path.join(base_path, rocket_config["INPUT_DATA_PATH"])
    exercise = rocket_config["EXERCISE"]
    multiclass_classification = rocket_config.as_bool("MULTICLASS_CLASSIFICATION")
    multiclass_dir = rocket_config["MULTICLASS_DIR"]
    output_path = os.path.join(base_path, rocket_config["OUTPUT_PATH"])

    config_parser = configparser.RawConfigParser()
    config_parser.read(args.exercise_config)
    valid_classes = config_parser.get(exercise, "valid_classes").split(",")
    class_combination = get_combinations(valid_classes, 2)
    label_index_mapping = {i + 1: value for i, value in enumerate(valid_classes)}
    index_label_mapping = {value: i + 1 for i, value in enumerate(valid_classes)}

    output_results_path = os.path.join(output_path, "Rocket")
    create_directory_if_not_exists(output_results_path)

    classification_report_list = []
    for seed_value in seed_values:
        logger.info("----------------------------------------------------")
        logger.info("Fitting Rocket for seed value: {}".format(seed_value))
        input_path_combined = os.path.join(input_data_path, exercise, seed_value, multiclass_dir)
        if not os.path.exists(input_path_combined):
            logger.info("Path does not exist for seed: {}".format(seed_value))
            continue
        x_train, y_train, x_test, y_test, x_val, y_val = read_dataset(input_path_combined)

        rocket_classifier = RocketTransformerClassifier(exercise)
        rocket_classifier.fit_rocket(x_train, y_train)
        rocket_classifier.predict_rocket(x_test, y_test, x_val, y_val)

    logger.info("Average classification report")
    logger.info(report_average(*classification_report_list))
