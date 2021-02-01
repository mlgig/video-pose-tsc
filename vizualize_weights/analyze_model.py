import os
import sys
import argparse
import logging

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

sns.set_style("dark")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnalyzeModel:
    def __init__(self, model, data_x, data_y, data_pid, enc=None):
        self.model = model
        self.enc = enc
        self.classes_list = np.unique(data_y)
        self.predictions = self.model.predict(data_x)
        try:
            y_pred = np.argmax(self.predictions, axis=1)
            self.y_pred_label = np.array([self.enc.categories_[0][i] for i in y_pred])
        except Exception as e:
            self.y_pred_label = self.predictions

        self.pid_info = np.zeros((data_pid.shape[0], data_pid.shape[1] + 1), dtype="<U31")
        self.pid_info[:, :-1] = data_pid
        self.pid_info[:, -1] = data_y == self.y_pred_label
        logger.info("Accuracy of the model is: {}".format(metrics.accuracy_score(data_y, self.y_pred_label)))

    def get_indices_data(self, data_y):
        class_instances_ind = {}
        wrong_indices = np.flatnonzero(data_y != self.y_pred_label)
        for class_name in self.classes_list:
            if class_name not in class_instances_ind:
                class_instances_ind[class_name] = {}
            total_indices = np.nonzero(np.isin(data_y, [class_name]))[0]
            class_instances_ind[class_name]["total_indices"] = total_indices
            incorrect_indices = np.intersect1d(total_indices, wrong_indices)
            class_instances_ind[class_name]["incorrect_indices"] = incorrect_indices
            correct_indices = list(set(total_indices) - set(incorrect_indices))
            class_instances_ind[class_name]["correct_indices"] = correct_indices
        return class_instances_ind

    def plot_prediction_probs(self, class_instances_ind, figsize=(15, 10)):
        plt.figure(figsize=figsize)
        for i, class_name in enumerate(self.classes_list):
            ax = plt.subplot(2, 2, i + 1)
            ax.set_title("Class: {}".format(class_name))
            probs = self.predictions[class_instances_ind[class_name]["total_indices"]][:, i]
            plt.hist(probs, bins=20)
        plt.savefig("/tmp/testing1.jpg")
        plt.close()
