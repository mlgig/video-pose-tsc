import logging
from functools import reduce

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def report_average(*args):
    initialize_result = False
    total_len = len(args)
    accuracy_avg = 0.0
    for report in args:
        splited = [' '.join(x.split()) for x in report.split('\n\n')]
        accuracy_avg += float(splited[2].split()[1])
        header = [x for x in splited[0].split(' ')]
        data = np.array(splited[1].split(' ')).reshape(-1, len(header) + 1)
        labels = data[:, 0]
        data = np.delete(data, 0, 1).astype(float)
        if not initialize_result:
            result_avg = np.zeros(data.shape, dtype=float)
            initialize_result = True
        result_avg += data

    result_avg = result_avg/total_len
    accuracy_avg = accuracy_avg/total_len
    temp = "precision    recall  f1-score   support\n\n"
    result_avg = np.round(result_avg, 2)
    logger.info("Average classification report")
    logger.info(result_avg)
    logger.info("Average accuracy")
    logger.info(round(accuracy_avg, 2))


def plot_confusion_matrix(output_path, seed_value, confusion_matrix, labels):
    # confusion_matrix = confusion_matrix / np.sum(confusion_matrix)
    cm_df = pd.DataFrame(confusion_matrix, index=labels, columns=labels)
    sns.heatmap(cm_df, cmap='Oranges', annot=True, fmt="d")  # fmt='.2%' or 'd'
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('{}/output_{}.png'.format(output_path, seed_value), bbox_inches='tight')
    plt.close()


# from sklearn.metrics import classification_report
# y1_predict = [0, 1, 1, 0]
# y1_dev = [0, 1, 1, 0]
# report_1 = classification_report(y1_dev, y1_predict)
# y2_predict = [1, 0, 1, 0]
# y2_dev = [1, 1, 0, 0]
# report_2 = classification_report(y2_dev, y2_predict)
#
# report_average(*[report_1, report_2])
