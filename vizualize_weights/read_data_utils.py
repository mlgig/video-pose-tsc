import os

import numpy as np
from sktime.utils.load_data import load_from_tsfile_to_dataframe

TRAIN_DATASET_X = "TRAIN_X"
TEST_DATASET_X = "TEST_X"
TRAIN_DATASET_Y = "TRAIN_Y"
TEST_DATASET_Y = "TEST_Y"
VAL_DATASET_Y = "VAL_Y"
VAL_DATASET_X = "VAL_X"
TRAIN_PID = "TRAIN_pid"
TEST_PID = "TEST_pid"


def read_datasets_numpy(input_path):
    x_train = np.load(os.path.join(input_path, "{}.npy".format(TRAIN_DATASET_X)), allow_pickle=True)
    y_train = np.load(os.path.join(input_path, "{}.npy".format(TRAIN_DATASET_Y)), allow_pickle=True)
    x_test = np.load(os.path.join(input_path, "{}.npy".format(TEST_DATASET_X)), allow_pickle=True)
    y_test = np.load(os.path.join(input_path, "{}.npy".format(TEST_DATASET_Y)), allow_pickle=True)
    train_pid = np.load(os.path.join(input_path, "{}.npy".format(TRAIN_PID)), allow_pickle=True)
    test_pid = np.load(os.path.join(input_path, "{}.npy".format(TEST_PID)), allow_pickle=True)

    return x_train, y_train, train_pid, x_test, y_test, test_pid


def read_datasets_sktime(input_path):
    x_train, y_train = load_from_tsfile_to_dataframe(os.path.join(input_path, "{}.ts".format(TRAIN_DATASET_X)))
    x_test, y_test = load_from_tsfile_to_dataframe(os.path.join(input_path, "{}.ts".format(TEST_DATASET_X)))
    train_pid = np.load(os.path.join(input_path, "{}.npy".format(TRAIN_PID)), allow_pickle=True)
    test_pid = np.load(os.path.join(input_path, "{}.npy".format(TEST_PID)), allow_pickle=True)

    return x_train, y_train, train_pid, x_test, y_test, test_pid


def get_custom_indices(pid, pid_info, display=False):
    mask = [True if i.split(" ")[0] == pid else False for i in pid_info[:, 0]]
    indices = [l for l, i in enumerate(pid_info[:, 0]) if i.split(" ")[0] == pid and pid_info[l, -1]]

    if display:
        with np.printoptions(threshold=np.inf):
            print(pid_info[mask])
    return indices
