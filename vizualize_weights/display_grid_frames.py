import os
import sys
import argparse
import logging
import math

import seaborn as sns
from matplotlib.image import imread
import matplotlib.pyplot as plt

sns.set_style("dark")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_frames_path(index_of_sample, data_pid, extracted_frames_path):
    file_name = data_pid[index_of_sample][0]
    pid = file_name.split(" ")[0]
    exercise_type = file_name.split("_")[-2]

    start_frame = data_pid[index_of_sample][1]
    end_frame = data_pid[index_of_sample][2]

    frames_path = os.path.join(extracted_frames_path, pid, exercise_type, str(start_frame) + "_" + str(end_frame))
    if not os.path.exists(frames_path):
        frames_path = os.path.join(extracted_frames_path, pid, exercise_type,
                                   str(start_frame) + "_" + str(int(end_frame) + 1))
    if not os.path.exists(frames_path):
        frames_path = os.path.join(extracted_frames_path, pid, exercise_type,
                                   str(int(start_frame) - 1) + "_" + str(int(end_frame)))
    logger.info("Frames path: {}".format(frames_path))
    return frames_path


def display_grid_frames(index_of_sample, data_pid, frames_output, extracted_frames_path, step=3):
    file_name = data_pid[index_of_sample][0]
    pid = file_name.split(" ")[0]
    exercise_type = file_name.split("_")[-2]

    start_frame = data_pid[index_of_sample][1]
    end_frame = data_pid[index_of_sample][2]

    frames_path = os.path.join(extracted_frames_path, pid, exercise_type, str(start_frame) + "_" + str(end_frame))
    if not os.path.exists(frames_path):
        frames_path = os.path.join(extracted_frames_path, pid, exercise_type,
                                   str(start_frame) + "_" + str(int(end_frame) + 1))
    if not os.path.exists(frames_path):
        frames_path = os.path.join(extracted_frames_path, pid, exercise_type,
                                   str(int(start_frame) - 1) + "_" + str(int(end_frame)))
    logger.info("Frames path: {}".format(frames_path))
    if not os.path.exists(frames_path):
        return None, None

    total_images = [frames_output[i] - frames_output[i - 1] + 1 for i in range(len(frames_output) - 1, 0, -2)]
    total_images = total_images[::-1]

    logger.info("Total images are: {} {}".format(sum(total_images), total_images))
    list_of_images = []
    list_of_titles = []
    count = 0
    region = 0
    for i in range(len(frames_output) - 1, 0, -2):
        starting_frame = frames_output[i - 1]
        ending_frame = frames_output[i]
        for frame in range(starting_frame, ending_frame, step):
            frame_path = os.path.join(frames_path, "frame_{}.jpg".format(frame))
            try:
                image = imread(frame_path)
            except FileNotFoundError:
                break
            list_of_images.append(image)
            count = count + 1
            list_of_titles.append("Region {}".format(region))
        region = region + 1
    return list_of_images, list_of_titles


def plot_grid(list_of_images, list_of_titles, actual_label, predicted_label, predicted_prob, file_name, output_path, figsize=(15, 15)):
    fig = plt.figure(figsize=figsize)
    n_cols = 4
    n_rows = math.ceil(len(list_of_images) / n_cols)
    logger.info("Total images are: {}".format(len(list_of_images)))
    logger.info("Rows: {} Cols: {}".format(n_rows, n_cols))
    for i in range(n_rows):
        for j in range(n_cols):
            if i * n_cols + j >= len(list_of_images):
                break
            ax1 = fig.add_subplot(n_rows, n_cols, i * n_cols + j + 1)
            ax1.set_title(list_of_titles[i * n_cols + j], fontsize=20)
            ax1.imshow(list_of_images[i * n_cols + j])
            ax1.axis("off")
    st = plt.suptitle(
        'True label:' + actual_label + ', Predicted label:' + predicted_label + ',  Likelihood of ' + predicted_label + ': ' + str(
            predicted_prob), fontsize=30)
    st.set_y(0.95)
    fig.subplots_adjust(top=0.85)
    plt.savefig("{}/grid_{}.png".format(output_path, file_name, dpi=300))
    # plt.savefig("{}/grid_{}.eps".format(output_path, file_name), format='eps')
    # fig.savefig("{}/grid_{}.svg".format(output_path, file_name), format='svg', dpi=1200)
    plt.close(fig)


def plot_frames(top_idx, top_values, frames_path, actual_label, predicted_label, predicted_prob, file_name, output_path, figsize=(15, 15)):
    fig = plt.figure(figsize=figsize)
    n_cols = 4
    n_rows = math.ceil(len(top_idx) / n_cols)
    logger.info("Rows: {} Cols: {}".format(n_rows, n_cols))

    for i in range(n_rows):
        for j in range(n_cols):
            if i * n_cols + j >= len(top_idx):
                continue
            ax1 = fig.add_subplot(n_rows, n_cols, i * n_cols + j + 1)
            frame_path = os.path.join(frames_path, "frame_{}.jpg".format(top_idx[i * n_cols + j]))
            image = imread(frame_path)
            ax1.imshow(image)
            ax1.axis("off")
            ax1.set_title("Index: " + str(top_idx[i * n_cols + j]) + " Weight: " + str(round(top_values[i * n_cols + j], 2)))
    st = plt.suptitle(
        'True label:' + actual_label + ', Predicted label:' + predicted_label + ',  Likelihood of ' + predicted_label + ': ' + str(
            predicted_prob), fontsize=30)
    st.set_y(0.95)
    fig.subplots_adjust(top=0.85)
    plt.savefig("{}/grid_{}.png".format(output_path, file_name, dpi=300))
    plt.close(fig)
