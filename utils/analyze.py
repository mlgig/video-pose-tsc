import os
import sys
from shutil import copyfile

from data_processing.create_train_test_data import VALID_CLASSES
from utils.util_functions import create_directory_if_not_exists


def move_filtered_files():
    extracted_files_list = os.listdir(full_extracted_clips_path)
    coordinates_files_list = [f for f in extracted_files_list if not f.startswith(".")]
    filtered_coordinates_files_list = []
    for f in coordinates_files_list:
        exercise_type = f.split("_")[3].strip()
        if exercise_type in VALID_CLASSES:
            filtered_coordinates_files_list.append(f)
            copyfile(full_extracted_clips_path + "/" + f, filtered_clips_path + "/" + f)


if __name__ == "__main__":
    base_path = sys.argv[1]
    exercise = sys.argv[2]
    extract_clips_dir = "ExtractedClips"
    full_extracted_clips_path = os.path.join(base_path, extract_clips_dir, exercise)

    filtered_dir = "FilteredClips"
    filtered_clips_path = os.path.join(base_path, filtered_dir, exercise)

    create_directory_if_not_exists(filtered_clips_path)
    move_filtered_files()
