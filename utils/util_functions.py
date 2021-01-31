import os
import shutil
import sys
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_directory_if_not_exists(path):
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except Exception as e:
        logger.info("Error creating the directory: {} {}".format(path, str(e)))


def delete_directory_if_exists(path):
    if os.path.exists(path) and os.path.isdir(path):
        shutil.rmtree(path)
