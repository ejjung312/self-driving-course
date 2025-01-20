import os
import sys
import cv2
import numpy as np

def get_file_list_recursively(top_directory):
    """
    Get list of full paths of all files found under root directory "top_directory".
    If a list of allowed file extensions is provided, files are filtered according to this list.
    """

    if not os.path.exists(top_directory):
        raise ValueError('Directory "{}" does NOT exist.'.format(top_directory))

    file_list = []

    for cur_dir, cur_subdirs, cur_files, in os.walk(top_directory):
        for file in cur_files:
            file_list.append(os.path.join(cur_dir, file))
            sys.stdout.write('\r[{}] - found {:06d} files...'.format(top_directory, len(file_list)))
            sys.stdout.flush()

    sys.stdout.write(' Done.\n')

    return file_list