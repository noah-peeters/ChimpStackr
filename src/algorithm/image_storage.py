"""
    Class handling Laplacian Pyramid writes/retrievals to/from disk.
    Used to prevent keeping pyramids in memory and overloading RAM.
"""

import os, tempfile

import numpy as np
from numba.typed import List
import cv2

import src.utilities as utilities


class ImageStorageHandler:
    def __init__(self):
        return

    # TODO: Don't pass root_dir as argument, but use global var
    # Write laplacian pyramid of image to archive on disk (as float16)
    def write_laplacian_pyramid_to_disk(self, laplacian_pyramid, root_dir):
        file_handle, tmp_file = tempfile.mkstemp(".npz", None, root_dir.name)

        # Append pyramid levels to dictionary
        dictionary = {}
        for i, pyramid_level in enumerate(laplacian_pyramid):
            dictionary["Laplacian_" + str(i)] = pyramid_level.astype(np.float16)

        # Write to uncompressed archive
        np.savez(tmp_file, **dictionary)

        # Ensure write to disk
        os.close(file_handle)

        return tmp_file

    # Load laplacian pyramid from archive on disk (as float32)
    def load_laplacian_pyramid(self, image_archive_filename):
        archive = np.load(image_archive_filename, allow_pickle=False)

        # Get laplacian pyramid (if there)
        laplacian_pyr_keys = sorted(archive.files, key=utilities.int_string_sorting)
        laplacian_pyr = List()
        for key in laplacian_pyr_keys:
            pyr_level = archive[key].astype(np.float32)
            laplacian_pyr.append(pyr_level)

        return laplacian_pyr
