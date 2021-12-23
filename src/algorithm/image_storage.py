"""
    Class handling Laplacian Pyramid writes/retrievals to/from disk.
    Used to prevent keeping pyramids in memory and overloading RAM.
"""

import os, tempfile

import numpy as np
from numba.typed import List

import src.utilities as utilities


class ImageStorageHandler:
    def __init__(self):
        return

    # Write laplacian pyramid of image to archive on disk
    def write_laplacian_pyramid_to_disk(self, laplacian_pyramid, root_dir):
        file_handle, tmp_file = tempfile.mkstemp(".npz", None, root_dir.name)
        dictionary = {}

        # Append pyramid levels to dictionary
        for i, pyramid_level in enumerate(laplacian_pyramid):
            dictionary["Laplacian_" + str(i)] = pyramid_level

        # Write to uncompressed archive
        np.savez(tmp_file, **dictionary)

        os.close(file_handle)
        return tmp_file

    # Load laplacian pyramid from archive on disk
    def load_laplacian_pyramid(self, image_archive_filename):
        archive = np.load(image_archive_filename, mmap_mode="r")

        # Get laplacian pyramid (if there)
        laplacian_pyr_keys = sorted(archive.files, key=utilities.int_string_sorting)
        laplacian_pyr = List()
        for key in laplacian_pyr_keys:
            laplacian_pyr.append(archive[key])

        return laplacian_pyr
