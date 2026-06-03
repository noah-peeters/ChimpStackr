"""
    Utility functions for UI/algorithm.
"""
import re

# Sort strings numerically; src: https://stackoverflow.com/questions/3426108/how-to-sort-a-list-of-strings-numerically
# Correctly handles: 11, 1, 2 --> 1, 2, 11
def int_string_sorting(text):
    def atof(text):
        try:
            retval = float(text)
        except ValueError:
            retval = text
        return retval

    return [atof(c) for c in re.split(r"[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)", text)]


def merge_image_paths(existing, new):
    """Combine two path lists, drop duplicates (keep first), numeric-aware sort.

    Used to append camera-captured frames to the already-loaded image list.
    """
    seen = set()
    combined = []
    for path in list(existing) + list(new):
        if path not in seen:
            seen.add(path)
            combined.append(path)
    return sorted(combined, key=int_string_sorting)