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