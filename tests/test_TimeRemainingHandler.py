"""
Test TimeRemainingHandler module.
"""

# Hack to import modules
# src: https://codeolives.com/2020/01/10/python-reference-module-in-parent-directory/
import os, sys

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from src.MainWindow.TimeRemainingHandler import TimeRemainingHandler

time_handler = TimeRemainingHandler()


def test_values():
    # Only one needed (no cache kept)
    assert (
        time_handler.calculate_progressbar_value("laplacian_pyramid_generation", 73)
        == 32.85
    )

    # Multiple checks needed (b/c of cache)
    assert (
        time_handler.calculate_time_remaining("laplacian_pyramid_generation", 10, 80, 5)
        == "Time left until program finish: 00:01:33"
    )
    assert (
        time_handler.calculate_time_remaining("laplacian_pyramid_generation", 5, 70, 2)
        == "Time left until program finish: 00:02:05"
    )

    time_handler.clear_cache()

    assert (
        time_handler.calculate_time_remaining(
            "laplacian_pyramid_focus_fusion", 10, 80, 5
        )
        == "Time left until program finish: 00:00:40"
    )
    assert (
        time_handler.calculate_time_remaining(
            "laplacian_pyramid_focus_fusion", 5, 70, 2
        )
        == "Time left until program finish: 00:00:49"
    )
    time_handler.clear_cache()
