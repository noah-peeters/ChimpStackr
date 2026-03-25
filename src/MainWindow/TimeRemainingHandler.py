"""
    Time remaining estimator.
    Simple and reliable: tracks time per image, predicts from rolling average.
"""
import time
from collections import deque


class TimeRemainingHandler:
    def __init__(self, window_size=10):
        self._times = deque(maxlen=window_size)

    def reset(self):
        self._times.clear()

    def calculate_time_remaining(self, percentage_increment, percentage_left, time_taken):
        """
        Args:
            percentage_increment: percentage each image represents (100/total)
            percentage_left: percentage remaining to 100%
            time_taken: seconds this image took
        Returns:
            Formatted string like "Time left: ~1m 23s"
        """
        self._times.append(time_taken)

        # Skip first sample (often inflated by JIT warmup / cold cache)
        if len(self._times) <= 1:
            return "Time left: estimating..."

        # Use median of recent times (robust to outliers)
        sorted_times = sorted(self._times)
        median = sorted_times[len(sorted_times) // 2]

        # How many images remain
        if percentage_increment > 0:
            images_left = percentage_left / percentage_increment
        else:
            images_left = 0

        time_left = max(0, images_left * median)

        if time_left < 2:
            return "Time left: finishing..."
        elif time_left < 60:
            return f"Time left: ~{int(time_left)}s"
        elif time_left < 3600:
            mins = int(time_left // 60)
            secs = int(time_left % 60)
            return f"Time left: ~{mins}m {secs:02d}s"
        else:
            hrs = int(time_left // 3600)
            mins = int((time_left % 3600) // 60)
            return f"Time left: ~{hrs}h {mins:02d}m"
