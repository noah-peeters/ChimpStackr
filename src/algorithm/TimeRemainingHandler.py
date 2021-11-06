"""
    Utility functions for computing the remaining time until program finishes.
    Uses average percentage of time spent in main parts of the algorithm.

    Main parts are: 
        * Laplacian pyramid generation
        * Laplacian pyramid focus fusion
    Other insignificant parts (only around 0.1% of time spent here):
        * Laplacian pyramid collapse/reconstruction of image
        * Post processing
    
    Quick note: when re-stacking the same images; laplacian pyramids are kept on disk.
    They don't need to be recalculated, making the algorithm faster.
"""
import time, statistics


time_spent_percentages = {
    "laplacian_generation": 40,
    "pyramid_focus_fusion": 60,
}


class TimeRemainingHandler:
    def __init__(self):
        self.percentage_increment = 0  # Increment of percentage change per call
        self.cached_time_taken = []  # List of cached time taken

    # Calculate progressbar value (range: [0, 100]) from current operation percentage
    def calculate_progressbar_value(self, operation_name, percentage_finished):
        calc = percentage_finished * time_spent_percentages[operation_name] / 100
        if operation_name == "pyramid_focus_fusion":
            # Return sum of previous
            return time_spent_percentages["laplacian_generation"] + calc
        else:
            return calc

    # Return remaining time of algorithm (hh:mm:ss)
    def calculate_time_remaining(self, operation_name, percentage_left, time_taken):
        self.cached_time_taken.append(time_taken)
        mean_time_taken = statistics.mean(self.cached_time_taken)

        # Time left to 100% completion of current operation
        time_left = percentage_left / self.percentage_increment * mean_time_taken

        if operation_name == "laplacian_generation":
            # Add in approx. time of focus fusion (100% completion)
            multiplier = (
                time_spent_percentages["pyramid_focus_fusion"]
                / time_spent_percentages["laplacian_generation"]
            )
            # Multiply with needed percentage
            time_left *= multiplier ** -1
            # Add percentage of other task
            time_left += 100 / self.percentage_increment * mean_time_taken * multiplier

        formatted = time.strftime("%H:%M:%S", time.gmtime(time_left))
        return "Time left until program finish: " + formatted

    # Remove cached variables
    def clear_cache(self):
        self.percentage_increment = 0
        self.cached_time_taken = []
