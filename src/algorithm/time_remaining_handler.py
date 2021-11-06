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
# TODO: Cleanup inside class + keep average of times (can be reset on ask)
import time


time_spent_percentages = {
    "laplacian_generation": 40,
    "pyramid_focus_fusion": 60,
}
# Calculate progressbar value (range: [0, 100]) from current operation percentage
def calculate_progressbar_value(operation_name, percentage_finished):
    calc = percentage_finished * time_spent_percentages[operation_name] / 100
    if operation_name == "pyramid_focus_fusion":
        # Return sum of previous
        return time_spent_percentages["laplacian_generation"] + calc
    else:
        return calc


# Return remaining time of algorithm (hh:mm:ss)
def calculate_time_remaining(
    operation_name, percentage_increment, percentage_left, time_taken
):
    # Time to 100% completion of current operation
    time_left = percentage_left / percentage_increment * time_taken

    if operation_name == "laplacian_generation":
        # Add in approx. time of focus fusion
        multiplier = (
            time_spent_percentages["pyramid_focus_fusion"]
            / time_spent_percentages["laplacian_generation"]
        )
        time_left += time_taken * multiplier

    formatted = time.strftime("%H:%M:%S", time.gmtime(time_left))
    return "Time left until program finish: " + formatted
