"""
Class that connects to QtSignals. Keeps track of timing of different parts of stacking.
Data can be retrieved once finished, to display in a graph.
"""

# TODO: Implement
class Logger:
    time_stamps = {}

    def __init__(self, signals):
        self.signals = signals
        signals.finished.connect(self.finished)
        signals.progress_update.connect(self.progress_update)
        # signals.status_update.connect(self.status_update)

    # Handle final data
    def finished(self):
        print()

    # Handle previous process data
    def progress_update(self, current_progress_percentage):
        print(current_progress_percentage)

    # def status_update(self, status_text):
    #     print()