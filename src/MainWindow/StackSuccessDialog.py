"""
    Dialog displayed at end of stacking.
    Shows stats about the stack.
"""
import PySide6.QtWidgets as qtw


class StatsWindow(qtw.QDialog):
    def __init__(self):
        super().__init__()

        time_taken_label = qtw.QLabel("Operation took: HH:MM:SS")
        stat2 = qtw.QLabel("Stat 2 test label")

        v_layout = qtw.QVBoxLayout()
        v_layout.addWidget(time_taken_label)
        v_layout.addWidget(stat2)

        self.setLayout(v_layout)
        self.setWindowTitle("Stacking stats")

        self.setModal(True)
        self.exec()


class Message(qtw.QMessageBox):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Stack finished")
        self.setText("The stack has successfully finished\n")
        self.setInformativeText("This is informative text.")

        self.setIcon(qtw.QMessageBox.Information)

        self.setStandardButtons(qtw.QMessageBox.Ok)
        self.setDefaultButton(qtw.QMessageBox.Ok)
        # Setup "Statistics" button
        stats_button = self.addButton("Show statistics...", qtw.QMessageBox.ActionRole)

        self.exec()

        if self.clickedButton() == stats_button:
            # Display stats QDialog
            print("Display stats QDialog")
            StatsWindow()
