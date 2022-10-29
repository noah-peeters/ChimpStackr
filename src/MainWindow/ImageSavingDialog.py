"""
Create dialog(s) for changing image export settings; and success/error message.
"""
import os
import cv2
import numpy as np
import PySide6.QtCore as qtc
import PySide6.QtWidgets as qtw


# Quality selection dialog (depends on type of exported img)
class SelectQualityDialog(qtw.QDialog):
    selectedQuality = None

    def __init__(self, imType):
        super().__init__()
        self.setWindowTitle("Export image as " + imType)

        self.slider = qtw.QSlider(qtc.Qt.Orientation.Horizontal)
        self.slider.setSingleStep(1)
        self.slider.valueChanged.connect(self.value_changed)

        self.spinBox = qtw.QSpinBox()
        self.spinBox.valueChanged.connect(self.value_changed)

        horizontalLayout = qtw.QHBoxLayout()
        if imType == "JPG":
            horizontalLayout.addWidget(qtw.QLabel("Quality level:"))
            self.setup_slider(0, 100, 95)
        elif imType == "PNG":
            horizontalLayout.addWidget(qtw.QLabel("Compression level:"))
            self.setup_slider(0, 9, 4)

        horizontalLayout.addWidget(self.slider)
        horizontalLayout.addWidget(self.spinBox)

        verticalLayout = qtw.QVBoxLayout()
        verticalLayout.addLayout(horizontalLayout)

        buttonBox = qtw.QDialogButtonBox(self)
        buttonBox.addButton("Cancel", qtw.QDialogButtonBox.RejectRole)
        buttonBox.addButton("Apply", qtw.QDialogButtonBox.AcceptRole)
        verticalLayout.addWidget(buttonBox)
        buttonBox.rejected.connect(self.close)
        buttonBox.accepted.connect(self.apply_settings)

        self.setLayout(verticalLayout)

    def setup_slider(self, low, high, val):
        """
        Shorthand for QSlider/QSpinBox setup.
        """
        self.slider.setMinimum(low)
        self.slider.setMaximum(high)
        self.slider.setValue(val)

        self.spinBox.setMinimum(low)
        self.spinBox.setMaximum(high)
        self.spinBox.setValue(val)

    def apply_settings(self):
        """
        Apply current settings and close dialog.
        """
        self.selectedQuality = self.slider.value()
        self.close()

    def value_changed(self, newValue):
        """
        Value of slider or spinbox changed; update them both.
        """
        if self.slider.value() != newValue:
            self.slider.setValue(newValue)
        elif self.spinBox.value() != newValue:
            self.spinBox.setValue(newValue)


# Result dialog on image saved
class ResultDialog(qtw.QMessageBox):
    def __init__(self, imgPath=None, errorStackTrace=None):
        super().__init__()

        if imgPath == None:
            # Error msg
            self.setStandardButtons(qtw.QMessageBox.Ok)
            self.setIcon(qtw.QMessageBox.Critical)
            self.setWindowTitle("Export failed")
            self.setText("Failed to export!\n")
            if errorStackTrace != None:
                self.setInformativeText("Error stack trace:")
                self.setDetailedText(str(errorStackTrace))

        elif imgPath != None and errorStackTrace == None:
            # Success msg
            self.setStandardButtons(qtw.QMessageBox.Ok)
            self.setIcon(qtw.QMessageBox.Information)
            self.setWindowTitle("Export success")
            self.setText(
                "Successfully exported output image.\n"
                + "File size is: "
                + str(round(os.path.getsize(imgPath) / 1024 / 1024, 2))
                + "MB.\n"
            )
            self.setInformativeText("File location:\n" + imgPath)


def createDialog(imageArray, imType, chosenPath):
    # Something went wrong
    if imType == None:
        return

    imgPath = None
    errorStackTrace = None

    compressionFactor = None
    if imType == "JPG" or imType == "PNG":
        qualityDialog = SelectQualityDialog(imType)
        qualityDialog.exec()
        if qualityDialog.selectedQuality != None:
            compressionFactor = qualityDialog.selectedQuality
        else:
            return

    # Convert float32 image to uint8
    imageArray = np.around(imageArray)
    imageArray[imageArray > 255] = 255
    imageArray[imageArray < 0] = 0
    imageArray = imageArray.astype(np.uint8)

    # Get compression/quality for JPG and PNG (don't compress tiff)
    compression_list = None
    if imType == "JPG":
        # 0->100 quality (JPG)
        compression_list = [cv2.IMWRITE_JPEG_QUALITY, compressionFactor]
    elif imType == "PNG":
        # 9->0 compression (PNG)
        compression_list = [cv2.IMWRITE_PNG_COMPRESSION, compressionFactor]

    # Try saving image to disk
    try:
        cv2.imwrite(chosenPath, imageArray, compression_list)
        imgPath = chosenPath
    except Exception as e:
        errorStackTrace = e

    ResultDialog(imgPath, errorStackTrace).exec()
