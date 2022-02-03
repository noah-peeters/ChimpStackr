import os

import cv2
import numpy as np
import PySide6.QtCore as qtc
import PySide6.QtWidgets as qtw


# Quality selection dialog (depends on type of img)
class SelectQualityDialog(qtw.QDialog):
    selectedQuality = None

    def __init__(self, imType):
        super().__init__()
        # Success msg
        self.setWindowTitle("Export image as " + imType)

        self.slider = qtw.QSlider(qtc.Qt.Orientation.Horizontal)
        self.slider.setSingleStep(1)

        horizontalLayout = qtw.QHBoxLayout()
        if imType == "JPG":
            horizontalLayout.addWidget(qtw.QLabel("Quality"))
            # TODO: Display value next to slider + allow keyboard input like gimp??
            self.slider.setMinimum(0)
            self.slider.setMaximum(100)
            self.slider.setValue(95)
        elif imType == "PNG":
            horizontalLayout.addWidget(qtw.QLabel("Compression level"))
            self.slider.setMinimum(9)
            self.slider.setMaximum(0)
            self.slider.setValue(4)
        horizontalLayout.addWidget(self.slider)

        verticalLayout = qtw.QVBoxLayout()
        verticalLayout.addLayout(horizontalLayout)

        buttonBox = qtw.QDialogButtonBox(self)
        buttonBox.addButton("Cancel", qtw.QDialogButtonBox.RejectRole)
        buttonBox.addButton("Apply", qtw.QDialogButtonBox.AcceptRole)
        verticalLayout.addWidget(buttonBox)
        buttonBox.rejected.connect(self.close)
        buttonBox.accepted.connect(self.applied_settings)

        self.setLayout(verticalLayout)

    def applied_settings(self):
        print("SET QUALITY" + str(self.slider.value()))
        self.selectedQuality = self.slider.value()
        self.close()


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
                self.setInformativeText("Error stack trace:\n" + str(errorStackTrace))
        elif imgPath != None and errorStackTrace == None:
            # Success msg
            self.setStandardButtons(qtw.QMessageBox.Ok)
            self.setIcon(qtw.QMessageBox.Information)
            self.setWindowTitle("Export success")
            # TODO: Display selected quality (if there)
            self.setText(
                "Successfully exported output image.\n"
                + "File size is: "
                + str(round(os.path.getsize(imgPath) / 1024 / 1024, 2))
                + "MB.\n"
            )
            self.setInformativeText("File location:\n" + imgPath)


def createDialog(imageArray, imType, chosenPath):
    imgPath = None
    errorStackTrace = None

    if imType == None:
        return

    # TODO: Allow user to change compression settings (for jpg and png)
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

    # Try saving image to disk
    if imType == "JPG":
        try:
            # 0-100 quality
            cv2.imwrite(
                chosenPath, imageArray, [cv2.IMWRITE_JPEG_QUALITY, compressionFactor]
            )
            imgPath = chosenPath
        except Exception as e:
            errorStackTrace = e
    elif imType == "PNG":
        try:
            # 9-0 compression
            cv2.imwrite(
                chosenPath, imageArray, [cv2.IMWRITE_PNG_COMPRESSION, compressionFactor]
            )
            imgPath = chosenPath
        except Exception as e:
            errorStackTrace = e
    elif imType == "TIFF":
        try:
            cv2.imwrite(chosenPath, imageArray)
            imgPath = chosenPath
        except Exception as e:
            errorStackTrace = e

    ResultDialog(imgPath, errorStackTrace).exec()
