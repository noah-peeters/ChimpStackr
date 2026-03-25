"""
Create dialog(s) for changing image export settings; and success/error message.
Supports 8-bit (JPG, PNG, TIFF) and 16-bit (PNG, TIFF) output.
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


class SelectBitDepthDialog(qtw.QDialog):
    """Dialog for selecting bit depth when saving TIFF or PNG."""
    selectedBitDepth = None  # 8 or 16

    def __init__(self, imType):
        super().__init__()
        self.setWindowTitle(f"Export image as {imType}")

        verticalLayout = qtw.QVBoxLayout()

        label = qtw.QLabel("Select bit depth:")
        verticalLayout.addWidget(label)

        self.radio_8bit = qtw.QRadioButton("8-bit (standard, smaller file)")
        self.radio_16bit = qtw.QRadioButton("16-bit (higher precision, larger file)")
        self.radio_8bit.setChecked(True)

        verticalLayout.addWidget(self.radio_8bit)
        verticalLayout.addWidget(self.radio_16bit)

        # Add compression for PNG
        self.compression_layout = None
        self.slider = None
        self.spinBox = None
        if imType == "PNG":
            self.compression_layout = qtw.QHBoxLayout()
            self.compression_layout.addWidget(qtw.QLabel("Compression level:"))
            self.slider = qtw.QSlider(qtc.Qt.Orientation.Horizontal)
            self.slider.setMinimum(0)
            self.slider.setMaximum(9)
            self.slider.setValue(4)
            self.slider.setSingleStep(1)
            self.spinBox = qtw.QSpinBox()
            self.spinBox.setMinimum(0)
            self.spinBox.setMaximum(9)
            self.spinBox.setValue(4)
            self.slider.valueChanged.connect(
                lambda v: self.spinBox.setValue(v) if self.spinBox.value() != v else None
            )
            self.spinBox.valueChanged.connect(
                lambda v: self.slider.setValue(v) if self.slider.value() != v else None
            )
            self.compression_layout.addWidget(self.slider)
            self.compression_layout.addWidget(self.spinBox)
            verticalLayout.addLayout(self.compression_layout)

        buttonBox = qtw.QDialogButtonBox(self)
        buttonBox.addButton("Cancel", qtw.QDialogButtonBox.RejectRole)
        buttonBox.addButton("Apply", qtw.QDialogButtonBox.AcceptRole)
        verticalLayout.addWidget(buttonBox)
        buttonBox.rejected.connect(self.close)
        buttonBox.accepted.connect(self.apply_settings)

        self.setLayout(verticalLayout)

    def apply_settings(self):
        self.selectedBitDepth = 16 if self.radio_16bit.isChecked() else 8
        self.close()


# Result dialog on image saved
class ResultDialog(qtw.QMessageBox):
    def __init__(self, imgPath=None, errorStackTrace=None):
        super().__init__()

        if imgPath is None:
            # Error msg
            self.setStandardButtons(qtw.QMessageBox.Ok)
            self.setIcon(qtw.QMessageBox.Critical)
            self.setWindowTitle("Export failed")
            self.setText("Failed to export!\n")
            if errorStackTrace is not None:
                self.setInformativeText("Error stack trace:")
                self.setDetailedText(str(errorStackTrace))

        elif imgPath is not None and errorStackTrace is None:
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


def _convert_to_uint8(imageArray):
    """Convert float32 image (0-255 range) to uint8."""
    out = np.clip(imageArray, 0, 255)
    return np.around(out).astype(np.uint8)


def _convert_to_uint16(imageArray):
    """Convert float32 image (0-255 range) to uint16 (0-65535 range).

    Scales from 0-255 float to 0-65535 uint16, preserving the extra
    precision from the float32 stacking pipeline. This gives 256x more
    tonal gradations than 8-bit output.
    """
    out = np.clip(imageArray, 0, 255)
    # Scale 0-255 -> 0-65535 (multiply by 257 = 65535/255)
    out = out * 257.0
    return np.around(out).astype(np.uint16)


def createDialog(imageArray, imType, chosenPath):
    # Something went wrong
    if imType is None:
        return

    imgPath = None
    errorStackTrace = None
    bit_depth = 8
    compressionFactor = None

    if imType == "JPG":
        # JPG: quality dialog only (always 8-bit)
        qualityDialog = SelectQualityDialog(imType)
        qualityDialog.exec()
        if qualityDialog.selectedQuality is not None:
            compressionFactor = qualityDialog.selectedQuality
        else:
            return

    elif imType == "PNG":
        # PNG: bit depth + compression dialog
        depthDialog = SelectBitDepthDialog("PNG")
        depthDialog.exec()
        if depthDialog.selectedBitDepth is not None:
            bit_depth = depthDialog.selectedBitDepth
            if depthDialog.slider is not None:
                compressionFactor = depthDialog.slider.value()
            else:
                compressionFactor = 4
        else:
            return

    elif imType == "TIFF":
        # TIFF: bit depth dialog only (no compression needed)
        depthDialog = SelectBitDepthDialog("TIFF")
        depthDialog.exec()
        if depthDialog.selectedBitDepth is not None:
            bit_depth = depthDialog.selectedBitDepth
        else:
            return

    elif imType == "EXR":
        # EXR: always 32-bit float, no dialog needed
        bit_depth = 32

    # Convert to target bit depth
    if bit_depth == 32:
        # EXR: keep as float32, scale to 0-1 range (EXR convention)
        imageOut = np.clip(imageArray, 0, 255).astype(np.float32) / 255.0
    elif bit_depth == 16:
        imageOut = _convert_to_uint16(imageArray)
    else:
        imageOut = _convert_to_uint8(imageArray)

    # Build compression/quality params
    compression_list = None
    if imType == "JPG":
        compression_list = [cv2.IMWRITE_JPEG_QUALITY, compressionFactor]
    elif imType == "PNG":
        compression_list = [cv2.IMWRITE_PNG_COMPRESSION, compressionFactor or 4]

    # Try saving image to disk
    try:
        cv2.imwrite(chosenPath, imageOut, compression_list)
        imgPath = chosenPath
    except Exception as e:
        errorStackTrace = e

    ResultDialog(imgPath, errorStackTrace).exec()
