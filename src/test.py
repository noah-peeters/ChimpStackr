import PySide6
from PySide6 import QtCore, QtGui

import PIL
from PIL import Image, ImageQt

import numpy as np

print(
    f"Tested with:\n- PIL version: {PIL.__version__},\n- PySide6 version: {PySide6.__version__},\n- Qt Version: {QtCore.qVersion()}\n- Numpy version: {np.__version__}"
)

app = QtGui.QGuiApplication()

ar = np.random.randint(255, size=(900, 800, 3), dtype=np.uint8)
img = Image.fromarray(ar.astype(np.uint8))
qim = ImageQt.ImageQt(img)
pm = QtGui.QPixmap.fromImage(qim)
assert not pm.isNull()