
from PySide2 import QtCore, QtWidgets
import pathAnim.utils.stretchFix
import aniCommon

reload(pathAnim.utils.stretchFix)

class StretchFixWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(StretchFixWidget, self).__init__(parent)
        self.setWindowTitle("Stretch Fix GUI")

        main_layout = QtWidgets.QVBoxLayout(self)
        label_0 = QtWidgets.QLabel('Adjust Range')
        main_layout.addWidget(label_0)
        label_0.setAlignment(QtCore.Qt.AlignCenter)

        slider_layout = QtWidgets.QHBoxLayout()
        self.value_lineEdit = QtWidgets.QLineEdit('0.100')
        self.value_lineEdit.setFixedWidth(50)
        self.value_lineEdit.textChanged.connect(self.valueChaned)

        self.value_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.value_slider.valueChanged.connect(self.sliderChanged)
        self.value_slider.setRange(1, 500)
        self.value_slider.setValue(100)
        self.value_slider.setSingleStep(1)
        self.value_slider.setFixedHeight(18)

        slider_layout.addWidget(self.value_lineEdit)
        slider_layout.addWidget(self.value_slider)

        self.reduce_btn = QtWidgets.QPushButton('Reduce Stretch')
        self.reduce_btn.clicked.connect(lambda: self.stretch(1))
        self.increase_btn = QtWidgets.QPushButton('Increase Stretch')
        self.increase_btn.clicked.connect(lambda: self.stretch(-1))

        main_layout.addLayout(slider_layout)
        main_layout.addWidget(self.reduce_btn)
        main_layout.addWidget(self.increase_btn)


    def valueChaned(self):
        value = self.value_lineEdit.text()
        if value:
            self.value_slider.setValue(float(value)*1000)

    def sliderChanged(self):
        value = str("{0:.3f}".format(self.value_slider.value()/1000.0))
        self.value_lineEdit.setText(value)

    @aniCommon.undo
    def stretch(self, multiplier):
        sr = float(self.value_lineEdit.text())
        pathAnim.utils.stretchFix.stretchBtnProc(multiplier=multiplier, stretchRange=sr)

