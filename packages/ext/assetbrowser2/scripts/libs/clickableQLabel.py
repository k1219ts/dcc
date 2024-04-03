#-*- coding: utf-8 -*-
from pymodule.Qt import QtCore
from pymodule.Qt import QtWidgets

class ClickableQLabel(QtWidgets.QLabel):

    clicked = QtCore.Signal()

    # def __init__(self, max_enlargement=2.0):
    #     QtWidgets.QLabel.__init__(self)

    def mousePressEvent(self, event):
        # TODO: RuntimeError: wrapped C/C++ object of type ClickableQLabel has been deleted
        self.clicked.emit()

    # def enterEvent(self, event):
    #     super(ClickableQLabel, self).enterEvent(event)
    #     self.setStyleSheet("QLabel {background-color: #333333;}")

    # def leaveEvent(self, event):
    #     super(ClickableQLabel, self).leaveEvent(event)
    #     self.setStyleSheet("")
