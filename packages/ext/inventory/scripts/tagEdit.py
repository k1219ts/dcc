# -*- coding: utf-8 -*-

from PyQt4 import QtCore, QtGui

class TagEdit(QtGui.QDialog):
    def __init__(self, parent=None, tags=[]):
        super(TagEdit, self).__init__(parent)

        self.gridLayout = QtGui.QGridLayout(self)

        self.label = QtGui.QLabel(self)
        self.label.setText('Tags')
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
        self.textEdit = QtGui.QTextEdit(self)

        self.gridLayout.addWidget(self.textEdit, 1, 0, 1, 1)
        self.buttonBox = QtGui.QDialogButtonBox(self)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(
            QtGui.QDialogButtonBox.Cancel | QtGui.QDialogButtonBox.Ok)

        self.gridLayout.addWidget(self.buttonBox, 2, 0, 1, 1)

        self.buttonBox.accepted.connect(self.accepted)
        self.buttonBox.rejected.connect(self.rejected)

        if tags:
            self.tags = tags
        else:
            self.tags = []

        for tag in self.tags:
            self.textEdit.insertPlainText(tag)






