# -*- coding: utf-8 -*-
from pymodule.Qt import QtWidgets

class TagModifyDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, tags=None):
        super(TagModifyDialog, self).__init__(parent)
        # self.resize(1200,900)
        self.setWindowTitle('tags modify')
        self.gridLayout = QtWidgets.QGridLayout(self)
        self.tagEdit = QtWidgets.QTextEdit(self)
        self.tagEdit.setText("\n".join(tags))
        # self.detailTree.setDetailData(infos)
        self.gridLayout.addWidget(self.tagEdit, 0, 0, 1, 2)
        self.setStyleSheet("""
        TagModifyDialog{color: rgb(200,200,200); background: rgb(48,48,48); border-width: 1px;}
        """)
        self.tagEdit.setStyleSheet("""
                QTextEdit {color: #CCCCCC; font: 14px; width: 100px; background-color: #323232;
                border: 1px solid #3d3d3d;}
                QTextEdit:selected { background-color: #FF8D1D; color: #000000;}
                """)

        self.okBtn = QtWidgets.QPushButton(self)
        self.okBtn.setText("OK")
        self.okBtn.clicked.connect(self.accept)
        self.gridLayout.addWidget(self.okBtn, 1, 0, 1, 1)

        self.cancelBtn = QtWidgets.QPushButton(self)
        self.cancelBtn.setText("CANCEL")
        self.cancelBtn.clicked.connect(self.reject)
        self.gridLayout.addWidget(self.cancelBtn, 1, 1, 1, 1)