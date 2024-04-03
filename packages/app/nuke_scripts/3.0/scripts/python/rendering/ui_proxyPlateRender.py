from PySide2 import QtWidgets, QtGui, QtCore

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(300, 619)
        self.btnDelete = QtWidgets.QPushButton(Form)
        self.btnDelete.setGeometry(QtCore.QRect(10, 580, 61, 27))
        self.btnDelete.setObjectName(_fromUtf8("btnDelete"))
        self.btnDelete.setEnabled(False)
        self.btnPlateLoad = QtWidgets.QPushButton(Form)
        self.btnPlateLoad.setGeometry(QtCore.QRect(80, 580, 98, 27))
        self.btnPlateLoad.setObjectName(_fromUtf8("btnPlateLoad"))
        self.btnRender = QtWidgets.QPushButton(Form)
        self.btnRender.setGeometry(QtCore.QRect(190, 580, 98, 27))
        self.btnRender.setObjectName(_fromUtf8("btnRender"))
        self.cmbMilestone = QtWidgets.QComboBox(Form)
        self.cmbMilestone.setGeometry(QtCore.QRect(100, 10, 191, 25))
        self.cmbMilestone.setObjectName(_fromUtf8("cmbMilestone"))
        self.label = QtWidgets.QLabel(Form)
        self.label.setGeometry(QtCore.QRect(20, 10, 71, 21))
        self.label.setObjectName(_fromUtf8("label"))
        self.listShot = QtWidgets.QListWidget(Form)
        self.listShot.setGeometry(QtCore.QRect(10, 50, 281, 511))
        self.listShot.setSelectionMode(QtWidgets.QListWidget.MultiSelection)
        self.listShot.setObjectName(_fromUtf8("listShot"))

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        #Form.setWindowTitle(QtWidgets.QApplication.translate("Form", "PRS 2K Plate Render", None))
        self.btnDelete.setText(QtWidgets.QApplication.translate("Form", "Del", None))
        self.btnPlateLoad.setText(QtWidgets.QApplication.translate("Form", "PlateLoad", None))
        self.btnRender.setText(QtWidgets.QApplication.translate("Form", "Render", None))
        self.label.setText(QtWidgets.QApplication.translate("Form", "Milestone", None))
