import logging
import modules; reload(modules)
from PySide2 import QtCore, QtGui, QtWidgets

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class MatchGround(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(MatchGround, self).__init__(parent)
        self.setWindowTitle("Match Ground")
        main_widget = QtWidgets.QWidget(self)
        main_layout = QtWidgets.QVBoxLayout(main_widget)
        self.setCentralWidget(main_widget)
        label1 = QtWidgets.QLabel("Ground Mesh")
        self.groundMeshLayout = QtWidgets.QHBoxLayout()
        self.groundMeshLineEdit = QtWidgets.QLineEdit()
        self.addGroundMesh_Btn = QtWidgets.QPushButton("<<")
        self.addGroundMesh_Btn.setMinimumWidth(30)
        self.groundMeshLayout.addWidget(self.groundMeshLineEdit)
        self.groundMeshLayout.addWidget(self.addGroundMesh_Btn)

        addOption_layout = QtWidgets.QHBoxLayout()
        addOption_radioButtons = QtWidgets.QButtonGroup(self)
        label2 = QtWidgets.QLabel("Add Option : ")
        self.addOption_radioButton1 = QtWidgets.QRadioButton(main_widget)
        self.addOption_radioButton1.setText("replace")
        self.addOption_radioButton1.setChecked(True)
        self.addOption_radioButton2 = QtWidgets.QRadioButton(main_widget)
        self.addOption_radioButton2.setText("add")
        verticalSpacer = QtWidgets.QSpacerItem(20, 30, QtWidgets.QSizePolicy.Expanding,
                                               QtWidgets.QSizePolicy.Minimum)
        addOption_radioButtons.addButton(self.addOption_radioButton1)
        addOption_radioButtons.addButton(self.addOption_radioButton2)
        addOption_layout.addWidget(label2)
        addOption_layout.addWidget(self.addOption_radioButton1)
        addOption_layout.addWidget(self.addOption_radioButton2)
        addOption_layout.addItem(verticalSpacer)


        self.tableLayout = QtWidgets.QHBoxLayout()
        self.tableWidget = QtWidgets.QTableWidget()
        self.tableItems = QtWidgets.QTableWidgetItem()
        self.tableWidget.setColumnCount(2)
        self.tableWidget.setHorizontalHeaderLabels(["Controler", "Offset"])
        header = self.tableWidget.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        self.tableWidget.setColumnWidth(1, 100)

        self.tableButtonLayout = QtWidgets.QVBoxLayout()
        self.addControlers_Btn = QtWidgets.QPushButton("+")
        self.addControlers_Btn.setMinimumWidth(30)
        self.removeControlers_Btn = QtWidgets.QPushButton("-")
        self.clearTable_Btn = QtWidgets.QPushButton("clear")
        horizontalSpacer1 = QtWidgets.QSpacerItem(20, 30, QtWidgets.QSizePolicy.Minimum,
                                                QtWidgets.QSizePolicy.Expanding)
        self.tableButtonLayout.addWidget(self.addControlers_Btn)
        self.tableButtonLayout.addWidget(self.removeControlers_Btn)
        self.tableButtonLayout.addWidget(self.clearTable_Btn)
        self.tableButtonLayout.addItem(horizontalSpacer1)

        self.tableLayout.addWidget(self.tableWidget)
        self.tableLayout.addLayout(self.tableButtonLayout)

        layerOption_layout = QtWidgets.QHBoxLayout()
        layerOption_radioButtonGroup = QtWidgets.QButtonGroup(self)
        layerOption_Label = QtWidgets.QLabel('Layer Option : ')
        self.baseLayer_radioBtn = QtWidgets.QRadioButton(main_widget)
        self.baseLayer_radioBtn.setText('Base MG Anim Layer')
        self.baseLayer_radioBtn.setChecked(True)
        self.newLayer_radioBtn = QtWidgets.QRadioButton(main_widget)
        self.newLayer_radioBtn.setText('New layer')
        verticalSpacer2 = QtWidgets.QSpacerItem(20, 30, QtWidgets.QSizePolicy.Expanding,
                                                QtWidgets.QSizePolicy.Minimum)
        layerOption_radioButtonGroup.addButton(self.baseLayer_radioBtn)
        layerOption_radioButtonGroup.addButton(self.newLayer_radioBtn)
        layerOption_layout.addWidget(layerOption_Label)
        layerOption_layout.addWidget(self.baseLayer_radioBtn)
        layerOption_layout.addWidget(self.newLayer_radioBtn)
        layerOption_layout.addItem(verticalSpacer2)

        button_layout = QtWidgets.QHBoxLayout()
        self.attach_btn = QtWidgets.QPushButton("Attach")
        self.attach_btn.setMinimumWidth(100)
        verticalSpacer1 = QtWidgets.QSpacerItem(20, 30, QtWidgets.QSizePolicy.Expanding,
                                                  QtWidgets.QSizePolicy.Minimum)
        self.deleteNode_btn = QtWidgets.QPushButton("Delete Nodes")
        self.deleteNode_btn.setMinimumWidth(100)
        button_layout.addWidget(self.attach_btn)
        button_layout.addItem(verticalSpacer1)
        button_layout.addWidget(self.deleteNode_btn)
        main_layout.addWidget(label1)
        main_layout.addLayout(self.groundMeshLayout)
        main_layout.addWidget(self.HLine())
        main_layout.addLayout(addOption_layout)
        main_layout.addLayout(self.tableLayout)
        main_layout.addWidget(self.HLine())
        main_layout.addLayout(layerOption_layout)
        main_layout.addWidget(self.HLine())
        main_layout.addLayout(button_layout)

        self.connectSignals()

    def connectSignals(self):
        self.addGroundMesh_Btn.clicked.connect(self.addGroundMesh)
        self.addControlers_Btn.clicked.connect(self.addControlers)
        self.removeControlers_Btn.clicked.connect(self.removeControlers)
        self.clearTable_Btn.clicked.connect(lambda :self.tableWidget.setRowCount(0))
        self.attach_btn.clicked.connect(self.attach)
        self.deleteNode_btn.clicked.connect(self.deleteMgNode)

    def messageBox(self, message):
        msg_box = QtWidgets.QMessageBox(self)
        msg_box.setWindowTitle('Warning!')
        msg_box.setIcon(QtWidgets.QMessageBox.Warning)
        msg_box.setText(message)
        msg_box.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        if msg_box.exec_() != QtWidgets.QMessageBox.Yes:
            raise Exception("Canceled!")

    def HLine(self):
        toto = QtWidgets.QFrame()
        toto.setFrameShape(QtWidgets.QFrame.HLine)
        toto.setFrameShadow(QtWidgets.QFrame.Sunken)
        return toto

    def VLine(self):
        toto = QtWidgets.QFrame()
        toto.setFrameShape(QtWidgets.QFrame.VLine)
        toto.setFrameShadow(QtWidgets.QFrame.Sunken)
        return toto

    def addText(self, widget, text):
        widget.clear()
        widget.setText(str(text))

    def addGroundMesh(self):
        sel = modules.getSelection()
        if not sel or len(sel) > 1:
            raise Exception("Select One Mesh")
        self.addText(self.groundMeshLineEdit, sel[0])

    def addControlers(self):
        isReplace = self.addOption_radioButton1.isChecked()
        sel = modules.getSelection()
        cons = list()


        if not sel:
            raise Exception("Select Controler")

        for con in sel:
            if not isReplace:
                if not self.tableWidget.findItems(con, QtCore.Qt.MatchFlags(0)):
                    cons.append(con)
            else:
                cons.append(con)

        if isReplace:
            exCount = 0
            self.tableWidget.clear()
            self.tableWidget.setHorizontalHeaderLabels(["Controler", "Offset"])
            self.tableWidget.setRowCount(len(cons))
        else:
            exCount = self.tableWidget.rowCount()
            self.tableWidget.setRowCount(self.tableWidget.rowCount() + len(cons))

        for i, con in enumerate(cons):
            i += exCount
            self.tableWidget.setItem(i, 0, QtWidgets.QTableWidgetItem(con))

            offset_spinBox = QtWidgets.QDoubleSpinBox()
            offset_spinBox.setMinimum(-999999999)
            offset_spinBox.setMaximum(999999999)
            offset_spinBox.setSingleStep(0.1)
            self.tableWidget.setCellWidget(i, 1, offset_spinBox)
        self.tableWidget.resizeRowsToContents()

    def removeControlers(self):
        sel = self.tableWidget.selectedItems()
        for item in sel:
            modelIndex = self.tableWidget.indexFromItem(item)
            self.tableWidget.removeRow(modelIndex.row())

    def getCtrlInfo(self):
        info = dict()
        tableCount = self.tableWidget.rowCount()
        for num in range(tableCount):
            con = self.tableWidget.item(num, 0).text()
            offset = self.tableWidget.cellWidget(num, 1).value()
            info[con] = offset

        return info

    def attach(self):
        ctrlInfo = self.getCtrlInfo()

        self.messageBox('Attach selected controlers to ground')

        modules.initPlugin()
        cls = modules.MatchGround()
        cls.groundMesh = str(self.groundMeshLineEdit.text())
        cls.controlersInfo = ctrlInfo
        cls.newLayer = self.newLayer_radioBtn.isChecked()
        cls.attach()

    def deleteMgNode(self):
        self.messageBox('Are you sure you want to remove <Match Ground> nodes?')

        modules.MatchGround.deleteNodes(modules.getSelection())

_win = None

def showUI():
    global _win
    if _win:
        _win.close()
    _win = MatchGround()
    _win.show()
    _win.resize(500, 500)
    return _win
