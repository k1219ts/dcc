import json
import os
from PyQt4 import QtGui, QtCore

class camPresetWindow(QtGui.QMainWindow):
    def __init__(self, parent=None, jsonfile=None):
        super(camPresetWindow, self).__init__(parent)
        self.jsonfile = jsonfile
        self.parent = parent

        self.setWindowTitle("Edit CameraType List")

        editTypesWidget = QtGui.QWidget(parent)
        editTypesvbox = QtGui.QVBoxLayout(editTypesWidget)
        self.setCentralWidget(editTypesWidget)

        list_groupBox = QtGui.QGroupBox()
        vbox = QtGui.QVBoxLayout(list_groupBox)
        vbox.setContentsMargins(3, 3, 3, 0)
        vbox.setSpacing(6)
        editTypesvbox.addWidget(list_groupBox)

        self.typeListWidget = QtGui.QListWidget()
        editTypesvbox.addWidget(self.typeListWidget)

        self.AddButton = QtGui.QPushButton()
        self.AddButton.setText("Add")
        self.RemoveButton = QtGui.QPushButton()
        self.RemoveButton.setText("Remove")
        hBox = QtGui.QHBoxLayout()
        hBox.addWidget(self.AddButton)
        hBox.addWidget(self.RemoveButton)
        editTypesvbox.addLayout(hBox)

        self.SaveButton = QtGui.QPushButton()
        self.SaveButton.setText("Save")
        self.CancelButton = QtGui.QPushButton()
        self.CancelButton.setText("Cancel")
        hBox = QtGui.QHBoxLayout()
        hBox.addWidget(self.SaveButton)
        hBox.addWidget(self.CancelButton)
        editTypesvbox.addLayout(hBox)

        self.AddButton.clicked.connect(self.addAction)
        self.RemoveButton.clicked.connect(self.removeAction)
        self.SaveButton.clicked.connect(self.saveAction)
        self.CancelButton.clicked.connect(self.cancelAction)

    def addAction(self):
        #typeList = self.getJsonData(jsonFile=self.jsonfile)["TYPES"]
        typeList = [str(self.typeListWidget.item(i).text()) for i in range(self.typeListWidget.count())]
        CustomTypeDialog = QtGui.QInputDialog()
        customType, ok = CustomTypeDialog.getText(self, "Enter Type", "Enter Custom Camera Type",
                                               QtGui.QLineEdit.Normal)
        if ok:
            typeList.append( str(customType) )
            self.typeListWidget.clear()
            self.typeListWidget.addItems(typeList)
        print typeList

    def removeAction(self):
        itemIndex = self.typeListWidget.currentRow()
        self.typeListWidget.takeItem(itemIndex)

    def saveAction(self):
        types = [str(self.typeListWidget.item(i).text()) for i in range(self.typeListWidget.count())]
        camTypeLog = self.getJsonData(jsonFile=self.jsonfile)
        camTypeLog["TYPES"] = types

        with open(self.jsonfile, 'w') as f:
            json.dump(camTypeLog, f, indent=4)
            print "\n"
            print "# Camera Type Preset Saved In :\n"
            print self.jsonfile
            print "\n"
            self.parent.type_comboBox.clear()
            self.parent.type_comboBox.addItems(types)
            self.close()

    def cancelAction(self):
        self.close()

    def getJsonData(self, jsonFile):
        if os.path.exists(jsonFile):
            with open(jsonFile, 'r') as f:
                presetData = json.load(f)
        else:
            presetData = dict()

        return presetData