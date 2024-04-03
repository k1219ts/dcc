import os
import requests
import re

from rv import rvtypes
from PySide2 import QtWidgets

import dxConfig
from ui_dxRenameEditOrder import Ui_Dialog
import dxTacticCommon as comm

API_KEY = 'c70181f2b648fdc2102714e8b5cb344d'

class dxRenameEditOrder(rvtypes.MinorMode):
    def __init__(self):
        rvtypes.MinorMode.__init__(self)
        self.init('dxRenameEditOrder', None, None)
        self.widgets = QtWidgets.QWidget()
        self.ui = Ui_Dialog()
        self.ui.setupUi(self.widgets)

        # self.rvSessionQObject = qtutils.sessionWindow()
        # self.dialog = QtWidgets.QDockWidget('dxRenameEditOrder', self.rvSessionQObject)
        # self.dialog.setWidget(self.widgets)

        self.showList = comm.getShowList()

        self.setShowList()
        self.ui.selectDirectory_pushButton.clicked.connect(self.selectDirectory)
        self.ui.apply_pushButton.clicked.connect(self.apply)
    
    def setShowList(self):
        self.ui.show_comboBox.clear()
        self.ui.show_comboBox.addItem('select', None)
        for prj in sorted(self.showList.keys()):
            title = self.showList[prj]['title']
            enTitle = title.split(' ')[-1]
            korTitle = title.split(' ')[0]

            self.ui.show_comboBox.addItem(enTitle[1:-1] + ' ' + korTitle, self.showList[prj])
        
    def selectDirectory(self):
        directory = QtWidgets.QFileDialog.getExistingDirectory(None, 'Select Directory')
        self.ui.directory_plainTextEdit.setPlainText(directory)

    def getShot(self, show, shot):
        params = {}

        params = {}
        params['api_key'] = API_KEY
        params['project_code'] = show
        params['code'] = shot

        response =  requests.get("http://%s/dexter/search/shot.php" % (dxConfig.getConf('TACTIC_IP')), params=params).json()
        if response:
            return response[0]

        return None

    def apply(self):
        index = self.ui.show_comboBox.currentIndex()
        project = self.ui.show_comboBox.itemData(index)
        if not project:
            return self.alert("Error: Project")

        directory = self.ui.directory_plainTextEdit.toPlainText()
        # print(directory)

        if not directory:
            return self.alert("Error: Select Directory")

        try:
            changed = 0
            
            for dir in os.listdir(directory):
                if not re.search(r'^' + project['name'] + '_\d*', dir, flags=re.IGNORECASE):
                    # print(dir)
                    continue

                parseShow = re.sub(r'_[^_]+_[^_]+$', '', dir)

                shot = self.getShot(project['code'], parseShow)
                if not shot:
                    continue

                editOrder = shot['edit_order']
                if not editOrder:
                    continue

                fromDir = '%s/%s' % (directory, dir)

                # File Rename
                for file in os.listdir(fromDir):
                    fromFile = '%s/%s' % (fromDir, file)
                    toFile = '%s/%d_%s' % (fromDir, editOrder, file)
                    os.rename(fromFile, toFile)
                
                # Directory Rename
                toDir = '%s/%d_%s' % (directory, editOrder, dir)
                os.rename(fromDir, toDir)

                changed += 1
                
            self.alert("Changed: %d" % changed)
            self.widgets.hide()
            
        except Exception as e:
            self.alert("Error: %s" % e)
        
    def alert(self, message):
        QtWidgets.QMessageBox.information(
            self.widgets,
            'Info',
            message,
            QtWidgets.QMessageBox.Ok
        )
    
    def activate(self):
        self.widgets.show()

def createMode():
    return dxRenameEditOrder()
