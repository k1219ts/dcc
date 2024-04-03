import Qt.QtCore as QtCore
import Qt.QtGui as QtGui
import Qt.QtWidgets as QtWidgets

# define column num
CHECKBOX = 0
FILENAME = 1

class EnvFileTreeWidgetItem(QtWidgets.QTreeWidgetItem):
    def __init__(self, parent, fileName):
        super(self.__class__, self).__init__(parent)
        
        # 0 == ASSET_NAME
        self.loadCheckBox = QtWidgets.QCheckBox()
        self.loadCheckBox.setCheckState(QtCore.Qt.Checked)
        self.loadCheckBox.setStyleSheet('''
                                        QCheckBox::indicator {
                                            width:20px;
                                            height:20px;
                                        }
                                        QCheckBox::indicator:checked {
                                            image:url(/netapp/backstage/pub/apps/maya2/versions/2017/team/asset/linux/scripts/AssetLookdevTool/images/Sign-Checkmark01-Green.png);
                                        }
                                        QCheckBox::indicator:unchecked:pressed {
                                            image:url(/netapp/backstage/pub/apps/maya2/versions/2017/team/asset/linux/scripts/AssetLookdevTool/images/Sign-Checkmark01-Green.png);
                                        }
                                        
                                        ''')
        parent.setItemWidget(self, CHECKBOX, self.loadCheckBox)
        
        # 1 == ALEMBIC_VERSION
        self.fileNameLabel = QtWidgets.QLabel()
        font = QtGui.QFont("Cantarell", 13)
        self.fileNameLabel.setFont(font)
        self.fileNameLabel.setText(fileName)
        parent.setItemWidget(self, FILENAME, self.fileNameLabel)
        
    def getState(self):
        if self.loadCheckBox.checkState() == QtCore.Qt.Checked:
            return True
        return False
    
    def getFileName(self):
        return str(self.fileNameLabel.text())
#     
#     def sizeHint(self):
#         return 30
