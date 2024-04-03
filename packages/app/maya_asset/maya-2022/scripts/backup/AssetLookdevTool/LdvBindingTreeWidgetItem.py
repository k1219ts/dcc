import Qt.QtCore as QtCore
import Qt.QtGui as QtGui
import Qt.QtWidgets as QtWidgets

import maya.cmds as cmds
import maya.mel as mel

# define column num
XPATH_COLUMN = 0
SHADER_COLUMN = 1

class LdvBindingTreeWidgetItem(QtWidgets.QTreeWidgetItem):
    def __init__(self, parent, xPath, shaderPay):
        super(self.__class__, self).__init__(parent)
        
        self.xPath = str(xPath)
        self.shaderPay = str(shaderPay)
        
        # Column Setting
        
        # 0 == XPATH_COLUMN
        self.xPathLabel = QtWidgets.QLabel()
        self.xPathLabel.setText(self.xPath)
        parent.setItemWidget(self, XPATH_COLUMN, self.xPathLabel)

        # 1 == SHADER_COLUMN        
        self.shaderLabel = QtWidgets.QLabel()
        self.shaderLabel.setText(self.shaderPay)
        parent.setItemWidget(self, SHADER_COLUMN, self.shaderLabel)