# QT
import pymodule.Qt as Qt
from pymodule.Qt import QtGui
from pymodule.Qt import QtWidgets
from pymodule.Qt import QtCore

### SHOT
class TreeWidget_CheckableItem(QtWidgets.QTreeWidgetItem):
    def __init__(self, parent=None, dataDic={}, assetItem=False ):
        QtWidgets.QTreeWidgetItem.__init__(self, parent)
        # self.setCheckState(0, QtCore.Qt.Unchecked)
        # self.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
        self.dataDic = dataDic
        self.pathData = dict()
        check_styles = """
                    QCheckBox:checked { color: rgb(150, 165, 255);}
                    QCheckBox:indicator { height:15px; width:15px;}
                    """
        geo_styles = """
                    QCheckBox::indicator { height:15px; width:15px;}
                    QCheckBox::indicator:checked { background: rgb(150, 165, 255);}
                    """
        zenn_styles = """
                    QCheckBox::indicator { height:15px; width:15px;}
                    QCheckBox::indicator:checked { background: rgb(150, 165, 255);}
                    """
        ver_styles = """
                   QLineEdit { background: rgb(70, 70, 70); color: white;}
                   QLineEdit:focus { border: 2px solid rgb(170, 170, 255, 150); border-radius: 1px; }
                   QLineEdit:hover { border: 2px solid rgb(170, 170, 255, 150); border-radius: 1px; }
                   """
        brush = QtGui.QBrush()
        brush.setColor(QtGui.QColor(207, 165, 255, 255))
        self.setForeground(4, brush)

        # assembly, camera
        if dataDic:
            self.importCheck = QtWidgets.QCheckBox()
            self.importCheck.setChecked(True)
            self.treeWidget().setItemWidget(self, 0, self.importCheck)
            self.importCheck.setStyleSheet(check_styles)

            self.importVer = QtWidgets.QLineEdit()
            self.importVer.setReadOnly(True)
            self.treeWidget().setItemWidget(self, 5, self.importVer)
            self.importVer.setStyleSheet(ver_styles)
            self.importVer.setMinimumHeight(25)

            self.setText(4, self.dataDic['data_type'])

        # assets
        else:
            self.emptyLabel = QtWidgets.QLabel()
            self.treeWidget().setItemWidget(self, 5, self.emptyLabel)
            self.emptyLabel.setMinimumHeight(25)

            self.geoCheck = QtWidgets.QCheckBox()
            self.geoCheck.setChecked(True)
            self.treeWidget().setItemWidget(self, 1, self.geoCheck)
            self.geoCheck.setStyleSheet(geo_styles)

            self.zennCheck = QtWidgets.QCheckBox()
            self.zennCheck.setChecked(True)
            self.treeWidget().setItemWidget(self, 2, self.zennCheck)
            self.zennCheck.setStyleSheet(zenn_styles)


            self.geoCheck.stateChanged.connect(self.geoChanged)
            self.zennCheck.stateChanged.connect(self.zennChanged)

        if assetItem:
            self.setText(4, 'assets')

        # for character item
        if not assetItem and not self.text(3) in ['camera','assembly']:
            brush = QtGui.QBrush()
            brush.setColor(QtGui.QColor(150, 165, 255, 255))
            self.setForeground(3, brush)

        if assetItem == True:
            geo_styles = """
                        QCheckBox::indicator { height:15px; width:15px;}
                        QCheckBox::indicator:checked { background: rgb(207, 165, 255);}
                        """
            zenn_styles = """
                        QCheckBox::indicator { height:15px; width:15px;}
                        QCheckBox::indicator:checked { background: rgb(207, 165, 255);}
                        """
            self.geoCheck.setStyleSheet(geo_styles)
            self.zennCheck.setStyleSheet(zenn_styles)
            self.geoCheck.stateChanged.connect(self.changeAllGeo)
            self.zennCheck.stateChanged.connect(self.changeAllZenn)

    def changeAllGeo(self, state):
        checkState = self.geoCheck.isChecked()
        for index in range(self.treeWidget().topLevelItemCount()):
            childItem = self.treeWidget().topLevelItem(index)
            if not childItem.text(4) in ['assembly','camera']:
                childItem.geoCheck.setChecked(checkState)

    def changeAllZenn(self, state):
        checkState = self.zennCheck.isChecked()
        for index in range(self.treeWidget().topLevelItemCount()):
            childItem = self.treeWidget().topLevelItem(index)
            if not childItem.text(4) in ['assembly','camera']:
                childItem.zennCheck.setChecked(checkState)

    def geoChanged(self, state):
        # IF ITEM IS TOP LEVEL
        if not (self.parent()):
            checkState = self.geoCheck.isChecked()
            for index in range(self.childCount()):
                childItem = self.child(index)
                childItem.geoCheck.setChecked(checkState)

    def zennChanged(self, state):
        if not (self.parent()):
            checkState = self.zennCheck.isChecked()
            for index in range(self.childCount()):
                childItem = self.child(index)
                childItem.zennCheck.setChecked(checkState)

    def setDict(self, dict):
        self.dataDic = dict

    def getDict(self):
        return self.dataDic
        

class TreeWidgetChild_CheckableItem(QtWidgets.QTreeWidgetItem):
    def __init__(self, parent=None, assetName='', dataDic={} ):
        QtWidgets.QTreeWidgetItem.__init__(self, parent)
        self.assetName = assetName
        self.dataDic = dataDic
        geo_styles = """
                        QCheckBox:checked { color: rgb(207, 165, 255);}
                        QCheckBox:indicator { height:15px; width:15px;}
                        """
        zenn_styles = """
                        QCheckBox:checked { color: rgb(207, 165, 255);}
                        QCheckBox:indicator { height:15px; width:15px;}
                        """
        comboBox_Style = '''
        QComboBox { padding : 0 5 0 5 }
        QComboBox QAbstractItemView::item {
                                            background: rgb(60, 60, 60);
                                            padding: 0 5 0 5 px; margin: 0px; border: 0 px;
                                            min-height: 25px; min-width: 120px; max-height: 250px;
                                            }
        QComboBox QAbstractItemView::item:selected { background: rgb(150, 120, 200); }
        QComboBox QAbstractItemView { font-size: 10pt;}
        '''
        
        self.geoCheck = QtWidgets.QCheckBox()
        self.geoCheck.setStyleSheet(geo_styles)
        self.treeWidget().setItemWidget(self, 1, self.geoCheck)
        # self.geoCheck.setChecked(True)

        self.zennCheck = QtWidgets.QCheckBox()
        self.zennCheck.setStyleSheet(zenn_styles)
        self.treeWidget().setItemWidget(self, 2, self.zennCheck)
        # self.zennCheck.setChecked(True)

        self.setText(3, self.assetName)
        self.pathData = self.dataDic[self.assetName]
        self.setText(4, 'geoCache')
        self.setText(6, 'zenn')

        self.geoPath = {}
        self.zennPath = {}
        self.geoVer = QtWidgets.QComboBox()
        self.geoVer.setView(QtWidgets.QListView())
        self.geoVer.setStyleSheet(comboBox_Style)
        self.geoVer.setMinimumHeight(25)
        self.zennVer = QtWidgets.QComboBox()
        self.zennVer.setView(QtWidgets.QListView())
        self.zennVer.setStyleSheet(comboBox_Style)
        self.zennVer.setMinimumHeight(25)
        self.treeWidget().setItemWidget(self, 5, self.geoVer)
        self.treeWidget().setItemWidget(self, 7, self.zennVer)

    def setDict(self, dict):
        self.dataDic = dict

    def getDict(self):
        return self.dataDic

### ASSET
class Asset_TreeWidget_CheckableItem(QtWidgets.QTreeWidgetItem):
    def __init__(self, parent=None, dataType='' ):
        QtWidgets.QTreeWidgetItem.__init__(self, parent)
        check_styles = """
                    QCheckBox:checked { color: rgb(207, 165, 255);}
                    QCheckBox:indicator { height:15px; width:15px; }
                    """
        ver_styles = """
                   QLineEdit { background: rgb(70, 70, 70); color: white; }
                   QLineEdit:focus { border: 2px solid rgb(170, 170, 255, 150); border-radius: 1px; }
                   QLineEdit:hover { border: 2px solid rgb(170, 170, 255, 150); border-radius: 1px; }
                   """

        self.importCheck = QtWidgets.QCheckBox()
        self.importCheck.setStyleSheet(check_styles)
        self.treeWidget().setItemWidget(self, 0, self.importCheck)
        self.setText(1, dataType)
        self.importVer = QtWidgets.QLineEdit()
        self.importVer.setReadOnly(True)
        self.importVer.setStyleSheet(ver_styles)
        self.treeWidget().setItemWidget(self, 2, self.importVer)
        self.importVer.setMinimumHeight(25)
        self.importVer.setMaximumWidth(80)

    def setDict(self, dict):
        self.dataDic = dict

    def getDict(self):
        return self.dataDic

class DataBaseViewerItem(QtWidgets.QTreeWidgetItem):
    def __init__(self, parent=None):
        QtWidgets.QTreeWidgetItem.__init__(self, parent)
        self.dataDict = {}

    def setDict(self, dict):
        self.dataDict = dict

    def getDict(self):
        return self.dataDict

    def setTexts(self):
        self.setText(0, self.dataDict['data_type'])
        self.setText(1, str(self.dataDict['version']))
        self.setText(2, str(self.dataDict['artist']))
        self.setText(3, str(self.dataDict['time'][0:16]))

class Download_CheckableItem(QtWidgets.QTreeWidgetItem):
    def __init__(self, parent=None, type='', fileName='' ):
        QtWidgets.QTreeWidgetItem.__init__(self, parent)

        self.check = QtWidgets.QCheckBox()
        self.treeWidget().setItemWidget(self, 0, self.check)
        self.check.setChecked(True)
        self.setText(1, type)
        self.setText(2, fileName)
