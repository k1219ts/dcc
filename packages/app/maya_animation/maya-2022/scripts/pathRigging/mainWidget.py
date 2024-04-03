from PySide2 import QtCore, QtGui, QtWidgets
import maya.cmds as cmds
import maya.mel as mel


class PathRiggingWidget(QtWidgets.QMainWindow):

    def __init__(self, parent=None):
        super(PathRiggingWidget, self).__init__(parent)
        self.setWindowTitle("Dexter Animation - Path Rigging")

        # main widget
        main_widget = QtWidgets.QWidget(self)
        self.setCentralWidget(main_widget)
        main_layout = QtWidgets.QVBoxLayout(main_widget)
        rebuildButton = QtWidgets.QPushButton('Rebuild Curve')
        updateButton = QtWidgets.QPushButton('Update')

        rebuildButton.clicked.connect(self.rebuild)
        updateButton.clicked.connect(self.update)

        main_layout.addWidget(rebuildButton)
        main_layout.addWidget(updateButton)



    def rebuild(self):
        if cmds.ls('rebuild*'):
            c = set(cmds.listConnections('*:pathSnake_CONShape.create'))
            RBCList = set(cmds.ls('rebuild*'))
            d = RBCList.difference(c)
            cmds.delete(list(d))

    def update(self):
        a = set(cmds.referenceQuery('whiteCobraRN', es=True, fld=True, ec='setAttr', showDagPath=False))
        # failed setAttr
        b = set(cmds.referenceQuery('whiteCobraRN', es=True, fld=False, ec='setAttr', showDagPath=False))
        # no failed setAttr
        if list(a.difference(b)):
            mel.eval(list(a.difference(b))[-1])
