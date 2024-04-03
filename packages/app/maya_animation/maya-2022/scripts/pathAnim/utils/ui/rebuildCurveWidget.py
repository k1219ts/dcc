import maya.cmds as cmds
import pathAnim.utils.rebuildCurve
from PySide2 import QtCore, QtWidgets
import aniCommon
reload(pathAnim.utils.rebuildCurve)


class RebuildCurveWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(RebuildCurveWidget, self).__init__(parent)
        self.setWindowTitle("Rebuild Curve GUI")

        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(0, 3, 0, 3)
        main_layout.setSpacing(3)
        label_0 = QtWidgets.QLabel('Path Spans')
        main_layout.addWidget(label_0)
        label_0.setAlignment(QtCore.Qt.AlignCenter)

        self.spans_lineEdit = QtWidgets.QLineEdit('50')
        self.rebuild_btn = QtWidgets.QPushButton('Rebuild Path')
        self.rebuild_btn.clicked.connect(self.rebuild)
        self.reverse_btn = QtWidgets.QPushButton('Reverse Path')
        self.reverse_btn.clicked.connect(self.reverse)
        self.projectMesh_btn = QtWidgets.QPushButton('Project to Selected Mesh')
        self.projectMesh_btn.clicked.connect(self.project)

        main_layout.addWidget(self.spans_lineEdit)
        main_layout.addWidget(self.rebuild_btn)
        main_layout.addWidget(self.reverse_btn)
        main_layout.addWidget(self.projectMesh_btn)

    @aniCommon.undo
    def rebuild(self):
        spans = int(self.spans_lineEdit.text())
        curve = cmds.ls(sl=True)
        pathAnim.utils.rebuildCurve.rebuildPathCurves(curve, spans)

    @aniCommon.undo
    def reverse(self):
        curves = cmds.ls(sl=True)
        for curve in curves:
            curveShp = cmds.listRelatives(curve, s=True)
            if cmds.objectType(curveShp) != 'nurbsCurve':
                raise Exception("Select Curve")

        pathAnim.utils.rebuildCurve.reverseCurve(curves)

    @aniCommon.undo
    def project(self):
        selection = cmds.ls(sl=True)
        pathAnim.utils.rebuildCurve.projectPathCurves(selection)
