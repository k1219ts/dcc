from PySide2 import QtWidgets
main_layout = QtWidgets.QWidget()
main_layout.resize(240,40)
btnLayout = QtWidgets.QHBoxLayout(main_layout)
pickBtn = QtWidgets.QPushButton('Pick')
pickBtn.resize(120,40)
setBtn = QtWidgets.QPushButton('Del Const')
setBtn.resize(120,40)
btnLayout.addWidget(pickBtn)
btnLayout.addWidget(setBtn)
pickBtn.clicked.connect(pickDF)
setBtn.clicked.connect(setDF)
def pickDF():
    sel = str(cmds.ls(sl=True)[0])
    nsChar = sel.split(":")[0]
    if nsChar:
        tempA = str(cmds.parentConstraint(sel, nsChar + "_PathAnimCurves", mo=True, w=1)[0])
        tempB = str(cmds.parentConstraint(sel, nsChar + "_curveF_crtl_GRP", mo=True, w=1)[0])
def setDF():
    sel = str(cmds.ls(sl=True)[0])
    nsChar = sel.split(":")[0]
    if nsChar:
        tempA = str(cmds.listRelatives(nsChar + "_PathAnimCurves", c=True, type="parentConstraint")[0])
        tempB = str(cmds.listRelatives(nsChar + "_curveF_crtl_GRP", c=True, type="parentConstraint")[0])
        cmds.delete(tempA)
        cmds.delete(tempB)
main_layout.show()