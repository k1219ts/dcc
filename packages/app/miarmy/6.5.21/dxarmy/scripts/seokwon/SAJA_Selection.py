#
import re
from PySide2 import QtWidgets, QtCore, QtGui

main_layout = QtWidgets.QWidget()
main_layout.resize(333,120)
main_layout.setWindowTitle("SAJA")

allBtn = QtWidgets.QPushButton(main_layout)
allBtn.setText("All Rigs")
allBtn.setGeometry(QtCore.QRect(5,5,159,30))
selBtn = QtWidgets.QPushButton(main_layout)
selBtn.setText('Sel Rig(s)')
selBtn.setGeometry(QtCore.QRect(169,5,159,30))

getLE = QtWidgets.QLineEdit(main_layout)
getLE.setGeometry(QtCore.QRect(5,40,323,40))
getLE.setReadOnly(True)

rtBtn = QtWidgets.QPushButton(main_layout)
rtBtn.setText('Root')
rtBtn.setGeometry(QtCore.QRect(5,85,77,30))
uaBtn = QtWidgets.QPushButton(main_layout)
uaBtn.setText('upArm')
uaBtn.setGeometry(QtCore.QRect(87,85,77,30))
pvBtn = QtWidgets.QPushButton(main_layout)
pvBtn.setText('poleVect')
pvBtn.setGeometry(QtCore.QRect(169,85,77,30))
ihBtn = QtWidgets.QPushButton(main_layout)
ihBtn.setText('IK_Hand')
ihBtn.setGeometry(QtCore.QRect(251,85,77,30))

handType = {"wellHandA" : ['L_shoulder_CON', 'L_IK_upArm_CON', 'L_PV_CON', 'L_IK_hand_HDL_CON'], "wellHandB" : ['R_shoulder_CON', 'R_IK_upArm_CON', 'R_PV_CON', 'R_IK_hand_HDR_CON'], "wellhandC" : ['L_shoulder_CON', 'L_IK_upArm_CON', 'L_PV_CON', 'L_IK_hand_HDL_CON'], "wellhandD" : ['R_shoulder_CON', 'R_IK_upArm_CON', 'R_PV_CON', 'R_IK_hand_HDL_CON']}

def allR():
    # 전체 다 바꾸기
    for i in cmds.ls(type="dxRig"):
        if "targetNode" in cmds.listAttr(i):
            atx = cmds.getAttr(i + ".targetNode")
            num = str(re.search('\d\d\d',str(atx.split("|")[-1])).group())
            cmds.namespace(ren = (i.split(":")[0],i.split(":")[0] + "_" + num))
def selR():
    # 선택한 애들만 바꾸기
    sel = cmds.ls(sl=True)
    for i in sel:
        if "targetNode" in cmds.listAttr(i):
            atx = cmds.getAttr(i + ".targetNode")
            num = str(re.search('\d\d\d',str(atx.split("|")[-1])).group())
            cmds.namespace(ren = (i.split(":")[0],i.split(":")[0] + "_" + num))

def selRoot():
    sel = cmds.ls(sl=True)
    tex = ""
    cmds.select(cl=True)
    for i in sel:
        nsChar = str(i.split(":")[0])
        chr = re.search("[a-zA-Z]+", nsChar).group()
        tex += nsChar + " "
        cmds.select(nsChar + ":" + handType[chr][0], add=True)
    getLE.setText(tex)

def selua():
    sel = cmds.ls(sl=True)
    tex = ""
    cmds.select(cl=True)
    for i in sel:
        nsChar = str(i.split(":")[0])
        chr = re.search("[a-zA-Z]+", nsChar).group()
        tex += nsChar + " "
        cmds.select(nsChar + ":" + handType[chr][1], add=True)
    getLE.setText(tex)

def selpv():
    sel = cmds.ls(sl=True)
    tex = ""
    cmds.select(cl=True)
    for i in sel:
        nsChar = str(i.split(":")[0])
        chr = re.search("[a-zA-Z]+", nsChar).group()
        tex += nsChar + " "
        cmds.select(nsChar + ":" + handType[chr][2], add=True)
    getLE.setText(tex)
    
def selih():
    sel = cmds.ls(sl=True)
    tex = ""
    cmds.select(cl=True)
    for i in sel:
        nsChar = str(i.split(":")[0])
        chr = re.search("[a-zA-Z]+", nsChar).group()
        tex += nsChar + " "
        cmds.select(nsChar + ":" + handType[chr][3], add=True)
    getLE.setText(tex)

allBtn.clicked.connect(allR)
selBtn.clicked.connect(selR)
rtBtn.clicked.connect(selRoot)
uaBtn.clicked.connect(selua)
pvBtn.clicked.connect(selpv)
ihBtn.clicked.connect(selih)
main_layout.show()