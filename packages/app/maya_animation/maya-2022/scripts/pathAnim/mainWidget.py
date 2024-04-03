from PySide2 import QtCore, QtGui, QtWidgets
import maya.cmds as cmds
import maya.mel as mel
import aniCommon
import subprocess
import pathAnim
reload(pathAnim)
import modules;
reload(modules)

class PathAnimWidget(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(PathAnimWidget, self).__init__(parent)
        self.setWindowTitle("Path Anim")
        bar = self.menuBar()
        options = bar.addMenu('Options')
        self.fixRigStretchGuiAction = QtWidgets.QAction('Fix Rig Stretch GUI', self)
        self.footLockerGuiAction = QtWidgets.QAction('Foot Locker GUI', self)
        self.movePathAnim = QtWidgets.QAction('Move Path Anim GUI', self)
        options.addAction(self.fixRigStretchGuiAction)
        options.addAction(self.footLockerGuiAction)
        options.addAction(self.movePathAnim)
        options.addSeparator()

        self.rebuilCrvAction = QtWidgets.QAction('Rebuild/Project Path Curves', self)
        self.resetCvsAction = QtWidgets.QAction('Reset Selected Path CVs', self)
        options.addAction(self.rebuilCrvAction)
        options.addAction(self.resetCvsAction)
        options.addSeparator()

        self.addRetimeAction = QtWidgets.QAction('Time Warp GUI', self)
        options.addAction(self.addRetimeAction)
        options.addSeparator()

        self.hideAnimLocAction = QtWidgets.QAction('Hide Anim Offset(Red) Locators', self)
        self.hideAnimLocAction.setCheckable(True)
        self.hideCtrlLocAction = QtWidgets.QAction('Hide Ctrl Pivot(Yellow) Locators', self)
        self.hideCtrlLocAction.setCheckable(True)
        options.addAction(self.hideAnimLocAction)
        options.addAction(self.hideCtrlLocAction)
        options.addSeparator()

        self.disableViewAction = QtWidgets.QAction('Disable View while Baking (FASTER!)', self)
        self.disableViewAction.setCheckable(True)
        self.disableViewAction.setChecked(True)
        self.restoreViewAction = QtWidgets.QAction('<<Restore the View>>', self)
        options.addAction(self.disableViewAction)
        options.addAction(self.restoreViewAction)
        main_widget = QtWidgets.QWidget(self)
        main_layout = QtWidgets.QVBoxLayout(main_widget)
        main_layout.setContentsMargins(3, 0, 3, 3)
        main_layout.setSpacing(5)
        prefixLayout = QtWidgets.QHBoxLayout()
        self.setCentralWidget(main_widget)
        label_0 = QtWidgets.QLabel('Prefix :')
        label_0.setAlignment(QtCore.Qt.AlignCenter)
        self.prefixLineEdit = QtWidgets.QLineEdit()
        prefixLayout.addWidget(label_0)
        prefixLayout.addWidget(self.prefixLineEdit)     # prefixLayout
        positionLayout = QtWidgets.QHBoxLayout()
        self.axisLabel = QtWidgets.QLabel('Path Direction Axis : ')
        self.axisCB = QtWidgets.QComboBox()
        self.axisCB.addItems(["+z", "-z", "+x", "-x"])
        self.attachCtrl = QtWidgets.QPushButton('Attach')
        positionLayout.addWidget(self.axisLabel)
        positionLayout.addWidget(self.axisCB)
        positionLayout.addWidget(self.attachCtrl)
        self.createPathButton = QtWidgets.QPushButton('Create Path Curves')
        self.scaleLineEdit = QtWidgets.QLineEdit()
        pathFrameLayout = QtWidgets.QHBoxLayout()
        label_1 = QtWidgets.QLabel('Scale Path System :')
        # label_1.setAlignment(QtCore.Qt.AlignCenter)
        pathFrameLayout.addWidget(label_1)
        pathFrameLayout.addWidget(self.scaleLineEdit)
        helpLayout = QtWidgets.QHBoxLayout()
        self.helpBtn = QtWidgets.QPushButton()
        self.helpBtn.setIcon(QtGui.QIcon(QtGui.QPixmap("/netapp/backstage/pub/apps/maya2/versions/2017/team/animation/linux/icons/system-help.png")))
        self.helpBtn.setMinimumSize(QtCore.QSize(24, 24))
        self.helpBtn.setMaximumSize(QtCore.QSize(24, 24))
        self.cbbx = QtWidgets.QComboBox()
        self.cbbx.addItems(["Default Curve", "Custom Curve"])
        helpLayout.addWidget(self.helpBtn)
        helpLayout.addWidget(self.cbbx)
        lineLengthLayout = QtWidgets.QHBoxLayout()
        self.lineLengthlineEdit = QtWidgets.QLineEdit()
        label_1B = QtWidgets.QLabel('Curve Length :')
        self.curveLengthBtn = QtWidgets.QPushButton('OK')
        self.curveLengthBtn.setMinimumSize(QtCore.QSize(38, 23))
        self.curveLengthBtn.setMaximumSize(QtCore.QSize(38, 23))
        lineLengthLayout.addWidget(label_1B)
        lineLengthLayout.addWidget(self.lineLengthlineEdit)
        lineLengthLayout.addWidget(self.curveLengthBtn)
        rebuildLayout = QtWidgets.QHBoxLayout()
        self.rebuildEdit = QtWidgets.QLineEdit()
        label_1C = QtWidgets.QLabel('Curve Rebuild :')
        self.curveRebuild = QtWidgets.QPushButton('OK')
        self.curveRebuild.setMinimumSize(QtCore.QSize(38, 23))
        self.curveRebuild.setMaximumSize(QtCore.QSize(38, 23))
        rebuildLayout.addWidget(label_1C)
        rebuildLayout.addWidget(self.rebuildEdit)
        rebuildLayout.addWidget(self.curveRebuild)  # rebuildLayout
        strLayout = QtWidgets.QHBoxLayout()
        label_str = QtWidgets.QLabel('    CV Position :')
        self.strLE = QtWidgets.QLineEdit()
        self.strBTN = QtWidgets.QPushButton('OK')
        self.strBTN.setMinimumSize(QtCore.QSize(38, 23))
        self.strBTN.setMaximumSize(QtCore.QSize(38, 23))
        strLayout.addWidget(label_str)
        strLayout.addWidget(self.strLE)
        strLayout.addWidget(self.strBTN)    # strLayout

        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.createGroundHookBtn = QtWidgets.QPushButton('Create Ground Hooks for Selected')
        self.createBodyHookBtn = QtWidgets.QPushButton('Create Body Hooks for Selected')
        line_1 = QtWidgets.QFrame()
        line_1.setFrameShape(QtWidgets.QFrame.HLine)
        line_1.setFrameShadow(QtWidgets.QFrame.Sunken)

        label_2 = QtWidgets.QLabel('Loop Frames :')
        label_2.setAlignment(QtCore.Qt.AlignCenter)
        loopFrameLayout = QtWidgets.QHBoxLayout()
        self.loopStartFrameLineEdit = QtWidgets.QLineEdit()
        self.loopEndFrameLineEdit = QtWidgets.QLineEdit()
        loopFrameLayout.addWidget(self.loopStartFrameLineEdit)
        loopFrameLayout.addWidget(self.loopEndFrameLineEdit)

        line_2 = QtWidgets.QFrame()
        line_2.setFrameShape(QtWidgets.QFrame.HLine)
        line_2.setFrameShadow(QtWidgets.QFrame.Sunken)

        useGroundLayout = QtWidgets.QHBoxLayout()
        self.useGroundMesh_checkBox = QtWidgets.QCheckBox('Use Ground Mesh')
        ugLabel = QtWidgets.QLabel('Ground : ')
        self.groundMesh_lineEdit = QtWidgets.QLineEdit()
        self.groundMesh_lineEdit.setEnabled(False)
        self.addGroundMesh_Btn = QtWidgets.QPushButton("<<")
        self.addGroundMesh_Btn.setFixedWidth(30)
        self.addGroundMesh_Btn.setEnabled(False)
        useGroundLayout.addWidget(ugLabel)
        useGroundLayout.addWidget(self.groundMesh_lineEdit)
        useGroundLayout.addWidget(self.addGroundMesh_Btn)
        groundOffsetLayout = QtWidgets.QHBoxLayout()
        offsetLabel = QtWidgets.QLabel('Ground Offset : ')
        offsetLabel.setFixedWidth(90)
        self.groundOffsetLindeEdit = QtWidgets.QLineEdit('0.0')
        self.groundOffsetLindeEdit.setFixedWidth(60)
        self.groundOffsetLindeEdit.setEnabled(False)
        groundOffsetLayout.addWidget(offsetLabel)
        groundOffsetLayout.addWidget(self.groundOffsetLindeEdit)
        groundOffsetLayout.setAlignment(QtCore.Qt.AlignLeft)

        self.attachRigBtn = QtWidgets.QPushButton('Attach Rig To Hooks')
        self.makeCtrl = QtWidgets.QPushButton('Make Controllers')
        self.loadPosition = QtWidgets.QPushButton('Load Position')
        line_3 = QtWidgets.QFrame()
        line_3.setFrameShape(QtWidgets.QFrame.HLine)
        line_3.setFrameShadow(QtWidgets.QFrame.Sunken)

        label_3 = QtWidgets.QLabel('Bake Path Anim To Rig Frames :')
        label_3.setAlignment(QtCore.Qt.AlignCenter)
        bakeFrameLayout = QtWidgets.QHBoxLayout()
        self.bakeStartFrameLineEdit = QtWidgets.QLineEdit()
        self.bakeEndFrameLineEdit = QtWidgets.QLineEdit()
        bakeFrameLayout.addWidget(self.bakeStartFrameLineEdit)
        bakeFrameLayout.addWidget(self.bakeEndFrameLineEdit)
        self.bakeBtn = QtWidgets.QPushButton('Bake Path Anim to Rig')
        self.progressBar = QtWidgets.QProgressBar()

        self.deletePathBtn = QtWidgets.QPushButton('Delete Path Anim System')

        main_layout.addLayout(helpLayout)
        main_layout.addLayout(prefixLayout)
        main_layout.addLayout(positionLayout)
        main_layout.addWidget(self.createPathButton)
        main_layout.addLayout(pathFrameLayout)
        main_layout.addLayout(lineLengthLayout)
        main_layout.addLayout(rebuildLayout)
        main_layout.addLayout(strLayout)
        main_layout.addWidget(line)
        main_layout.addWidget(self.createGroundHookBtn)
        main_layout.addWidget(self.createBodyHookBtn)
        main_layout.addWidget(line_1)
        main_layout.addWidget(label_2)
        main_layout.addLayout(loopFrameLayout)
        main_layout.addWidget(line_2)
        main_layout.addWidget(self.useGroundMesh_checkBox)
        main_layout.addLayout(useGroundLayout)
        main_layout.addLayout(groundOffsetLayout)
        main_layout.addWidget(self.attachRigBtn)
        main_layout.addWidget(self.makeCtrl)
        main_layout.addWidget(self.loadPosition)
        main_layout.addWidget(line_3)
        main_layout.addWidget(label_3)
        main_layout.addLayout(bakeFrameLayout)
        main_layout.addWidget(self.bakeBtn)
        main_layout.addWidget(self.progressBar)
        main_layout.addWidget(self.deletePathBtn)

        self.connectSignals()
        self.initUI()
        self.checkChange()

    def connectSignals(self):
        self.fixRigStretchGuiAction.triggered.connect(pathAnim.showSfUI)
        self.movePathAnim.triggered.connect(pathAnim.showMoveUI)
        self.footLockerGuiAction.triggered.connect(lambda: mel.eval("bh_footLocker;"))
        self.rebuilCrvAction.triggered.connect(pathAnim.showRebuildUI)
        self.resetCvsAction.triggered.connect(self.resetCurve)
        self.addRetimeAction.triggered.connect(pathAnim.showTimewarpUI)
        haoCheckBox = self.hideAnimLocAction
        cplCheckBox = self.hideCtrlLocAction
        self.hideAnimLocAction.toggled.connect(lambda: modules.hideAnimOffsetLocators(haoCheckBox))
        self.hideCtrlLocAction.toggled.connect(lambda: modules.hideCtrlPivotLocators(cplCheckBox))
        self.restoreViewAction.triggered.connect(modules.restoreTheView)
        self.createPathButton.clicked.connect(self.createPath)
        self.curveLengthBtn.clicked.connect(self.mdfLength)
        self.scaleLineEdit.textEdited.connect(self.scaleChanged)
        self.lineLengthlineEdit.textEdited.connect(self.lengthChanged)
        self.useGroundMesh_checkBox.stateChanged.connect(self.useGroundMeshChecked)
        self.addGroundMesh_Btn.clicked.connect(self.addGroundMesh)
        self.createGroundHookBtn.clicked.connect(lambda: self.createHooks("_GroundHook"))
        self.createBodyHookBtn.clicked.connect(lambda: self.createHooks("_BodyHook"))
        self.groundOffsetLindeEdit.textEdited.connect(self.groundOffsetChanged)
        self.attachRigBtn.clicked.connect(self.attachRig)
        self.bakeBtn.clicked.connect(self.bakeRigs)
        self.deletePathBtn.clicked.connect(self.deleteAnimPath)
        self.curveRebuild.clicked.connect(self.crvRebuild)
        self.attachCtrl.clicked.connect(self.atcCurve)
        self.strBTN.clicked.connect(self.trackCv)
        self.cbbx.currentIndexChanged.connect(self.checkChange)
        self.makeCtrl.clicked.connect(self.makeCtrls)
        self.loadPosition.clicked.connect(self.loadPnt)
        self.helpBtn.clicked.connect(self.openML)

    def initUI(self):
        startFrame = cmds.playbackOptions(q=True, min=True)
        endFrame = cmds.playbackOptions(q=True, max=True)
        self.prefixLineEdit.setText("prefix")
        self.scaleLineEdit.setText("1.00")
        self.lineLengthlineEdit.setText("1.0")
        self.loopStartFrameLineEdit.setText(str(startFrame))
        self.loopEndFrameLineEdit.setText(str(endFrame))
        self.bakeStartFrameLineEdit.setText(str(startFrame))
        self.bakeEndFrameLineEdit.setText(str(endFrame))

    def getOptions(self):
        options = dict()
        options['prefix'] = str(self.prefixLineEdit.text())
        options['length'] = float(self.lineLengthlineEdit.text())
        options['scale'] = float(self.scaleLineEdit.text())
        options['loopStartFrame'] = float(self.loopStartFrameLineEdit.text())
        options['loopEndFrame'] = float(self.loopEndFrameLineEdit.text())
        options['bakeStartFrame'] = float(self.bakeStartFrameLineEdit.text())
        options['bakeEndFrame'] = float(self.bakeEndFrameLineEdit.text())
        options['hideOption'] = int(self.disableViewAction.isChecked())
        options['ground'] = str(self.groundMesh_lineEdit.text())
        options['groundOffset'] = float(self.groundOffsetLindeEdit.text())
        return options

    def scaleChanged(self):
        options = self.getOptions()
        prefix = options['prefix']
        scale = options['scale']
        pathAnimCurveGrp = prefix + "_PathAnimCurves"
        cmds.xform(pathAnimCurveGrp, scale=(scale, scale, scale))

    def checkChange(self):
        if str(self.cbbx.currentText()) == "Default Curve":
            self.strLE.setEnabled(False)
            self.strBTN.setEnabled(False)
            self.attachCtrl.setEnabled(False)
            self.axisCB.setEnabled(False)
            self.loadPosition.setEnabled(False)
            self.lineLengthlineEdit.setEnabled(True)
            self.curveLengthBtn.setEnabled(True)
            self.rebuildEdit.setEnabled(True)
            self.curveRebuild.setEnabled(True)
            self.createPathButton.setEnabled(True)
            self.attachCtrl.setText("Attach")
            self.createPathButton.setText("1. Create Path Curves")
            self.curveLengthBtn.setText("2. OK")
            self.curveRebuild.setText("3. OK")
            self.strBTN.setText("OK")
            self.createGroundHookBtn.setText("4. Create Ground Hooks for Selected")
            self.createBodyHookBtn.setText("5. Create Body Hooks for Selected")
            self.attachRigBtn.setText("6. Attach Rig To Hooks")
            self.makeCtrl.setText("7. Make Controllers")
            self.loadPosition.setText("Load Position")
        else:
            self.strLE.setEnabled(True)
            self.strBTN.setEnabled(True)
            self.attachCtrl.setEnabled(True)
            self.axisCB.setEnabled(True)
            self.loadPosition.setEnabled(True)
            self.lineLengthlineEdit.setEnabled(False)
            self.curveLengthBtn.setEnabled(False)
            self.rebuildEdit.setEnabled(True)
            self.curveRebuild.setEnabled(True)
            self.createPathButton.setEnabled(False)
            self.attachCtrl.setText("1. Attach")
            self.curveLengthBtn.setText("OK")
            self.curveRebuild.setText("OK")
            self.strBTN.setText("2. OK")
            self.createGroundHookBtn.setText("3. Create Ground Hooks for Selected")
            self.createBodyHookBtn.setText("4. Create Body Hooks for Selected")
            self.attachRigBtn.setText("5. Attach Rig To Hooks")
            self.makeCtrl.setText("6. Make Controllers")
            self.loadPosition.setText("7. Load Position")

    def lengthChanged(self):
        att = ["translateX", "translateY", "translateZ", "rotateX", "rotateY", "rotateZ", "scaleX", "scaleY", "scaleZ"]
        options = self.getOptions()
        prefix = options['prefix']
        length = options['length']
        curveNameG = prefix + "_groundCurve"
        curveNameB = prefix + "_bodyCurve"
        for i in att:
            cmds.setAttr(curveNameG + "." + i, l=False)
            cmds.setAttr(curveNameB + "." + i, l=False)
        cmds.setAttr(curveNameG + ".scaleZ", length)
        cmds.setAttr(curveNameB + ".scaleZ", length)

    def openML(self):
        exp = "/usr/bin/evince"
        fileName = "/netapp/backstage/pub/apps/maya2/versions/2017/team/animation/linux/scripts/pathAnim/bh_pathAnimML.pdf"
        subprocess.Popen([exp, fileName])

    def groundOffsetChanged(self):
        if not self.groundOffsetLindeEdit.text():
            return
        options = self.getOptions()
        modules.groundOffset(options)


    def useGroundMeshChecked(self):
        state = self.useGroundMesh_checkBox.isChecked()
        self.groundMesh_lineEdit.setEnabled(state)
        self.addGroundMesh_Btn.setEnabled(state)
        self.groundOffsetLindeEdit.setEnabled(state)

    def addGroundMesh(self):
        sel = cmds.ls(sl=True)
        if not sel or len(sel) >= 2:
            raise Exception("Select a ground mesh object")
        self.groundMesh_lineEdit.clear()
        self.groundMesh_lineEdit.setText(str(sel[0]))

    def mdfLength(self):
        att = ["translateX", "translateY", "translateZ", "rotateX", "rotateY", "rotateZ", "scaleX", "scaleY", "scaleZ"]
        options = self.getOptions()
        prefix = options['prefix']
        curveNameG = prefix + "_groundCurve"
        curveNameB = prefix + "_bodyCurve"
        try:
            cmds.makeIdentity(curveNameG, n=0, s=1, r=1, t=1, apply=True, pn=1)
            cmds.makeIdentity(curveNameB, n=0, s=1, r=1, t=1, apply=True, pn=1)
            for i in att:
                cmds.setAttr(curveNameG + "." + i, l=True)
                cmds.setAttr(curveNameB + "." + i, l=True)
        except:
            pass
        spNum = cmds.getAttr(curveNameG + ".spans")
        self.rebuildEdit.setText(str(spNum))

    def crvRebuild(self):
        options = self.getOptions()
        prefix = options['prefix']
        rebNum = int(self.rebuildEdit.text())
        if str(self.cbbx.currentText()) == "Custom Curve":
            sel = str(cmds.ls(sl=True)[0])
            cmds.rebuildCurve(sel, rt=0, end=1, d=3, kr=1, s=rebNum, kcp=0, tol=0.01, kt=0, rpo=1, kep=1)
        else:
            cuvList = ["_groundCurve", "_bodyCurve"]
            for i in cuvList:
                cmds.rebuildCurve(prefix + i, rt=0, end=1, d=3, kr=1, s=rebNum, kcp=0, tol=0.01, kt=0, rpo=1, kep=1)

    def atcCurve(self):
        sel, resCurve = self.curveSet()
        self.savePnt(resCurve)
        self.atctrl(sel, resCurve)
        self.createPath(selCon=sel)

    def curveSet(self):
        axis = str(self.axisCB.currentText())
        sel = cmds.ls(sl=True)
        zeroPosition = cmds.pointPosition(str(sel[-1]) + ".cv[0]")
        nsChar = sel[0].split(":")[0]
        selRoot = str(cmds.listRelatives(nsChar + ":place_CON", p=True, f=True)[0].split("|")[1])
        if axis == "+z":
            cmds.setAttr(str(sel[0]).split(":")[0] + ":place_CON.rotateY", 0)
            bs = cmds.getAttr(selRoot + ".boundingBoxSize")[0]
            newCurveZeroP = [zeroPosition[0], zeroPosition[1], zeroPosition[2] - bs[2] - bs[2] / 10]
        elif axis == "-z":
            cmds.setAttr(str(sel[0]).split(":")[0] + ":place_CON.rotateY", 180)
            bs = cmds.getAttr(selRoot + ".boundingBoxSize")[0]
            newCurveZeroP = [zeroPosition[0], zeroPosition[1], zeroPosition[2] + bs[2] + bs[2] / 10]
        elif axis == "+x":
            cmds.setAttr(str(sel[0]).split(":")[0] + ":place_CON.rotateY", 90)
            bs = cmds.getAttr(selRoot + ".boundingBoxSize")[0]
            newCurveZeroP = [zeroPosition[0] - bs[0] - bs[0] / 10, zeroPosition[1], zeroPosition[2]]
        elif axis == "-x":
            cmds.setAttr(str(sel[0]).split(":")[0] + ":place_CON.rotateY", -90)
            bs = cmds.getAttr(selRoot + ".boundingBoxSize")[0]
            newCurveZeroP = [zeroPosition[0] + bs[0] + bs[0] / 4, zeroPosition[1], zeroPosition[2]]
        newCurve = cmds.curve(p=[newCurveZeroP, zeroPosition], d=1, ws=True)
        cmds.rebuildCurve(newCurve, rt=0, end=1, d=3, kr=1, s=16, kcp=0, tol=0.01, kt=0, rpo=1, kep=1)
        resCurve = str(cmds.attachCurve(newCurve, sel[-1], ch=0, bb=0.5, kmk=1, m=0, bki=0, p=0.1, rpo=0)[0])
        cmds.setAttr(sel[-1] + ".visibility", 0)
        cmds.setAttr(newCurve + ".visibility", 0)
        mel.eval("DeleteHistory;")
        return sel[0], resCurve

    def atctrl(self, selA, selB):
        self.selCv = [selA, selB]
        wspace = cmds.xform(self.selCv[1] + ".controlPoints[0]", t=True, q=True, ws=True)
        cmds.xform(self.selCv[0], t=wspace, ws=True)
        drspace = cmds.xform(self.selCv[1] + ".controlPoints[1]", t=True, q=True, ws=True)
        str(cmds.spaceLocator(n="temptemp_LOC")[0])
        cmds.xform("temptemp_LOC", ws=True, t=drspace)
        temp = str(cmds.aimConstraint("temptemp_LOC", self.selCv[0], weight=1, upVector=(0, 1, 0), worldUpType="vector", offset=(0, 0, 0), aimVector=(0, 0, 1), worldUpVector=(0, 1, 0))[0])
        cmds.delete(temp)
        cmds.delete("temptemp_LOC")
        shpNode = str(cmds.listRelatives(self.selCv[1], c=True, type="shape")[0])
        cmds.setAttr(shpNode + ".dispCV", 1)

    def trackCv(self):
        if str(cmds.ls(sl=True)[0]).count("cv["):
            cvP = str(cmds.ls(sl=True)[0])
            setPoint = int(cvP.split("cv[")[1].split("]")[0]) + 1
            self.strLE.setText(str(setPoint))
        else:
            if self.strLE.text():
                setPoint = int(self.strLE.text()) + 1
            else:
                setPoint = 3
        options = self.getOptions()
        prefix = options['prefix']
        
        cvList = dict()
        for i in range(setPoint + 1):
            getPosition = cmds.pointPosition(self.selCv[1] + ".cv[" + str(i) + "]", l=True)
            cvList[int(i)] = getPosition
        trgPnt = [cvList[setPoint][0], cvList[0][1], cvList[setPoint][2]]
        cmds.xform(self.selCv[1] + ".controlPoints[" + str(setPoint) + "]", t=trgPnt)
        zInd = float((cvList[int(setPoint)][2] - cvList[0][2]) / setPoint)
        for j in range(1, setPoint):
            xpn = cvList[0][0]
            ypn = cvList[0][1]
            zpn = cvList[0][2] + (zInd * j)
            cmds.xform(self.selCv[1] + ".controlPoints[" + str(j) + "]", t=[xpn, ypn, zpn])
        selc = prefix + "_groundCurve"
        seld = prefix + "_bodyCurve"
        spanNum = cmds.getAttr(self.selCv[1] + ".spans")
        cmds.rebuildCurve(selc, rt=0, end=1, d=3, kr=0, s=spanNum, kcp=0, tol=0.01, kt=0, rpo=1, kep=1)
        cmds.rebuildCurve(seld, rt=0, end=1, d=3, kr=0, s=spanNum, kcp=0, tol=0.01, kt=0, rpo=1, kep=1)
        aTr = cmds.xform(selc + ".controlPoints[0]", t=True, q=True, ws=True)
        bTr = cmds.xform(seld + ".controlPoints[0]", t=True, q=True, ws=True)
        Ysub = abs(bTr[1] - aTr[1])
        keyDict = dict()
        for i in range(spanNum + 3):
            getVc = cmds.xform(self.selCv[1] + ".controlPoints[" + str(i) + "]", t=True, q=True, ws=True)
            keyDict[int(i)] = getVc
        for vtx in keyDict.keys():
            cmds.xform(selc + ".controlPoints[" + str(vtx) + "]", t=keyDict[vtx], ws=True)
            cmds.xform(seld + ".controlPoints[" + str(vtx) + "]", t=keyDict[vtx], ws=True)
            cmds.move(0, Ysub, 0, seld + ".controlPoints[" + str(vtx) + "]", r=True)
        if cmds.ls(prefix + "_LAY"):
            cmds.delete(prefix + "_LAY")
        cmds.setAttr(self.selCv[1] + ".visibility", False)

    def makeCtrls(self):
        animPlug = '/usr/autodesk/maya2017/bin/plug-ins/matrixNodes.so'
        if cmds.pluginInfo(animPlug, q=True, l=True) == False:
            try:
                cmds.loadPlugin(animPlug)
                cmds.pluginInfo(animPlug, edit=True, autoload=True)
            except:
                pass
        options = self.getOptions()
        prefix = options['prefix']
        grdCrv = prefix + "_groundCurve"
        bdCrv = prefix + "_bodyCurve"
        Gshp = str(cmds.listRelatives(grdCrv, c=True, type="shape")[0])
        Bshp = str(cmds.listRelatives(bdCrv, c=True, type="shape")[0])
        a = cmds.getAttr(Gshp + '.spans') + 3
        g1 = cmds.group(n=prefix + '_curveF_attach_GRP', em=True)
        g2 = cmds.group(n=prefix + '_curveF_crtl_GRP', em=True)
        ctrList = list()
        for i in range(a):
            b = cmds.xform(str(grdCrv) + '.cv[' + str(i) + ']', q=True, t=True, ws=True)
            sph_ = cmds.curve(p=[(0, 0, 0.5), (0.353554, 0, 0.353554), (0.5, 0, 0), (0.353554, 0, -0.353554), (0, 0, -0.5), (-0.353554, 0, -0.353554), (-0.5, 0, 0), (-0.353554, 0, 0.353554), (0, 0, 0.5), (0, 0.25, 0.433013),(0, 0.433013, 0.25), (0, 0.5, 0), (0, 0.433013, -0.25), (0, 0.25, -0.433013), (0, 0, -0.5), (0, -0.25, -0.433013), (0, -0.433013, -0.25), (0, -0.5, 0), (0, -0.433013, 0.25), (0, -0.25, 0.433013), (0, 0, 0.5), (0.353554, 0, 0.353554), (0.5, 0, 0), (0.433013, 0.25, 0), (0.25, 0.433013, 0), (0, 0.5, 0), (-0.25, 0.433013, 0), (-0.433013, 0.25, 0), (-0.5, 0, 0), (-0.433013, -0.25, 0), (-0.25, -0.433013, 0), (0, -0.5, 0), (0.25, -0.433013, 0), (0.433013, -0.25, 0), (0.5, 0, 0)], d=1)
            sph_nul = cmds.group(sph_)
            loc_ = str(cmds.spaceLocator()[0])
            new_loc = cmds.rename(loc_, prefix + '_pathAttach' + str(i) + '_LOC')
            locShape_ = str(cmds.listRelatives(new_loc, type="shape")[0])
            new_sph = cmds.rename(sph_, prefix + '_pathCurve' + str(i) + '_CON')
            ctrList.append(new_sph)
            new_sphNul = cmds.rename(sph_nul, prefix + '_pathCurve' + str(i) + '_NUL')
            cmds.move(b[0], b[1], b[2], new_loc)
            mtm = str(cmds.shadingNode('multMatrix', asUtility=1))
            cmds.connectAttr(locShape_ + '.worldMatrix[0]', mtm + '.matrixIn[0]', force=1)
            cmds.connectAttr(Gshp + '.parentInverseMatrix[0]', mtm + '.matrixIn[1]', f=1)
            dcm = str(cmds.shadingNode('decomposeMatrix', asUtility=1))
            cmds.connectAttr(mtm + '.matrixSum', dcm + '.inputMatrix', force=1)
            cmds.connectAttr(dcm + '.outputTranslate', Gshp + '.controlPoints[' + str(i) + ']', f=1)
            cmds.move(b[0], b[1], b[2], new_sphNul)
            cmds.parentConstraint(prefix + '_pathCurve' + str(i) + '_CON', new_loc)
            cmds.parent(new_loc, g1)
            cmds.parent(new_sphNul, g2)
            cmds.hide(new_loc)
            cmds.setAttr(new_sph + '.overrideEnabled', 1)
            cmds.setAttr(new_sph + '.overrideColor', 22)
            cmds.cluster(new_sph)
            cmds.delete(new_sph, ch=True)
            Gtr = cmds.xform(Gshp + ".controlPoints[0]", q=True, t=True)[1]
            Btr = cmds.xform(Bshp + ".controlPoints[0]", q=True, t=True)[1]
            sub = Btr - Gtr
            plusNode = str(cmds.shadingNode('plusMinusAverage', asUtility=1))
            cmds.connectAttr(Gshp + '.controlPoints[' + str(i) + ']', plusNode + '.input3D[0]', f=1)
            cmds.setAttr(plusNode + ".input3D[1].input3Dy", sub)
            cmds.connectAttr(plusNode + '.output3D', Bshp + '.controlPoints[' + str(i) + ']', f=1)
        cmds.addAttr(g2, ln="Controller_Scale", at="short")
        for j in ctrList:
            cmds.connectAttr(g2 + ".Controller_Scale", j + ".scaleX")
            cmds.connectAttr(g2 + ".Controller_Scale", j + ".scaleY")
            cmds.connectAttr(g2 + ".Controller_Scale", j + ".scaleZ")
        cmds.setAttr(g2 + ".Controller_Scale", 7)
        attLst = ["translateX", "translateY", "translateZ", "rotateX", "rotateY", "rotateZ", "scaleX", "scaleY", "scaleZ", "visibility"]
        for k in attLst:
            cmds.setAttr(g2 + '.' + k, k=False)
        cmds.setAttr(g2 + '.Controller_Scale', k=True)

    def savePnt(self, sel):
        self.orPs = cmds.pointPosition(sel + ".cv[0]", w=True)
        cmds.xform(sel, ws=True, piv=(self.orPs[0], self.orPs[1], self.orPs[2]))
        self.svloc = str(cmds.spaceLocator(n=sel + "_SavePoint_LOC")[0])
        cmds.xform(self.svloc, ws=True, t=self.orPs)
        self.rtSave = cmds.xform(sel, q=True, ro=True)
        self.trloc = str(cmds.spaceLocator(n=sel + "_ZeroPoint_LOC")[0])
        self.temps = str(cmds.parentConstraint(self.trloc, sel, mo=False, w=1)[0])
        spNum = cmds.getAttr(sel + ".spans")
        self.rebuildEdit.setText(str(spNum))

    def loadPnt(self):
        options = self.getOptions()
        prefix = options['prefix']
        en = str(cmds.parentConstraint(prefix + "_curveF_crtl_GRP", prefix + "_PathAnimCurves", w=1, mo=True)[0])
        wn = str(cmds.parentConstraint(prefix + "_curveF_crtl_GRP", self.selCv[0], w=1, mo=True)[0])
        cmds.xform(prefix + "_curveF_crtl_GRP", ws=True, t=self.orPs)
        cmds.xform(prefix + "_curveF_crtl_GRP", ws=True, ro=self.rtSave)
        cmds.delete(self.temps)
        cmds.delete(en)
        cmds.delete(wn)
        cmds.delete(self.svloc)
        cmds.delete(self.trloc)

    @aniCommon.undo
    def resetCurve(self):
        sel = cmds.ls(sl=True)
        pathAnim.rebuildCurve.resetCurvePoints(sel)

    @aniCommon.undo
    def createPath(self, selCon=None):
        if selCon == None:
            selCon = str(cmds.ls(sl=True)[0])
        else:
            pass
        options = self.getOptions()
        cls = modules.CreatePathCurves(options)
        cls.create()
        prefix = options['prefix']
        groundCurveName = prefix + "_groundCurve"
        getScale = cmds.getAttr(groundCurveName + ".scaleZ")
        self.lineLengthlineEdit.setText(str(getScale))
        if selCon:
            if selCon.count(":"):
                curveGrp = prefix + "_PathAnimCurves"
                nsChar = selCon.split(":")[0]
                mvCon = nsChar + ":move_CON"
                tempConst = cmds.parentConstraint(mvCon, curveGrp, w=True, mo=False, st="none", sr="none")[0]
                cmds.delete(tempConst)
            else:
                curveGrp = prefix + "_PathAnimCurves"
                tempConst = cmds.parentConstraint(selCon, curveGrp, w=True, mo=False, st="none", sr="none")[0]
                cmds.delete(tempConst)
        else:
            pass
        cmds.select(cl=True)
        if str(self.cbbx.currentText()) == "Custom Curve":
            gndC = prefix + "_groundCurve"
            bodC = prefix + "_bodyCurve"
            if len(cmds.ls(prefix + "_LAY")) == 0:
                cmds.select(gndC, bodC)
                cmds.createDisplayLayer(name=prefix + "_LAY", number=1, nr=True)
                mel.eval("layerEditorLayerButtonTypeChange %s_LAY;" % prefix)
                cmds.select(cl=True)

    @aniCommon.undo
    def createHooks(self, part):
        options = self.getOptions()
        options.update({'groundOrBody': part})
        cls = modules.CreateHooks(options)
        cls.createAHook()

    @aniCommon.undo
    def attachRig(self):
        options = self.getOptions()
        cls = modules.AttachToHooks(options)
        cls.attach()

    @aniCommon.undo
    def bakeRigs(self):
        options = self.getOptions()
        cls = modules.bakePathAnimToRigProc(options=options, parent=self)
        cls.bake()

    @aniCommon.undo
    def deleteAnimPath(self):
        options = self.getOptions()
        modules.deletePathAnim(options['prefix'])
