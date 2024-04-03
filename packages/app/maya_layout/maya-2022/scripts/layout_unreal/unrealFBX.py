# -*- coding: utf-8 -*-
####################################################
########## coding by RND youkyoung.kim #############
####################################################
import os, sys, subprocess, json
import Qt
from Qt import QtWidgets, QtGui, QtCore
from Qt.QtGui import QClipboard
from fbx_ui import Ui_Form
from maya import cmds as cmds
from maya import mel as mel

CURRENTPATH = os.path.dirname(os.path.abspath(__file__))
# Maya PyQt4 or PySide2
if "Side" in Qt.__binding__:
    import maya.OpenMayaUI as mui

    if Qt.__qt_version__ > "5.0.0":
        import shiboken2 as shiboken
    else:
        import shiboken as shiboken


    def getMayaWindow():
        ptr = mui.MQtUtil.mainWindow()
        return shiboken.wrapInstance(long(ptr), QtWidgets.QWidget)


    def main():
        mainVar = UnrealFbxExport(getMayaWindow())
        mainVar.move(QtWidgets.QDesktopWidget().availableGeometry().center() - mainVar.frameGeometry().center())
        mainVar.show()

if __name__ == "__main__":
    main()


class UnrealFbxExport(QtWidgets.QWidget):
    def __init__(self, parent):
        QtWidgets.QWidget.__init__(self, parent)
        self.setWindowFlags(QtCore.Qt.Window)
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.setWindow()
        self.loadFbxSet()
        self.connectMove()

    def setWindow(self):
        self.bakeanimation = {}
        self.fbxnonanimation = []
        self.bakecam = {}
        self.animation_ck = False
        self.nonanimation_ck = False
        self.camerack = False

        # shotsequence frame range get
        self.framestart, self.frameend = self.frameRange()

        self.ui.animation_ck.setChecked(self.animation_ck)
        self.ui.nonanimation_ck.setChecked(self.nonanimation_ck)
        self.ui.camera_ck.setChecked(self.camerack)
        self.ui.animation_selbtn.setEnabled(self.animation_ck)
        self.ui.nonanimation_selbtn.setEnabled(self.nonanimation_ck)
        self.ui.camera_selbtn.setEnabled(self.camerack)
        self.ui.shot_ck.setChecked(True)
        self.ui.scale_ck.setChecked(True)
        self.cssSetting()
        self.outputDirSet()

    def cssSetting(self):
        # open dir toolbutton image set and treewidget column set
        self.ui.open_btn.setStyleSheet(
            "QToolButton#open_btn {background-color: #666; border: 1px solid #999;}"
            "QToolButton#open_btn:hover {background-color: #999; border: 1px solid #999;}"
        )
        self.ui.open_btn.setIcon(
            QtGui.QIcon(QtGui.QPixmap(os.path.join(CURRENTPATH, "src/open.png"))))
        self.ui.animation_tree.setColumnWidth(0, 200)
        self.ui.nonanimation_tree.setColumnWidth(0, 200)
        self.ui.camera_tree.setColumnWidth(0, 200)

    def outputDirSet(self):
        # history output directory get
        self.clipb = QtWidgets.QApplication.clipboard()
        self.outputpath = CURRENTPATH
        dir = self.clipb.text()

        if self.clipb.text():
            if os.path.isdir(dir):
                self.outputpath = self.clipb.text()
        self.ui.dir_txt.setText(self.outputpath)

    def loadFbxSet(self):
        # fbx option setting
        mel.eval('FBXExportSmoothingGroups -v true')
        mel.eval('FBXExportSmoothMesh -v true')
        mel.eval('FBXExportReferencedAssetsContent -v true')

        mel.eval('FBXExportCameras -v true')
        mel.eval('FBXExportLights -v true')

        mel.eval('FBXExportInputConnections -v true')

        mel.eval('FBXExportGenerateLog -v true')
        mel.eval('FBXExportScaleFactor 1.0')
        mel.eval('FBXExportUpAxis y')  # z

        mel.eval('FBXExportBakeComplexAnimation -v true')
        mel.eval('FBXExportShapes -v true')
        mel.eval('FBXExportSkins -v true')
        # mel.eval('FBXExportDxfDeformation, 'true')# linux, mac only
        # mel.eval('FBXExportAnimationOnly -q')  # only animation

    def connectMove(self):
        self.ui.animation_ck.stateChanged.connect(self.animationCk)
        self.ui.nonanimation_ck.stateChanged.connect(self.nonanimationCk)
        self.ui.camera_ck.stateChanged.connect(self.cameraCk)
        self.ui.animation_selbtn.clicked.connect(self.selectAni)
        self.ui.nonanimation_selbtn.clicked.connect(self.selectNonAni)
        self.ui.camera_selbtn.clicked.connect(self.selectCamera)
        self.ui.open_btn.clicked.connect(self.openDialog)
        self.ui.fbx_btn.clicked.connect(self.fbxSelect)

    def animationCk(self):
        # animation check button check = True : select button enabled = True
        # animation check button check = False : select button enabled = False, treewidget data clear
        self.animation_ck = self.ui.animation_ck.isChecked()
        if self.animation_ck:
            self.ui.animation_selbtn.setEnabled(True)
        else:
            self.ui.animation_selbtn.setEnabled(False)
            if self.bakeanimation:
                self.removePrint(self.ui.animation_tree, self.bakeanimation)

    def nonanimationCk(self):
        self.nonanimation_ck = self.ui.nonanimation_ck.isChecked()
        if self.nonanimation_ck:
            self.ui.nonanimation_selbtn.setEnabled(True)
        else:
            self.ui.nonanimation_selbtn.setEnabled(False)
            if self.fbxnonanimation:
                self.removePrint(self.ui.nonanimation_tree, self.fbxnonanimation)

    def cameraCk(self):
        # camera check button check = True : select button enabled = True
        # camera check button check = False : select button enabled = False, treewidget data clear
        self.camerack = self.ui.camera_ck.isChecked()
        if self.camerack:
            self.ui.camera_selbtn.setEnabled(True)
        else:
            self.ui.camera_selbtn.setEnabled(False)
            if self.bakecam:
                self.removePrint(self.ui.camera_tree, self.bakecam)

    def removePrint(self, object=None, clearnode=None):
        # treewidget data clear
        total = len(clearnode)
        clearnode = {}
        for i in range(total):
            object.takeTopLevelItem(0)

    def outputPrint(self, object=None, inputdata=None, align=None):
        # treewidget data print
        print inputdata
        for i in inputdata:
            num = len(inputdata[i])
            if num > 4:  ### camera list
                num = 4
            item = Treeitem(object, num, align)
            for j in range(num):
                item.setText(j, str(inputdata[i][j]))
        object.sortByColumn(num - 1, QtCore.Qt.AscendingOrder)

    def frameRange(self):
        # shotsequence exists check >> playback slider setting
        shots = sorted(cmds.ls(type='shot'))
        minlist = []
        maxlist = []
        if shots:
            for i in shots:
                start = int(cmds.getAttr(i + '.startFrame'))
                end = int(cmds.getAttr(i + '.endFrame'))
                minlist.append(start)
                maxlist.append(end)
            start = min(set(minlist))
            end = max(set(maxlist))
            cmds.playbackOptions(ast=start, min=start, aet=end, max=end)

        start = int(cmds.playbackOptions(q=True, ast=True))
        end = int(cmds.playbackOptions(q=True, aet=True))
        return start, end + 1

    #############################################################################################################
    # animation node select
    #############################################################################################################
    def selectAni(self):
        # select button clicked >> treewidget data none >> select node list get
        btnactive = int(self.ui.animation_tree.topLevelItemCount())
        if btnactive == 0:
            selectnode = cmds.ls(sl=True)
            if selectnode:
                cmds.select(cl=True)
                parentnode = cmds.listRelatives(selectnode)
                self.findNode(parentnode)
            else:
                self.warningDialog('char_GRP >> animation node >> [model_GRP or mod_GRP, skinJoint_GRP]')

    def findNode(self, parentnode=None):
        # select charecter group node >> get child node lists >> call modskinNode()
        # fbx file name query >> reference file ':' split
        nodeprint = {}
        for i, j in enumerate(parentnode):
            nodes = []
            cklist = cmds.ls(j, dag=True)
            start = self.framestart
            end = self.frameend
            fbxtagname = '_' + str(start).zfill(4) + '_' + str(end).zfill(4)
            fbxname = self.fbxAninameFix(j, i) + fbxtagname
            nodes = self.modskinNode(cklist)

            if nodes:
                nodes.append(fbxname)
                self.bakeanimation[j] = nodes
                nodeprint[j] = [j, start, end, fbxname]
        self.outputPrint(self.ui.animation_tree, nodeprint, 'left')

    def fbxAninameFix(self, node=None, num=None):
        # animation fbx file name query
        num = str(num).zfill(3)
        if node.endswith('_GRP'):
            name = node.rstrip('_GRP')
        else:
            name = node

        if ":" in node:
            fbxname = num + '_' + name.split(":")[1]
        else:
            fbxname = num + '_' + str(name)
        return fbxname

    def modskinNode(self, cklist=None):
        #  [mod_GRP, skinJoint_GRP] node find >> node name return
        nodes = []
        for k in cklist:
            nodefind = k.upper()
            if nodefind.endswith('MOD_GRP') or nodefind.endswith('MODEL_GRP'):
                # advanced skeleton joint, nonjoint model
                nodes.append(k)
            elif nodefind.endswith('SKINJOINT_GRP'):
                nodes.append(k)

            if nodefind.endswith('DEFORMATIONSYSTEM'):
                nodes.append(k)
            elif nodefind.endswith('GEOMETRY'):
                nodes.append(k)
        if nodes:
            return nodes

    def bakeCharac(self, selectnode=None):
        # character bake
        start = self.framestart
        end = self.frameend
        if len(selectnode) == 2:
            model = selectnode[0]
            skinjoint = selectnode[1]
            bakenode = "\"%s\", \"%s\"" % (model, skinjoint)
        else:
            model = selectnode[0]
            bakenode = "\"%s\"" % model

        bake = 'bakeResults -simulation true -t \"%s:%s\" -hierarchy below -sampleBy 1 ' \
               '-oversamplingRate 1 -disableImplicitControl true -preserveOutsideKeys true ' \
               '-sparseAnimCurveBake false -removeBakedAttributeFromLayer false -removeBakedAnimFromLayer false ' \
               '-bakeOnOverrideLayer false -minimizeRotation true -controlPoints false -shape true {%s};' \
               % (start, end, bakenode)
        mel.eval(bake)

    #############################################################################################################
    # Nonanimation node select
    #############################################################################################################
    def selectNonAni(self):
        # select button clicked >> treewidget data none >> select node list get
        btnactive = int(self.ui.nonanimation_tree.topLevelItemCount())
        if btnactive == 0:
            selectnode = cmds.ls(sl=True)
            if selectnode:
                cmds.select(cl=True)
                parentnode = cmds.listRelatives(selectnode)
                self.assetNode(parentnode)
            else:
                self.warningDialog('Please dhoose first : NonAnimation Group Select !!')

    def assetNode(self, assetnode=None):
        nodeprint = {}
        for i, j in enumerate(assetnode):
            nodefind = j.upper()
            # if nodefind.endswith('_ASSET'):
            nodeprint[j] = [j, j]
            self.fbxnonanimation.append(j)
        self.outputPrint(self.ui.nonanimation_tree, nodeprint, 'left')

    #############################################################################################################
    # camera node select
    #############################################################################################################
    def selectCamera(self):
        # select camera node >> shot check : get shot camera list, shot noncheck : get select camera list
        # camera strat, end frame >> camera bake >> camera treewidget print
        btnactive = int(self.ui.camera_tree.topLevelItemCount())
        if btnactive == 0:
            shots = sorted(cmds.ls(type='shot'))
            cameralist = sorted(cmds.ls(sl=True))
            shotck = self.ui.shot_ck.isChecked()
            inputdata = {}
            if shots or cameralist:
                if shots and shotck:
                    for i, j in enumerate(shots):
                        cameras = cmds.shot(j, q=True, cc=True)
                        cameras = self.cameraType(cameras)
                        start = cmds.getAttr(j + '.startFrame')
                        end = cmds.getAttr(j + '.endFrame') + 1
                        scale = 1 / (cmds.shot(j, q=True, s=True))  # unreal scale
                        sequencestart = cmds.shot(j, q=True, sst=True)
                        sequenceend = cmds.shot(j, q=True, set=True)
                        num = i
                        if scale == 1:
                            fbxtagname = ''
                        else:
                            fbxtagname = '_%0.3f' % scale
                        fbxname = j + fbxtagname
                        inputdata[sequencestart] = [cameras, start, end, fbxname,
                                                    sequencestart, sequenceend, scale]
                        self.bakecam[sequencestart] = inputdata[sequencestart]
                    inputdata = self.sortSequence(inputdata)

                elif cameralist and not shotck:
                    start = self.framestart
                    end = self.frameend
                    for i, j in enumerate(cameralist):
                        fbxname = j
                        inputdata[i] = [j, start, end, fbxname]
                        self.bakecam[i] = inputdata[i]
                else:
                    self.warningDialog('Select camera None')
                self.outputPrint(self.ui.camera_tree, inputdata, 'left')
            else:
                self.warningDialog('Select camera nodes >> C001, C002......')

    def sortSequence(self, inputdata=None):
        # sequence time >> sort
        datakeys = []
        datakeys = sorted(inputdata.keys())
        for i in range(len(inputdata)):
            num = datakeys[i]
            fbxtagname = inputdata[num][3]
            fbxtagname = 'C' + str(i + 1).zfill(3) + '_' + fbxtagname
            inputdata[num][3] = fbxtagname
        return inputdata

    def cameraType(self, cameras=None):
        # camera name type >> transform or camera shape check
        types = cmds.nodeType(cameras)
        if types == 'camera':
            return cmds.listRelatives(cameras, parent=True)[0]
        else:
            return cameras

    def bakeCamera(self, camera=None, start=None, end=None, namespace=None):
        # new camera >> attribute copy >> constrain bake >> attribute lock
        attr = ["horizontalFilmAperture", "verticalFilmAperture", "filmFit", "lensSqueezeRatio",
                "horizontalFilmOffset", "verticalFilmOffset", "preScale", "postScale", "shutterAngle", "fStop"]

        orgcam = cmds.ls(camera, dag=True)  # [u'C0010', u'C0010|C0001']
        orgcamname = orgcam[0]
        orgcamshape = orgcam[1]

        bakename = "%s%s" % (namespace, camera) # bakecams = cmds.duplicate(name=bakename, rr=True)
        bakecam = cmds.camera(name=bakename)
        bakecamname = bakecam[0]
        bakecamshape = bakecam[1]

        self.setCamera(orgcamname, bakecamname, ["rotateOrder"], 1)
        self.setCamera(orgcamshape, bakecamshape, ["focalLength", "nearClipPlane", "farClipPlane"], 0)
        self.setCamera(orgcamshape, bakecamshape, attr, 1)

        self.connectCamera(orgcamshape, bakecamshape)
        self.constrainCamera(orgcamname, bakecamname, bakecamshape, start, end)
        print '+++++++++++++++++++++++++++++++++++++++++++++++++++'
        print orgcamname, bakecamname, bakecamshape
        print '+++++++++++++++++++++++++++++++++++++++++++++++++++'
        self.unlockCamera(bakecamname, bakecamshape) # unlock bake ok
        return bakecamname

    def setCamera(self, getcam=None, setcam=None, attr=None, locktoggle=None):
        # select camera get attribute >> bake camera set attribute
        for i in attr:
            getattr = "%s.%s" % (getcam, i)
            setattr = "%s.%s" % (setcam, i)
            cmds.setAttr(setattr, cmds.getAttr(getattr), lock=locktoggle)

    def connectCamera(self, orgcam=None, bakecam=None):
        # camera connect node check >> backe camera connect
        connectlist = cmds.listConnections(orgcam)
        if connectlist:
            count = 0;
            for i in connectlist:
                if (cmds.nodeType(i) == "imagePlane"):
                    cmds.connectAttr(i + ".message", bakecam + ".imagePlane[%d]" % count)
                    count += 1
                elif (cmds.nodeType(i) == "animCurveTU"):
                    if "focalLength" in i:
                        cmds.connectAttr(i + ".output", bakecam + ".focalLength")

    def constrainCamera(self, orgcamname=None, bakecamname=None, bakecamshape=None, start=None, end=None):
        # camera bake
        constinfo = cmds.parentConstraint(orgcamname, bakecamname, weight=1, maintainOffset=0,
                                          skipTranslate="none", skipRotate="none")
        cmds.bakeResults(bakecamname, t=(start, end), pok=True, sm=True, at=["tx", "ty", "tz", "rx", "ry", "rz"])
        cmds.bakeResults(bakecamshape, t=(start, end), pok=True, sm=True, at=["fl"])
        cmds.delete(constinfo)

    def unlockCamera(self, bakecamname=None, bakecamshape=None):
        # bake camera attribute lock
        cmds.setAttr(bakecamname + ".translate", lock=False)
        cmds.setAttr(bakecamname + ".rotate", lock=False)
        cmds.setAttr(bakecamshape + ".focalLength", lock=False)

    #############################################################################################################
    # fbx export button click
    #############################################################################################################
    def fbxSelect(self):
        self.loadFbxSet()
        if self.animation_ck:
            self.aniExport()

        if self.camerack:
            self.cameraExport()

        if self.nonanimation_ck:
            self.nonAniExport()

    def nonAniExport(self):
        # non animation >> asset node fbx export
        if self.fbxnonanimation:
            for i in self.fbxnonanimation:
                cmds.select(i)
                fbxname = str(i)
                self.fbxExport(fbxname)

    def aniExport(self):
        # animation mod, skinjoint select >> bake >> fbx export
        if self.bakeanimation:
            for i in self.bakeanimation:
                total = len(self.bakeanimation[i])
                selectnode = []
                model = self.bakeanimation[i][0]
                selectnode.append(model)
                fbxname = self.bakeanimation[i][-1]
                if total == 3:
                    skinjoint = self.bakeanimation[i][1]
                    selectnode.append(skinjoint)
                cmds.select(selectnode)
                self.bakeCharac(selectnode)
                self.fbxExport(fbxname)

    def cameraExport(self):
        # camera bake
        self.bakecamlists = []
        if self.bakecam:
            for i in self.bakecam:
                camera = self.bakecam[i][0]
                start = self.bakecam[i][1]
                end = self.bakecam[i][2]
                fbxname = self.bakecam[i][3]
                namespace = ''
                bakename = self.bakeCamera(camera, start, end, namespace)
                self.bakecamlists.append(bakename)
                if not self.ui.scale_ck.isChecked():#1배 카메라 베이크
                    cmds.select(bakename)
                    self.fbxExport(fbxname)
            if self.ui.scale_ck.isChecked():#10배 카메라 베이크
                self.bakeGroup()
            print"=========================================================="
            self.jsonWrite(self.bakecam)  # json data output

    def bakeGroup(self):
        # camera bake group and scale X 10 >> bake again >> fbx export >> json write
        self.bakegrouplists = []
        groupname = 'bakecam'
        cmds.group(em=True, n=groupname)
        for i in self.bakecamlists:
            cmds.parent(i, groupname)
        cmds.select(groupname)
        cmds.setAttr(groupname + '.scaleX', 10.0)
        cmds.setAttr(groupname + '.scaleY', 10.0)
        cmds.setAttr(groupname + '.scaleZ', 10.0)

        for i, j in enumerate(self.bakecam):
            camera = self.bakecamlists[i]
            start = self.bakecam[j][1]
            end = self.bakecam[j][2]
            fbxname = self.bakecam[j][3]
            namespace = fbxname.split('_')[0] + '_'
            bakename = self.bakeCamera(camera, start, end, namespace)
            self.bakegrouplists.append(bakename)
            cmds.select(bakename)
            self.fbxExport(fbxname)

    def jsonWrite(self, data=None):
        # camera json write >> unreal sequence value
        try:
            jsonlists = {}
            fpsmodel = cmds.currentUnit(q=True, t=True)
            if fpsmodel == "film":
                fps = 24
            elif fpsmodel == "ntsc":
                fps = 30
            elif fpsmodel == "ntscf":
                fps = 60
            elif fpsmodel == "100fps":
                fps = 100

            for k, i in enumerate(data):
                lists = {}
                fbx = data[i][3]
                camera = self.bakecamlists[k]
                camshape = cmds.ls(camera, dag=True)[1] #cmds.listRelatives(camera, shapes=True)[0]
                focal = cmds.getAttr(camshape + '.focalLength')
                vfilm = cmds.getAttr(camshape + '.verticalFilmAperture')
                hfilm = cmds.getAttr(camshape + '.horizontalFilmAperture')
                lists['bakename'] = self.bakegrouplists[k]
                lists['fps'] = fps
                lists['start'] = data[i][1]
                lists['end'] = data[i][2]
                lists['fbx'] = fbx
                lists['sequencestart'] = data[i][4]
                lists['sequenceend'] = data[i][5]
                lists['scale'] = data[i][6]
                lists['focal'] = focal
                lists['vfilm'] = vfilm * 25.4  # inch >> mm
                lists['hfilm'] = hfilm * 25.4
                jsonlists[fbx] = lists
            jsondata = json.dumps(jsonlists, indent=4)
            output_json = '%s/camerafbx.json' % (self.outputpath)
            f = open(output_json, 'w')
            f.write(jsondata)
            f.close()
            self.warningDialog('json file write success !!!!')
        except:
            self.warningDialog('not shot sequencer : json file not create !')
            pass

    def fbxExport(self, fbxname=None):
        # fbx export
        output_fbx = '%s/%s.fbx' % (self.outputpath, fbxname)
        output = 'file -force -options \"v=0;\" -typ \"FBX export\" -pr -es  \"%s\";' % (output_fbx)
        mel.eval(output)
        cmds.select(cl=True)

    def isolatePanel(self, toggleset=None):
        # isolate panel view toggle
        panname = cmds.getPanel(type='modelPanel')
        for i in panname:
            cmds.isolateSelect(i, state=toggleset)

    def openDialog(self):
        # open button click >> directory select window dialog open
        self.outputpath = QtWidgets.QFileDialog.getExistingDirectory(self,
                                                                     "Select Output FBX Directory", self.outputpath)
        self.ui.dir_txt.setText(self.outputpath)
        self.clipb.setText(self.outputpath, mode=self.clipb.Clipboard)

    def warningDialog(self, messages=None):
        # warning dialog open
        cmds.confirmDialog(title='Warning !!', message=messages,
                           messageAlign='center', icon='warning',
                           button='ok')


class Treeitem(QtWidgets.QTreeWidgetItem):
    def __init__(self, parent, num, align):
        super(Treeitem, self).__init__(parent)
        font = Qt.QtGui.QFont()
        font.setPointSize(11)

        for i in range(num):
            self.setFont(i, font)
            if align == 'center':
                self.setTextAlignment(i, QtCore.Qt.AlignCenter)
            elif align == 'left':
                self.setTextAlignment(i, QtCore.Qt.AlignLeft)
            elif align == 'right':
                self.setTextAlignment(i, QtCore.Qt.AlignRight)

