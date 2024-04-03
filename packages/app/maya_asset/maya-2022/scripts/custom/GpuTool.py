from GpuToolUI import Ui_Form
import os
from PySide2 import QtWidgets
from PySide2 import QtCore
import maya.cmds as cmds
import sgComponent as sgc
import maya.mel as mel
import random
import dxCommon

class Asset_gpuControlTool_v01(QtWidgets.QWidget):

    def __init__(self,parent = dxCommon.getMayaWindow()):
        QtWidgets.QWidget.__init__(self, parent)
        self.setWindowFlags(QtCore.Qt.Window)

        self.ui = Ui_Form()
        self.ui.setupUi(self)

        # Layout Export Event
        self.ui.pushButton_2.clicked.connect(self.set_selectedExportMode)
        self.ui.pushButton_3.clicked.connect(self.set_ExportMode)

        # connect GPU Event
        self.ui.pushButton_5.clicked.connect(self.connect_gpu)

        # Path Edit Event

        self.ui.comboBox.currentIndexChanged.connect(self.currentIndexChanged)
        self.ui.pushButton.clicked.connect(self.path_change)
        self.ui.pushButton_4.clicked.connect(self.path_selectedChange)

        # multi export Event
        self.ui.pushButton_14.clicked.connect(self.multiExport)

        #gpu mode change
        self.ui.comboBox_3.currentIndexChanged.connect(self.gpuModeChange)
        self.ui.comboBox_4.currentIndexChanged.connect(self.gpuModeChange)
        self.ui.pushButton_6.clicked.connect(self.gpuReload)

        #create Group
        self.ui.pushButton_9.clicked.connect(self.createGroup)

        #random color Assign
        self.ui.pushButton_8.clicked.connect(self.lambertRandomAssign)
        self.ui.pushButton_lambert1.clicked.connect(self.lambert1Assign)

        #color Assign
        self.ui.pushButton_RED.clicked.connect(self.redColorAssign)
        self.ui.pushButton_Orange.clicked.connect(self.orangeColorAssign)
        self.ui.pushButton_Yellow.clicked.connect(self.yellowColorAssign)
        self.ui.pushButton_YellowGreen.clicked.connect(self.yellowGreenColorAssign)
        self.ui.pushButton_Green.clicked.connect(self.greenColorAssign)
        self.ui.pushButton_Sky.clicked.connect(self.skyColorAssign)
        self.ui.pushButton_Blue.clicked.connect(self.blueColorAssign)
        self.ui.pushButton_Purple.clicked.connect(self.purpleColorAssign)
        self.ui.pushButton_Purple_2.clicked.connect(self.magentaColorAssign)
        self.ui.pushButton_Trans.clicked.connect(self.transColorAssign)



    # Selected GPU Mode Changer for ASB exporting
    def set_selectedExportMode(self):

        sel = cmds.ls(sl=1, type='dxComponent')
        print sel

        for i in sel:
            # action
            cmds.setAttr('%s.action' % i, 2)
            # mode
            cmds.setAttr('%s.mode' % i, 1)
            # display
            cmds.setAttr('%s.display' % i, 3)
            # objectInstance
            cmds.setAttr('%s.objectInstance' % i, 1)
            # reload
            sgc.componentReload(i)

        selCreator = cmds.ls(type='ZGpuMeshCreator')
        print selCreator

        for sc in selCreator:
            selConnection = cmds.listConnections(sc, source=False, destination=True)
            print selConnection
            print len(selConnection)

            if len(selConnection) == 1:
                for model in selConnection:
                    cmds.setAttr('%s.objectInstance' % model, 0)

            else:
                pass


    # GPU Mode Changer for ASB exporting
    def set_ExportMode(self):

        sel = cmds.ls(type='dxComponent')
        print sel

        for i in sel:
            # action
            cmds.setAttr('%s.action' % i, 2)
            # mode
            cmds.setAttr('%s.mode' % i, 1)
            # display
            cmds.setAttr('%s.display' % i, 3)
            # objectInstance
            cmds.setAttr('%s.objectInstance' % i, 1)
            # reload
            sgc.componentReload(i)

        selCreator = cmds.ls(type='ZGpuMeshCreator')
        print selCreator

        for sc in selCreator:
            selConnection = cmds.listConnections(sc, source=False, destination=True)
            print selConnection
            print len(selConnection)

            if len(selConnection) == 1:
                for model in selConnection:
                    cmds.setAttr('%s.objectInstance' % model, 0)

            else:
                pass

    # connect GPU
    def connect_gpu(self):
        sel = cmds.ls(sl=1, type='dxComponent')
        print sel
        SELcreator = cmds.listConnections('%s.input' % sel[0])
        print SELcreator
        print sel[1:]
        for R in cmds.ls(sel[1:], dag=True, typ='ZGpuMeshShape'):

            A = cmds.getAttr('%s.input' % R, se=1)
            if A == False:
                pass
            else:
                cmds.connectAttr('%s.output' % SELcreator[0], '%s.input' % R)

    # Path Edit
    def currentIndexChanged(self, text):
        print self.ui.comboBox.currentText()

    def path_change(self):

        old_path = self.ui.lineEdit.text()
        new_path = self.ui.lineEdit_2.text()

        print old_path
        print new_path

        if self.ui.comboBox.currentText() == 'gpu':
            print 'gpu yes'
            for node in cmds.ls(type='dxComponent'):
                fileName = cmds.getAttr('%s.abcFileName' % node)
                newFileName = fileName.replace(old_path, new_path)
                print newFileName
                cmds.setAttr('%s.abcFileName' % node, newFileName, type='string')

                sgc.componentImport("%s.abcFileName" % node, newFileName)

                sgc.componentReload("%s." % node)
        if self.ui.comboBox.currentText() == 'ZEnv':
            print 'ZEnv yes'
            for node in cmds.ls(type='ZEnvSource'):
                fileName = cmds.getAttr('%s.assetPath' % node)
                newFileName = fileName.replace(old_path, new_path)
                print newFileName
                cmds.setAttr('%s.assetPath' % node, newFileName, type='string')

        if self.ui.comboBox.currentText() == 'ZEnv_MAP':
            print 'ZEnv_MAP yes'
            for ZEnvPointSetNode in cmds.ls(type='ZEnvPointSet'):
                densityMapPath = cmds.getAttr('%s.densityMap' % ZEnvPointSetNode)
                removeMapPath = cmds.getAttr('%s.removeMap' % ZEnvPointSetNode)
                scaleMapPath = cmds.getAttr('%s.scaleMap' % ZEnvPointSetNode)

                if densityMapPath:
                    newDensityMapPath = densityMapPath.replace(old_path, new_path)
                    cmds.setAttr('%s.densityMap' % TNnode, newDensityMapPath, type='string')

                if removeMapPath:
                    newRemoveMapPath = removeMapPath.replace(old_path, new_path)
                    cmds.setAttr('%s.removeMap' % TNnode, newRemoveMapPath, type='string')

                if scaleMapPath:
                    newScaleMapPath = scaleMapPath.replace(old_path, new_path)
                    cmds.setAttr('%s.scaleMap' % TNnode, newScaleMapPath, type='string')

        if self.ui.comboBox.currentText() == 'TANE':
            print 'TANE yes'
            for node in cmds.ls(type='TN_AbcProxyMPxSurfaceShape'):
                fileName = cmds.getAttr('%s.filepath' % node)
                proxyName = cmds.getAttr('%s.proxypath' % node)

                newFileName = fileName.replace(old_path, new_path)
                newProxyName = proxyName.replace(old_path, new_path)

                cmds.setAttr('%s.filepath' % node, newFileName, type='string')
                cmds.setAttr('%s.proxypath' % node, newProxyName, type='string')

        if self.ui.comboBox.currentText() == 'TANE_MAP':
            print 'TANE_MAP yes'
            for TNnode in cmds.ls(type='TN_EnvironmentMPxNode'):
                densityMapPath = cmds.getAttr('%s.densityMap' % TNnode)
                removeMapPath = cmds.getAttr('%s.removeMap' % TNnode)
                scaleMapPath = cmds.getAttr('%s.scaleMap' % TNnode)

                if densityMapPath:
                    newDensityMapPath = densityMapPath.replace(old_path, new_path)
                    cmds.setAttr('%s.densityMap' % TNnode, newDensityMapPath, type='string')

                if removeMapPath:
                    newRemoveMapPath = removeMapPath.replace(old_path, new_path)
                    cmds.setAttr('%s.removeMap' % TNnode, newRemoveMapPath, type='string')

                if scaleMapPath:
                    newScaleMapPath = scaleMapPath.replace(old_path, new_path)
                    cmds.setAttr('%s.scaleMap' % TNnode, newScaleMapPath, type='string')

    def path_selectedChange(self):

        old_path = self.ui.lineEdit.text()
        new_path = self.ui.lineEdit_2.text()

        print old_path
        print new_path

        if self.ui.comboBox.currentText() == 'gpu':
            print 'gpu yes'
            for node in cmds.ls(sl=1, type='dxComponent'):
                fileName = cmds.getAttr('%s.abcFileName' % node)
                newFileName = fileName.replace(old_path, new_path)
                print newFileName
                cmds.setAttr('%s.abcFileName' % node, newFileName, type='string')

                sgc.componentImport("%s.abcFileName" % node, newFileName)

                sgc.componentReload("%s." % node)

        if self.ui.comboBox.currentText() == 'ZEnv':
            print 'ZEnv yes'
            for node in cmds.ls(sl=1, type='ZEnvSource'):
                fileName = cmds.getAttr('%s.assetPath' % node)
                newFileName = fileName.replace(old_path, new_path)
                print newFileName
                cmds.setAttr('%s.assetPath' % node, newFileName, type='string')

        if self.ui.comboBox.currentText() == 'ZEnv_MAP':
            print 'ZEnv_MAP yes'
            for ZEnvPointSetNode in cmds.ls(sl=1, type='ZEnvPointSet'):
                densityMapPath = cmds.getAttr('%s.densityMap' % ZEnvPointSetNode)
                removeMapPath = cmds.getAttr('%s.removeMap' % ZEnvPointSetNode)
                scaleMapPath = cmds.getAttr('%s.scaleMap' % ZEnvPointSetNode)

                if densityMapPath:
                    newDensityMapPath = densityMapPath.replace(old_path, new_path)
                    cmds.setAttr('%s.densityMap' % TNnode, newDensityMapPath, type='string')

                if removeMapPath:
                    newRemoveMapPath = removeMapPath.replace(old_path, new_path)
                    cmds.setAttr('%s.removeMap' % TNnode, newRemoveMapPath, type='string')

                if scaleMapPath:
                    newScaleMapPath = scaleMapPath.replace(old_path, new_path)
                    cmds.setAttr('%s.scaleMap' % TNnode, newScaleMapPath, type='string')

        if self.ui.comboBox.currentText() == 'TANE':
            print 'TANE yes'
            for node in cmds.ls(sl=1, type='TN_AbcProxyMPxSurfaceShape'):
                fileName = cmds.getAttr('%s.filepath' % node)
                proxyName = cmds.getAttr('%s.proxypath' % node)

                newFileName = fileName.replace(old_path, new_path)
                newProxyName = proxyName.replace(old_path, new_path)

                cmds.setAttr('%s.filepath' % node, newFileName, type='string')
                cmds.setAttr('%s.proxypath' % node, newProxyName, type='string')

        if self.ui.comboBox.currentText() == 'TANE_MAP':
            print 'TANE_MAP yes'
            for TNnode in cmds.ls(sl=1, type='TN_EnvironmentMPxNode'):
                densityMapPath = cmds.getAttr('%s.densityMap' % TNnode)
                removeMapPath = cmds.getAttr('%s.removeMap' % TNnode)
                scaleMapPath = cmds.getAttr('%s.scaleMap' % TNnode)

                if densityMapPath:
                    newDensityMapPath = densityMapPath.replace(old_path, new_path)
                    cmds.setAttr('%s.densityMap' % TNnode, newDensityMapPath, type='string')

                if removeMapPath:
                    newRemoveMapPath = removeMapPath.replace(old_path, new_path)
                    cmds.setAttr('%s.removeMap' % TNnode, newRemoveMapPath, type='string')

                if scaleMapPath:
                    newScaleMapPath = scaleMapPath.replace(old_path, new_path)
                    cmds.setAttr('%s.scaleMap' % TNnode, newScaleMapPath, type='string')

    def multiExport(self):

        module_path = self.ui.lineEdit_3.text()

        for selected in cmds.ls(sl=1):
            module_name = selected.split('_')
            print module_name
            cmds.select(selected)
            select_one = cmds.ls(sl=1)[0]
            print  select_one
            export_file_name = selected.replace('_GRP', '_v01.abc')
            print export_file_name
            export_abc = os.path.join(module_path, export_file_name)
            print export_abc

            mel_cmd = 'AbcExport -j "-writeVisibility -attr ObjectSet -attr ObjectName -attrPrefix rman -dataFormat ogawa -writeUVSets -root %s -file %s "' % (
            select_one, export_abc)
            mel.eval(mel_cmd)


    # gpu mode chage
    def gpuModeChange(self, text):
        print self.ui.comboBox_3.currentText()
        print self.ui.comboBox_4.currentText()

    def gpuReload(self):

        for gpuNode in cmds.ls(sl=1, type='dxComponent'):
            if self.ui.comboBox_3.currentText() == 'GPU':
                cmds.setAttr('%s.mode' % gpuNode, 1)
            if self.ui.comboBox_3.currentText() == 'Mesh':
                cmds.setAttr('%s.mode' % gpuNode, 0)
            if self.ui.comboBox_4.currentText() == 'Low':
                print self.ui.comboBox_4.currentText()
                cmds.setAttr('%s.display' % gpuNode, 3)
            if self.ui.comboBox_4.currentText() == 'Render':
                print self.ui.comboBox_4.currentText()
                cmds.setAttr('%s.display' % gpuNode, 1)



            sgc.componentReload(gpuNode)

    def createGroup(self):
        # select GPU
        nodeT = cmds.nodeType(cmds.ls(sl=1))
        print nodeT

        if nodeT == 'dxComponent':

            sel = cmds.ls(sl=True, type='dxComponent')
            print sel
            for i in sel:
                Grp = cmds.listRelatives(i, parent=True)[0]
                print Grp
                GpuArc = cmds.ls(i, type='ZGpuMeshShape', dag=1)[0]
                print GpuArc
                gpuCreator = cmds.listConnections(GpuArc, source=True, destination=False)[0]
                print gpuCreator
                getFilePath = cmds.getAttr('%s.file' % gpuCreator)
                print getFilePath
                s = getFilePath.split("/")[-1:]
                print s
                getgpuName = s[0].split('_model_')[0]
                print getgpuName
                selgpu = cmds.ls('%s|%s*' % (Grp, getgpuName), type='dxComponent')
                print selgpu
                new = '%s_%s_GRP' % (Grp, getgpuName)
                print new
                # cmds.select(selgpu)
                cmds.group(selgpu, n=new)

        if nodeT == 'transform':
            # select Group
            selGroup = cmds.ls(sl=1)
            cmds.select(selGroup, hi=1)
            selGpu = cmds.ls(sl=1, type='dxComponent')
            print selGpu

            getCreator = []
            for i in selGpu:
                print i
                selGpuArc = cmds.ls(i, type='ZGpuMeshShape', dag=1)[0]
                print selGpuArc
                Creator = cmds.listConnections(selGpuArc, source=True, destination=False)[0]
                print  Creator

                if Creator in getCreator:
                    pass
                else:
                    getCreator.append(Creator)

            print getCreator

            for a in getCreator:
                print a

                getFilePath = cmds.getAttr('%s.file' % a)
                print getFilePath

                s = getFilePath.split("/")[-1:]
                print s
                getgpuN = s[0].split('_model_')[0]
                print getgpuN

                selgpu = cmds.ls('%s|%s*' % (selGroup[0], getgpuN), type='dxComponent')
                cmds.select(selgpu)
                print selgpu
                new = '%s_%s_GRP' % (selGroup[0], getgpuN)
                print new
                cmds.group(selgpu, n=new)

    def lambertRandomAssign(self):
        # color Assign script
        sels = cmds.ls(sl=True)

        nodeName = sels[0].split('_model')[0]
        sortName = cmds.ls('%s*' % nodeName)

        colorA = cmds.shadingNode('lambert', asShader=True)
        colors = []
        for i in range(3):
            rand = random.uniform(0.0, 1.0)
            colors.append(rand)
        cmds.setAttr((colorA + '.color'), colors[0], colors[1], colors[2], type='double3')

        for selMesh in sels:

            if cmds.nodeType(selMesh) == 'transform':
                cmds.select(selMesh)
                cmds.hyperShade(assign=colorA)
            if cmds.nodeType(selMesh) == 'dxComponent':
                cmds.select(sortName)
                cmds.hyperShade(assign=colorA)
            else:
                pass

    def lambert1Assign(self):
        # Assign lambert1
        sels = cmds.ls(sl=1)
        nodeName = sels[0].split('_model')[0]
        print nodeName
        sortName = cmds.ls('%s*' % nodeName, type='dxComponent')
        print sortName

        # Assign lambert1
        for selMesh in sels:

            if cmds.nodeType(selMesh) == 'transform':
                cmds.select(selMesh)
                cmds.sets(selMesh, forceElement='initialShadingGroup')

            if cmds.nodeType(selMesh) == 'dxComponent':
                cmds.select(sortName)
                cmds.sets(sortName, forceElement='initialShadingGroup')
            else:
                pass

    def redColorAssign(self):
        # Red
        sels = cmds.ls(sl=1)
        nodeName = sels[0].split('_model')[0]
        print nodeName
        sortName = cmds.ls('%s*' % nodeName, type='dxComponent')
        print sortName

        Red = cmds.shadingNode('lambert', asShader=True, n='Red')
        cmds.setAttr((Red + '.color'), 1, 0, 0, type='double3')
        RedSG = cmds.sets(renderable=1, noSurfaceShader=1, empty=1, name='RedSG')
        cmds.defaultNavigation(connectToExisting=1, source=Red, destination=RedSG)
        for selMesh in sels:

            if cmds.nodeType(selMesh) == 'transform':
                cmds.select(selMesh)
                cmds.sets(selMesh, forceElement=RedSG)

            if cmds.nodeType(selMesh) == 'dxComponent':
                cmds.select(sortName)
                cmds.sets(sortName, forceElement=RedSG)
            else:
                pass

    def orangeColorAssign(self):
        sels = cmds.ls(sl=1)
        nodeName = sels[0].split('_model')[0]
        print nodeName
        sortName = cmds.ls('%s*' % nodeName, type='dxComponent')
        print sortName

        # Orange
        Orange = cmds.shadingNode('lambert', asShader=True, n='Orange')
        cmds.setAttr((Orange + '.color'), 1, 0.5, 0, type='double3')
        OrangeSG = cmds.sets(renderable=1, noSurfaceShader=1, empty=1, name='OrangeSG')
        cmds.defaultNavigation(connectToExisting=1, source=Orange, destination=OrangeSG)
        for selMesh in sels:

            if cmds.nodeType(selMesh) == 'transform':
                cmds.select(selMesh)
                cmds.sets(selMesh, forceElement=OrangeSG)

            if cmds.nodeType(selMesh) == 'dxComponent':
                cmds.select(sortName)
                cmds.sets(sortName, forceElement=OrangeSG)
            else:
                pass

    def yellowColorAssign(self):
        sels = cmds.ls(sl=1)
        nodeName = sels[0].split('_model')[0]
        print nodeName
        sortName = cmds.ls('%s*' % nodeName, type='dxComponent')
        print sortName

        # Yellow
        Yellow = cmds.shadingNode('lambert', asShader=True, n='Yellow')
        cmds.setAttr((Yellow + '.color'), 1, 1, 0, type='double3')
        YellowSG = cmds.sets(renderable=1, noSurfaceShader=1, empty=1, name='YellowSG')
        cmds.defaultNavigation(connectToExisting=1, source=Yellow, destination=YellowSG)
        for selMesh in sels:

            if cmds.nodeType(selMesh) == 'transform':
                cmds.select(selMesh)
                cmds.sets(selMesh, forceElement=YellowSG)

            if cmds.nodeType(selMesh) == 'dxComponent':
                cmds.select(sortName)
                cmds.sets(sortName, forceElement=YellowSG)
            else:
                pass

    def greenColorAssign(self):
        sels = cmds.ls(sl=1)
        nodeName = sels[0].split('_model')[0]
        print nodeName
        sortName = cmds.ls('%s*' % nodeName, type='dxComponent')
        print sortName

        # Green
        Green = cmds.shadingNode('lambert', asShader=True, n='Green')
        cmds.setAttr((Green + '.color'), 0, 0.5, 0, type='double3')
        GreenSG = cmds.sets(renderable=1, noSurfaceShader=1, empty=1, name='GreenSG')
        cmds.defaultNavigation(connectToExisting=1, source=Green, destination=GreenSG)
        for selMesh in sels:

            if cmds.nodeType(selMesh) == 'transform':
                cmds.select(selMesh)
                cmds.sets(selMesh, forceElement=GreenSG)

            if cmds.nodeType(selMesh) == 'dxComponent':
                cmds.select(sortName)
                cmds.sets(sortName, forceElement=GreenSG)
            else:
                pass

    def yellowGreenColorAssign(self):
        sels = cmds.ls(sl=1)
        nodeName = sels[0].split('_model')[0]
        print nodeName
        sortName = cmds.ls('%s*' % nodeName, type='dxComponent')
        print sortName

        # yellowGreen
        yellowGreen = cmds.shadingNode('lambert', asShader=True, n='yellowGreen')
        cmds.setAttr((yellowGreen + '.color'), 0.5, 0.5, 0, type='double3')
        yellowGreenSG = cmds.sets(renderable=1, noSurfaceShader=1, empty=1, name='yellowGreenSG')
        cmds.defaultNavigation(connectToExisting=1, source=yellowGreen, destination=yellowGreenSG)

        for selMesh in sels:

            if cmds.nodeType(selMesh) == 'transform':
                cmds.select(selMesh)
                cmds.sets(selMesh, forceElement=yellowGreenSG)

            if cmds.nodeType(selMesh) == 'dxComponent':
                cmds.select(sortName)
                cmds.sets(sortName, forceElement=yellowGreenSG)
            else:
                pass

    def skyColorAssign(self):
        sels = cmds.ls(sl=1)
        nodeName = sels[0].split('_model')[0]
        print nodeName
        sortName = cmds.ls('%s*' % nodeName, type='dxComponent')
        print sortName

        # Sky
        Sky = cmds.shadingNode('lambert', asShader=True, n='Sky')
        cmds.setAttr((Sky + '.color'), 0, 0.5, 1, type='double3')
        SkySG = cmds.sets(renderable=1, noSurfaceShader=1, empty=1, name='SkySG')
        cmds.defaultNavigation(connectToExisting=1, source=Sky, destination=SkySG)
        for selMesh in sels:

            if cmds.nodeType(selMesh) == 'transform':
                cmds.select(selMesh)
                cmds.sets(selMesh, forceElement=SkySG)

            if cmds.nodeType(selMesh) == 'dxComponent':
                cmds.select(sortName)
                cmds.sets(sortName, forceElement=SkySG)
            else:
                pass

    def blueColorAssign(self):
        sels = cmds.ls(sl=1)
        nodeName = sels[0].split('_model')[0]
        print nodeName
        sortName = cmds.ls('%s*' % nodeName, type='dxComponent')
        print sortName

        # Blue
        Blue = cmds.shadingNode('lambert', asShader=True, n='Blue')
        cmds.setAttr((Blue + '.color'), 0, 0, 1, type='double3')
        BlueSG = cmds.sets(renderable=1, noSurfaceShader=1, empty=1, name='BlueSG')
        cmds.defaultNavigation(connectToExisting=1, source=Blue, destination=BlueSG)
        for selMesh in sels:

            if cmds.nodeType(selMesh) == 'transform':
                cmds.select(selMesh)
                cmds.sets(selMesh, forceElement=BlueSG)

            if cmds.nodeType(selMesh) == 'dxComponent':
                cmds.select(sortName)
                cmds.sets(sortName, forceElement=BlueSG)
            else:
                pass

    def purpleColorAssign(self):
        sels = cmds.ls(sl=1)
        nodeName = sels[0].split('_model')[0]
        print nodeName
        sortName = cmds.ls('%s*' % nodeName, type='dxComponent')
        print sortName

        # Purple
        Purple = cmds.shadingNode('lambert', asShader=True, n='Purple')
        cmds.setAttr((Purple + '.color'), 0.5, 0, 1, type='double3')
        PurpleSG = cmds.sets(renderable=1, noSurfaceShader=1, empty=1, name='PurpleSG')
        cmds.defaultNavigation(connectToExisting=1, source=Purple, destination=PurpleSG)
        for selMesh in sels:

            if cmds.nodeType(selMesh) == 'transform':
                cmds.select(selMesh)
                cmds.sets(selMesh, forceElement=PurpleSG)

            if cmds.nodeType(selMesh) == 'dxComponent':
                cmds.select(sortName)
                cmds.sets(sortName, forceElement=PurpleSG)
            else:
                pass

    def magentaColorAssign(self):
        sels = cmds.ls(sl=1)
        nodeName = sels[0].split('_model')[0]
        print nodeName
        sortName = cmds.ls('%s*' % nodeName, type='dxComponent')
        print sortName

        # Magenta
        Magenta = cmds.shadingNode('lambert', asShader=True, n='Magenta')
        cmds.setAttr((Magenta + '.color'), 1, 0, 1, type='double3')
        MagentaSG = cmds.sets(renderable=1, noSurfaceShader=1, empty=1, name='MagentaSG')
        cmds.defaultNavigation(connectToExisting=1, source=Magenta, destination=MagentaSG)
        for selMesh in sels:

            if cmds.nodeType(selMesh) == 'transform':
                cmds.select(selMesh)
                cmds.sets(selMesh, forceElement=MagentaSG)

            if cmds.nodeType(selMesh) == 'dxComponent':
                cmds.select(sortName)
                cmds.sets(sortName, forceElement=MagentaSG)
            else:
                pass

    def transColorAssign(self):
        sels = cmds.ls(sl=1)
        nodeName = sels[0].split('_model')[0]
        print nodeName
        sortName = cmds.ls('%s*' % nodeName, type='dxComponent')
        print sortName

        # Trans
        Trans = cmds.shadingNode('lambert', asShader=True, n='Trans')
        cmds.setAttr((Trans + '.color'), 0.162, 0.4, 0.4, type='double3')
        cmds.setAttr((Trans + '.transparency'), 0.4, 0.4, 0.4, type='double3')
        TransSG = cmds.sets(renderable=1, noSurfaceShader=1, empty=1, name='TransSG')
        cmds.defaultNavigation(connectToExisting=1, source=Trans, destination=TransSG)
        for selMesh in sels:

            if cmds.nodeType(selMesh) == 'transform':
                cmds.select(selMesh)
                cmds.sets(selMesh, forceElement=TransSG)

            if cmds.nodeType(selMesh) == 'dxComponent':
                cmds.select(sortName)
                cmds.sets(sortName, forceElement=TransSG)
            else:
                pass

def main():
    # app = QtWidgets.QApplication(sys.argv)
    mainVar = Asset_gpuControlTool_v01()
    mainVar.show()
    # sys.exit(app.exec_())


if __name__ == "__main__":
    main()
