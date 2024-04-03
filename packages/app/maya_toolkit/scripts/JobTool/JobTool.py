#encoding=utf-8

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *
import QtTopUiLoader
print QtTopUiLoader
reload(QtTopUiLoader)

from pymel.all import *
import os, re, math
import shutil
import configobj
import json
import socket
import time
import pprint

import maya.OpenMayaUI as mui
import shiboken2

import maya.cmds as cmds
import jobscript
reload(jobscript)


#USER = getpass.getuser()
USER = os.environ["HOSTNAME"]
if ( len(USER)  > 1 ):
    if '\\' in USER:
        USER = USER.split("\\")[1]
# initialize jobtool_config
HOME = os.getenv('HOME')


config_file = os.path.join(os.path.abspath(__file__+'/../'), 'jobtool_config.ini')
print config_file
jobtool_config = configobj.ConfigObj(config_file)
#-------------------------------------------------------------------------------
#
#   Render Joob Tool
#
#-------------------------------------------------------------------------------
# load ui
# jb_uifile = os.path.join(os.path.abspath(__file__+'/../'), 'ijobtool.ui')
# jb_fromClass, jb_baseClass = QtTopUiLoader.UiLoader().loadUiType(jb_uifile)
# maya version check
mayaVersion = cmds.about(v=True)

def getMayaWindow():
    ptr = mui.MQtUtil.mainWindow()
    return shiboken2.wrapInstance(long(ptr), QtWidgets.QWidget)

# class Job_UI(jb_fromClass, jb_baseClass):
class Job_UI(QWidget):
    def __init__(self, parent=None):
        super(Job_UI, self).__init__(parent)

        jb_uifile = os.path.join(os.path.abspath(__file__ + '/../'), 'ijobtool.ui')
        QtTopUiLoader.UiLoader().loadUi(jb_uifile, self)


        # self.setupUi(self)
        # Product
        self.prod_comboBox.addItems(jobtool_config['product']['type'])
        self.artname_lineEdit.setText('%s' %USER)

        # Engine
        self.tengine_comboBox.addItems(jobtool_config['tractor_engine']['engines'])
        self.priority_lineEdit.setText('100')
        self.packet_lineEdit.setText('1')

        self.product_setup()

        # Renderer
        self.cmdtype_comboBox.addItems(jobtool_config["renderer"].keys())

        # frame range
        self.frameSet_ComboBox.addItem("Time Slider")
        self.frameSet_ComboBox.addItem("Render Setting")
        self.frameSet_ComboBox.activated[int].connect(self.changeFramgeSet)
        self.changeFramgeSet(0)

        self.byFrame_lineEdit.setText('1')

        start = int(playbackOptions(q=True, min=True))
        end = int(playbackOptions(q=True, max=True))


        # camera
        caminfo = GetCameraList()
        print caminfo

        ignoreCanCheck = ["front", "persp", "side", "top"]
        camTableH = 1
        self.carmeras={}
        for camName in caminfo[0]:
            if not camName in ignoreCanCheck:
                self.carmera_ComboBox.addItem(camName)
                self.carmeras.update( { camName: '%s-%s' % (start, end)})
                print ">>>",  camName, '%s-%s' % (start, end)


        # Default Set
        self.setDefault()

        # hostname
        self.hostName =  socket.gethostname()

        # Render Layer Set
        self.renderLayers=self.setRenderLayer()
        print ">>>", self.renderLayers
        # cmd bind
        self.close_Button.clicked.connect(self.close_process)
        self.render_Button.clicked.connect(self.run_render)
        # self.prod_comboBox.currentIndexChanged.connect(self.updateFarm)


    def keyPressEvent(self, event):
        pass

    # def setRendererOpt(self, statusV, rendererV):
    #     if rendererV == "arnold":
    #         if self.arnoldAutoTx_checkBox.isChecked():
    #             cmds.setAttr("defaultArnoldRenderOptions.autotx", 1)
    #         else:
    #             cmds.setAttr("defaultArnoldRenderOptions.autotx", 0)

    def changeFramgeSet(self, frameSet):
        if frameSet == 0:
            start = int(playbackOptions(q=True, min=True))
            end = int(playbackOptions(q=True, max=True))
        elif frameSet == 1:
            start = int( cmds.getAttr("defaultRenderGlobals.startFrame") )
            end = int (cmds.getAttr("defaultRenderGlobals.endFrame") )
        else:
            start = int(playbackOptions(q=True, min=True))
            end = int(playbackOptions(q=True, max=True))

        self.frame_lineEdit.setText('%s-%s' % (start, end))


    def product_setup(self):
        # # scenePathV = cmds.file(q=True, sn=True)
        # mayaproj = os.path.dirname(workspace(q=True, rd=True))#.replace("LOTAS","LOTAS_NETA" )

        current = sceneName()
        if '/scenes/' in current:
            mayaproj = current.split('/scenes/')[0]
        else:
            mayaproj = os.path.dirname(workspace(q=True, rd=True))

        self.prod_comboBox.setCurrentIndex(0)
        self.mproj_lineEdit.setText(mayaproj)
        self.moutdir_lineEdit.setText(os.path.join('images', os.path.basename(sceneName()).split('.')[0]))

    def close_process(self):
        jobWindow.close()

    # fileout path
    def outputPath(self):
        outdir = os.path.join(str(self.mproj_lineEdit.text()), str(self.moutdir_lineEdit.text()))
        return outdir


    # Get UI value
    def get_ui(self):

        opts = {}
        opts['mayaversion'] = mayaVersion
        opts['product'] = str(self.prod_comboBox.currentText())
        opts['showname'] = os.path.dirname(workspace(q=True, rd=True)).split(os.sep)[2]#.replace("LOTAS","LOTAS_NETA" )  #/show/projectName/
        opts['profile'] = opts['product'] #A or B
        opts['mayaProj'] = str(self.mproj_lineEdit.text())
        opts['mayaScene'] = str(sceneName())
        opts['baseName'] = os.path.basename(opts['mayaScene']).split('.')[0]
        opts['mayaOutDir'] = self.outputPath()
        opts['imgDirName'] = opts['mayaOutDir']
        opts['imgName'] = os.path.basename(opts['mayaOutDir'])
        opts['renderType'] = str(self.cmdtype_comboBox.currentText())
        #opts['frameRange'] = str(self.frame_lineEdit.text())
        opts['byFrame'] = str(self.byFrame_lineEdit.text())
        #opts['camera'] = str(self.camera_comboBox.currentText())
        opts['cameraInfo'] = {self.carmera_ComboBox.currentText(): str(self.frame_lineEdit.text())}


        #general option
        opts['Engine'] = str(self.tengine_comboBox.currentText())
        opts['priority'] = int(self.priority_lineEdit.text())
        opts['packet'] = int(self.packet_lineEdit.text())
        #opts['envkey'] = str(self.envkey_comboBox.currentText())

        opts['hostName'] = self.hostName
        opts['artistName'] = str(self.artname_lineEdit.text())


        opts['renderLayer'] = self.renderLayers
        # opts['makeRenderLayerJob'] = self.ma0keRenderLayerJob_checkBox.isChecked()
        opts['makeRenderLayerJob']=0

        opts['mayabin'] = jobtool_config['maya']['mayabin']
        opts['nodeNumber'] = jobtool_config['nodeNumber']
        opts['nowtime'] = time.strftime("%Y%m%d_%H%M%S")


        return opts

    # create jobscript
    def create_jobscript(self):
        opts = self.get_ui()
        render_script(opts)
        jobWindow.close()

    def joblog_process(self):
        runtime.SaveScene()
        opts = self.get_ui()
        metafile = os.path.join(HOME, '.irman', '%s.job' % opts['imgName'])
        f = open(metafile, 'w')
        f.write(json.dumps(opts))
        f.close()
        jobWindow.close()

    # Render
    def run_render(self):
        opts = self.get_ui()
        # # debug
        # pprint.pprint(opts, indent=4)
        # return

        # render file
        runtime.SaveScene()
        mayafile = opts['mayaScene']

        if not os.path.isdir("%s/tmp" %os.path.dirname(mayafile)):
            os.makedirs("%s/tmp" %os.path.dirname(mayafile))

        refilename = '%s/tmp/%s.%s' % (os.path.dirname(mayafile), opts['imgName'],opts['nowtime'])
        print ">>>>", refilename
        shutil.copy2(mayafile, refilename)
        opts['mayaScene'] = refilename

        scriptdir = os.path.join(opts['mayaProj'], 'tmp', 'alfscript')

        if not os.path.exists(scriptdir):
            os.makedirs(scriptdir)


        def runJob():

            if not os.path.exists(opts['mayaOutDir']):
                print("makedirs {}".format(opts['mayaOutDir']))
                os.makedirs(opts['mayaOutDir'])

            scriptfile = render_script(opts)

            cmd = '{0}/tractor-spool --engine="{1}:80" --user="{2}" --priority={3} {4}'  \
                  .format( jobtool_config['tractor_blade']['path'], opts['Engine'], USER, opts['priority'], scriptfile)

            print cmd, '<<<<<< <<<<'
            os.system(cmd)

        if opts['makeRenderLayerJob']:
            mayaOutDir = opts['mayaOutDir']
            mayaOutDirList = mayaOutDir.split("/")
            outDirName = mayaOutDirList[-1].split("/")[-1]
            outDirNameList = outDirName.split("_")

            renderLayerDic = {}
            for i in range(self.renderLayerTableWidget.rowCount()):
                layerItem = self.renderLayerTableWidget.cellWidget(i, 0)
                farmItem = self.renderLayerTableWidget.cellWidget(i, 1)
                camItem = self.renderLayerTableWidget.item(i, 2)
                frameItem = self.renderLayerTableWidget.item(i, 3)

                if layerItem.isChecked():
                    layerName = layerItem.text()
                    farmV = farmItem.currentText()
                    camV = camItem.text()
                    frameV = frameItem.text()
                    if farmV != "" and camV != "" and frameV != "":
                        print farmV, camV, frameV
                        layerNum = layerName.split("_")[-1]
                        outDirNameN = "_".join([outDirNameList[0], layerNum, "_".join(outDirNameList[2:])])

                        mayaOutDirN= "/".join(["/".join(mayaOutDirList[:-1]), outDirNameN])
                        print mayaOutDirN

                        renderLayerDic[layerName] = {}
                        renderLayerDic[layerName]["mayaOutDir"] = mayaOutDirN
                        renderLayerDic[layerName]["imgDirName"] = mayaOutDirN
                        renderLayerDic[layerName]["cameraInfo"] = {}
                        renderLayerDic[layerName]["cameraInfo"][camV] = frameV

                        opts["renderLayer"] = {}
                        opts["renderLayer"][layerName] = farmV
                        opts["cameraInfo"] = {}
                        opts["cameraInfo"][camV] = frameV
                        opts["mayaOutDir"] = mayaOutDirN
                        opts["imgDirName"] = mayaOutDirN
                        opts['baseName'] = outDirNameN
                        print "runJob"
                        runJob()
        else:
            runJob()

        jobWindow.close()

    # Save Default value
    def saveDefault(self):
        attr = {}
        attr['self.prod_comboBox'] = int(self.prod_comboBox.currentIndex())
        attr['self.cmdtype_comboBox'] = int(self.cmdtype_comboBox.currentIndex())
        attr['self.artname_lineEdit'] = str(self.artname_lineEdit.text())
        attr['self.tengine_comboBox'] = int(self.tengine_comboBox.currentIndex())
        attr['self.priority_lineEdit'] = str(self.priority_lineEdit.text())
        attr['self.comment_lineEdit'] = str(self.comment_lineEdit.text())
        f = file(os.path.join(HOME, '.irman', 'jobtool.init'), 'w')
        f.write(json.dumps(attr))
        f.close()

    # Set Default value
    def setDefault(self):
        initfile = os.path.join(HOME, 'jobtool.init')
        attr = {}
        if os.path.exists(initfile):
            f = file(initfile, 'r')
            attr = json.loads(f.read())
            f.close()
        if attr:
            for i in attr:
                i_split = i.split('_')
                if i_split[-1] == 'comboBox':
                    eval(i).setCurrentIndex(attr[i])
                elif i_split[-1] == 'lineEdit':
                    eval(i).setText(attr[i])


    def setRenderLayer(self):
        renderLayerList = cmds.ls(type="renderLayer")

        renderLayerDic = {}
        for renderLayerName in renderLayerList:
            if len(renderLayerName.split(":")) == 1 : #referenc render layer  check
                if re.search("defaultRenderLayer", renderLayerName):
                    if renderLayerName != "defaultRenderLayer":
                        continue
                renderableV = cmds.getAttr("%s.renderable"%renderLayerName)
                if renderableV :
                    renderableID = cmds.getAttr("%s.identification" % renderLayerName)
                    renderLayerDic[renderLayerName] = str(self.prod_comboBox.currentText())

        return renderLayerDic


    def updateFarm(self, currentIndex):

        for i in self.renderLayerComboList:
            i.setCurrentIndex(currentIndex)


def JobTool():
#def showUI():
    global jobWindow
    jobWindow = Job_UI()
    jobWindow.show()

#-------------------------------------------------------------------------------
#
#   Support Command
#
#-------------------------------------------------------------------------------
#	Render Camera
def GetCameraList():
    rendercam = []
    camList = []
    cameras = cmds.ls(type='camera')
    for cam in cameras:
        trnode = cmds.listRelatives(cam, p=True)[0]
        camList.append(trnode)
        getv = cmds.getAttr('%s.renderable' % cam)
        if getv == True:
            rendercam.append(trnode)

    fcam = rendercam[-1]

    return camList, fcam

#	Create JobScript
def render_script(opts):
    alf = ''
    alfclass = eval('jobscript.' + jobtool_config["renderer"][opts['renderType']] + '_AlfredScript(opts)')
    print 'ijobscript.' + jobtool_config["renderer"][opts['renderType']] + '_AlfredScript(opts)'
    script = file(alfclass.script_file, 'w')
    script.write(alfclass.alf)
    script.close()
    print "alfclass.script_file : ", alfclass.script_file
    return alfclass.script_file
