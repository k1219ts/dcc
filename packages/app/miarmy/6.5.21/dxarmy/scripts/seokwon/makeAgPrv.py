# encoding:utf-8
# !/usr/bin/env python

import os
import site, getpass
import maya.cmds as cmds
site.addsitedir('/netapp/backstage/pub/apps/tractor/linux/Tractor-2.2/lib/python2.7/site-packages')
import tractor.api.author as author
from PySide2 import QtCore, QtGui, QtWidgets, QtCompat

currentpath = os.path.abspath(__file__)
uiFile = os.path.join(os.path.dirname(currentpath), "../ui/prvExport.ui")

EXPORT_TYPE = ['Crowd', 'Animation']
CHLIST = list()

def setup_ui(ui, base_instance=None):
    for member in dir(ui):
        if not member.startswith('__') and member is not 'staticMetaObject':
            setattr(base_instance, member, getattr(ui, member))

class Window(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(Window, self).__init__(parent)
        ui = QtCompat.load_ui(uiFile)
        setup_ui(ui, self)
        self.mayaFile = ""
        self.connectSignal()

    def connectSignal(self):
        self.stSB.setValue(1001)
        self.edSB.setValue(1050)
        self.dvSB.setValue(0)
        self.expCB.addItems(EXPORT_TYPE)
        self.inputBtn.clicked.connect(self.getifPath)
        self.outputBtn.clicked.connect(self.setofPath)
        self.prvBtn.clicked.connect(self.getList)
        self.onBtn.clicked.connect(self.openDir)
        self.onBtn.setIcon(QtGui.QIcon(QtGui.QPixmap("/dexter/Cache_DATA/animation/A0_Artist/Choi_SeokWon/icons/folder.png")))
        self.onBtn.setIconSize(QtCore.QSize(26, 26))

    def openDir(self):
        pbDir = os.sep.join(self.outLE.text().split(os.sep)[:-1])
        self.makeDir(pbDir)
        os.system('/usr/bin/nautilus %s &' % pbDir)

    def getifPath(self):
        multipleFilter = "Maya Files (*.mb *.ma);;Maya ASCII (*.ma);;Maya Binary (*.mb)"
        inputFile = str(cmds.fileDialog2(fileMode=1, fileFilter=multipleFilter, caption="Import Maya Scene File")[0])
        self.inLE.setText(inputFile)
        defaultPath = os.sep.join(inputFile.split(os.sep)[:-2]) + "/preview/"
        prvFileName = inputFile.split(os.sep)[-1].split(".")[0] + ".mp4"
        self.outLE.setText(defaultPath + prvFileName)

    def makeDir(self, tarPath):
        if not os.path.exists(tarPath):
            os.mkdir(tarPath)

    def setofPath(self):
        if self.outLE.text():
            stPoint = os.sep.join(str(self.outLE.text()).split(os.sep)[:-1])
            self.makeDir(stPoint)
        else:
            stPoint = "/show/"
        outputFile = str(cmds.fileDialog2(startingDirectory=stPoint, fileMode=0, fileFilter="Preview File (*.mp4)", caption="Export Preview File")[0])
        self.outLE.setText(outputFile)

    def divDuration(self):
        fileName = str(self.inLE.text()).split(os.sep)[-1].split(".")[0]
        job = author.Job()
        job.title = fileName
        job.comment = ''
        job.metadata = ''
        job.envkey = ['cache2-2017']
        job.service = 'Miarmy'
        job.maxactive = 10
        job.tier = 'cache'
        job.projects = ['export']
        job.tags = ['GPU']
        JobTask = author.Task(title='Job')
        JobTask.serialsubtasks = 1
        ScriptRoot = "/dexter/Cache_DATA/animation/A0_Artist/Choi_SeokWon/Scripts"
        AlembicTask = author.Task(title='AlembicExport')
        AlembicTask.serialsubtasks = 0
        JobTask.addChild(AlembicTask)
        minTime = self.stSB.value()
        maxTime = self.edSB.value()
        drst = int(maxTime - minTime + 1)
        mayaFile = str(self.inLE.text())
        mayaEx = str(self.outLE.text())
        divNum = int(self.dvSB.value())
        divRt = divmod(drst, divNum)
        if self.expCB.currentText() == 'Crowd':
            dataType = 0
        else:
            dataType = 1
        st = int(minTime) - 1
        if divNum != 0 or divNum != 1:
            for i in range(divNum):
                ens = list()
                for j in range(divRt[0]):
                    st += 1
                    ens.append(st)
                if divRt[1] != 0 and i == (divNum - 1):
                    for k in range(divRt[1]):
                        st += 1
                        ens.append(st)
                AlmFrameTask = author.Task(title='prvExport')
                command = ['mayapy', '%%D(%s/makeAgPrv_fc.py)' % ScriptRoot, mayaFile, mayaEx, minTime, maxTime, dataType, ens]
                AlmFrameTask.addCommand(author.Command(service='', envkey=['cache2-2017'], tags=['py'], argv=command))
                AlembicTask.addChild(AlmFrameTask)
        else:
            AlmFrameTask = author.Task(title='prvExport')
            command = ['mayapy', '%%D(%s/makeAgPrv_fc.py)' % ScriptRoot, mayaFile, mayaEx, minTime, maxTime, dataType]
            AlmFrameTask.addCommand(author.Command(service='', envkey=['cache2-2017'], tags=['py'], argv=command))
            AlembicTask.addChild(AlmFrameTask)
        job.addChild(JobTask)
        return job

    def getList(self):
        pubType = self.divDuration()
        self.tracshot(pubType)

    def tracshot(self, job):
        job.priority = 1000.0
        author.setEngineClientParam(hostname='10.0.0.25', port=80, user=getpass.getuser(), debug=True)
        job.spool()
        author.closeEngineClient()

def main():
    global myWindow
    try:
        myWindow.close()
    except:
        pass
    # app = QtWidgets.QApplication(sys.argv)
    myWindow = Window()
    myWindow.show()
    # sys.exit(app.exec_())

if __name__ == '__main__':
    main()