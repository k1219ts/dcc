# encoding:utf-8
# !/usr/bin/env python

import os, sys
import site, getpass
import maya.cmds as cmds
import shutil
site.addsitedir('/netapp/backstage/pub/apps/tractor/linux/Tractor-2.2/lib/python2.7/site-packages')
import tractor.api.author as author
from PySide2 import QtCore, QtGui, QtWidgets, QtCompat

CHLIST = list()
currentpath = os.path.abspath(__file__)
uiFile = os.path.join(os.path.dirname(currentpath), "../ui/abcPoints.ui")

def setup_ui(ui, base_instance=None):
    for member in dir(ui):
        if not member.startswith('__') and member is not 'staticMetaObject':
            setattr(base_instance, member, getattr(ui, member))

class DragDropTest(QtWidgets.QListWidget):
    def __init__(self, parent):
        QtWidgets.QListWidget.__init__(self, parent)
        self.setAcceptDrops(True)
        self.setDragEnabled(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls:
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls:
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasUrls:
            event.accept()
            lst = list()
            for i in event.mimeData().urls():
                lst.append(i.toLocalFile())
                CHLIST.append(str(i.toLocalFile()))
            doIt = Window()
            doIt.getList()
        else:
            event.ignore()

class Window(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(Window, self).__init__(parent)
        ui = QtCompat.load_ui(uiFile)
        setup_ui(ui, self)
        self.connectSignal()

    def connectSignal(self):
        self.actList = DragDropTest(self.wd)
        self.actList.setGeometry(QtCore.QRect(1, 1, 460, 264))

    def getList(self):
        self.pathSet()
        mayafile_tmp = self.createMayaTempFile()
        pubType = self.trcJob('(Crw-Alm) ', 'AlembicExport_rmantd', mayafile_tmp)
        self.tracshot(pubType)

    def createMayaTempFile(self):   # Save Scene File for Backup
        name = os.path.splitext(str(os.path.basename(cmds.file(q=1, sn=1))))[0]
        exr = os.path.splitext(str(os.path.basename(cmds.file(q=1, sn=1))))[-1]
        temp_name = name
        mayafile_tmp = os.path.join(os.path.dirname(str(cmds.file(q=1, sn=1))), 'renderScenes', temp_name + exr)
        if not os.path.exists(os.path.dirname(mayafile_tmp)):
            os.makedirs(os.path.dirname(mayafile_tmp))
        cmds.file(save=True)
        shutil.copy2( str(cmds.file(q=True, sn=True)), mayafile_tmp)
        return mayafile_tmp

    def pathSet(self):
        selFile = str(os.path.basename(cmds.file(q=1, sn=1)))
        fPat = str(os.path.dirname(cmds.file(q=1, sn=1)))
        setPath = str(os.sep.join(fPat.split(os.sep)[0:-1]))
        if not os.path.exists(setPath + "/cache/alembic/"):
            if not os.path.exists(setPath + "/cache/"):
                os.mkdir(setPath + "/cache/")
            os.mkdir(setPath + "/cache/alembic/")
        if len(cmds.ls("agline_*", type="displayLayer")) != 0:
            shtFd = str(selFile).split(".")[0]
            shtPth = str(setPath + "/cache/alembic/%s/" % shtFd)
            if not os.path.exists(shtPth):
                os.mkdir(shtPth)
            else:
                pass
        else:
            pass

    def trcJob(self, name, ofile, mayaFile):
        job = author.Job()
        job.title = name + str(os.path.basename(mayaFile).split('.')[0])
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
        ScriptRoot = "/dexter/Cache_DATA/animation/A0_Artist/Choi_SeokWon/Crowd_RnD/script"
        AlembicTask = author.Task(title='AlembicExport')
        AlembicTask.serialsubtasks = 0
        JobTask.addChild(AlembicTask)
        dpList = cmds.ls("agline_*", type="displayLayer")
        stframe = cmds.playbackOptions(q=True, min=True)
        enframe = cmds.playbackOptions(q=True, max=True)
        divFileNumb = 1
        for i in range(len(dpList)):
            AlmFrameTask = author.Task(title='crwExport')
            command = ['mayapy', '%%D(%s/%s.py)' % (ScriptRoot, ofile), mayaFile, stframe, enframe, i, str(divFileNumb)]
            AlmFrameTask.addCommand(author.Command(service='', envkey=['cache2-2017'], tags=['py'], argv=command))
            AlembicTask.addChild(AlmFrameTask)
            divFileNumb += 1
        # self.postScript(mayaFile, Parent=JobTask)
        job.addChild(JobTask)
        return job

    def postScript(self, mayaPath, Parent=None):
        Parent.addCleanup(author.Command(argv=['/bin/rm -f', mayaPath], service=''))

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