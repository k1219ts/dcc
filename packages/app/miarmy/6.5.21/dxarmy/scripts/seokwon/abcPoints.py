# encoding:utf-8
# !/usr/bin/env python

import os, sys
import site, getpass
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
            doIt.getList(lst, self)
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

    def getList(self, lst, UIadd):
        pntName = lst[0].split(os.sep)[-1].split(".")[0] + "_Points.abs"
        exPath = os.sep.join(lst[0].split(os.sep)[:-2]) + "/cache/alembic/" + pntName
        UIadd.addItems([exPath])
        mayaFile = lst[0]
        pubType = self.trcJob('(Crw-Alm) ', 'abcPoints_fc', mayaFile)
        self.tracshot(pubType)

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
        ScriptRoot = "/dexter/Cache_DATA/animation/A0_Artist/Choi_SeokWon/Scripts/"
        AlembicTask = author.Task(title='AlembicExport')
        AlembicTask.serialsubtasks = 0
        JobTask.addChild(AlembicTask)
        AlmFrameTask = author.Task(title='crwExport')
        command = ['mayapy', '%%D(%s/%s.py)' % (ScriptRoot, ofile), mayaFile]
        AlmFrameTask.addCommand(author.Command(service='', envkey=['cache2-2017'], tags=['py'], argv=command))
        AlembicTask.addChild(AlmFrameTask)
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