import os, sys
import site
import optparse
import shutil
import datetime
import getpass

try:
    # import pymodule.Qt as Qt
    # from pymodule.Qt import QtGui
    # from pymodule.Qt import QtWidgets
    # from pymodule.Qt import QtCore
    from PySide2 import QtWidgets, QtCore, QtGui
    from ui.spanner2_addToInventory import Ui_Form as InvenUI
    import historyAction
except:
    pass

# IP config
import dxConfig
DB_IP = dxConfig.getConf('DB_IP')

# mongoDB
import pymongo
from pymongo import MongoClient
client = MongoClient(DB_IP)

# tractor api setup
TractorRoot = '/netapp/backstage/pub/apps/tractor/linux/Tractor-2.2'
site.addsitedir( '%s/lib/python2.7/site-packages' % TractorRoot )
import tractor.api.author as author
TRACTOR_IP = '10.0.0.25'

CURRENTPATH = os.path.dirname( os.path.abspath( __file__ ) )

if Qt:
    class InventoryDialog(QtWidgets.QDialog):
        def __init__(self, parent=None):
            QtWidgets.QDialog.__init__(self, parent)
            self.ui = InvenUI()
            self.ui.setupUi(self)
            self.setWindowTitle("Save to Inventory")
            self.ui.del_pushButton.setIcon(
                QtGui.QIcon(QtGui.QPixmap(os.path.join(CURRENTPATH, "resource/trashCan.png"))))
            self.ui.add_pushButton.clicked.connect(self.addPath)
            self.ui.del_pushButton.clicked.connect(self.delPath)
            self.ui.insertInventory_pushButton.clicked.connect(self.addToInventory)

        def addPath(self):
            if self.parent().ui.workCode_listWidget.currentItem().text() == 'rig' \
                    and self.parent().ui.fileName_lineEdit.text():
                path = self.parent().ui.filePath_lineEdit.text() + '/' + self.parent().ui.fileName_lineEdit.text()
                filename = os.path.basename(path)
                imageFile = '.%s.thumb.jpg' % filename
                imagePath = os.path.join(os.path.dirname(path), imageFile)
                if os.path.exists(path) and os.path.splitext(path)[-1] == '.mb':
                    if os.path.exists(imagePath):
                        self.ui.fileList_listWidget.addItem(path)
                    else:
                        print '*** MUST TAKE SNAPSHOT'

        def delPath(self):
            row = self.ui.fileList_listWidget.currentRow()
            self.ui.fileList_listWidget.takeItem(row)

        def addToInventory(self):
            for i in range(self.ui.fileList_listWidget.count()):
                path =  self.ui.fileList_listWidget.item(i).text()
                try:
                    print path
                    self.spoolInventory(path)
                except:
                    print 'ADD TO INVENTORY FAILED'
            self.ui.fileList_listWidget.clear()


        def spoolInventory(self, path):
            # src name
            src = path.split('/')
            filename = os.path.basename(path)

            job = author.Job()
            job.title = '(Inventory) ' + str(filename)
            job.comment = 'Inventory'
            job.metadata = 'source file : %s' % str(path)

            maya_version = os.getenv('MAYA_VER')
            job.service = 'Cache||USER'
            job.envkey = ['cache2-2017']
            job.maxactive = 24
            job.tier = 'cache'
            job.projects = ['export']
            job.tags = ['GPU']

            JobTask = author.Task(title='Inventory Job')
            JobTask.serialsubtasks = 1

            taskCmd = ['python', '%%D(%s/InventorySpool.py)' % CURRENTPATH]
            taskCmd += ['-f',path]
            JobTask.addCommand(
                author.Command(argv=taskCmd, service='', tags=['py']))
            job.addChild(JobTask)
            job.priority = 100

            # job.paused = True
            author.setEngineClientParam(hostname=TRACTOR_IP, port=80,
                                        user=getpass.getuser(), debug=True)
            job.spool()
            author.closeEngineClient()
            print '# INVENTORY SPOOL FINISHED'

if __name__ == '__main__':
    optparser = optparse.OptionParser()

    optparser.add_option(
        '-f', '--file', dest='file', type='string',
        help='input file name.')

    (opts, args) = optparser.parse_args(sys.argv)

    # src name
    savePubPath = opts.file
    src = savePubPath.split('/')
    rigtype = src[src.index('asset') + 1]
    rigname = src[src.index('asset') + 2]
    filename = os.path.basename(savePubPath)
    imageFile = '.%s.thumb.jpg' % filename
    imagePath = os.path.join(os.path.dirname(savePubPath), imageFile)

    # copy name
    copyPath = '/assetlib/3D/rigging/%s/%s' % (rigtype, rigname)
    if not os.path.exists(copyPath):
        os.makedirs(copyPath)
    copyFilePath = os.path.join(copyPath, filename)
    copyImagePath = os.path.join(copyPath, '%s.jpg' % filename)
    shutil.copyfile(savePubPath, copyFilePath)
    shutil.copyfile(imagePath, copyImagePath)

    # DB
    client = MongoClient(DB_IP)
    db = client['inventory']
    coll = db['assets']
    result = coll.find({"name": os.path.splitext(filename)[0],'type':'RIG_REF'}
                       ).sort('version', pymongo.DESCENDING)
    if result.count() > 0:
        ver = int(result[0]['version']) + 1
    else:
        ver = 0

    invenRigDict = {
        "files": {
            "org": copyFilePath,
            "thumbnail": copyImagePath,
            "preview": copyImagePath
        },
        "name": os.path.splitext(filename)[0],
        "tags": [
            "rigging",
            rigtype,
            rigname,
            os.path.splitext(filename)[0]
        ],
        "project_desc": os.path.splitext(filename)[0],
        "enabled": True,
        "project": rigtype,
        "type": "RIG_REF",
        "time": datetime.datetime.now().isoformat(),
        "version": ver
    }
    coll.insert_one(invenRigDict)

    print '*** INSERTED TO INVENTORY / CHECK TRACTOR ***'
