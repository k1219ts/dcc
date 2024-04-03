#encoding=utf-8
#--------------------------------------------------------------------------------
#
#    Dexter RND
#
#        Daeseok Chae, cds7031@gmail.com
#
#    daseok.chae 2017.09.06 $5
#
#-------------------------------------------------------------------------------

# import Basic Module
import sys
import os
import getpass

# import Mari
import mari

# import Config File
from dxConfig import dxConfig

# import MongoDB
from pymongo import MongoClient

# import Tractor API
import site
site.addsitedir(dxConfig.getConf("TRACTOR_API"))
import tractor.api.author as author


from PySide2 import QtWidgets, QtGui, QtCore
# from pymodule.Qt import QtWidgets as QtGui
# from pymodule.Qt import QtCore
# from pymodule import Qt


class projectViewDialog(QtWidgets.QDialog):
    def __init__(self, parent = None):
        QtWidgets.QDialog.__init__(self, parent)

        self.resize(1280, 800)
        self.setWindowTitle( 'Archive To Inventory' )

        # set ui
        self.projectList = QtGui.QListWidget(self)
        self.projectList.setGeometry(10, 10, 1260, 780)
        self.projectList.setViewMode(QtGui.QListView.IconMode)
        self.projectList.setIconSize(QtCore.QSize(310, 200))
        self.projectList.setGridSize(QtCore.QSize(310, 220))
        self.projectList.setMovement(QtGui.QListView.Static)

        self.projectList.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.projectList.customContextMenuRequested.connect(self.rmbClicked)

        # try:
        self.cachePath = mari.projects.cachePath()
        for project in mari.projects.list():
            # print(project.name(), i.uuid())
            item = QtGui.QListWidgetItem()
            item.setText(project.name())
            item.setData(QtCore.Qt.UserRole, project.uuid())
            print(os.path.join(self.cachePath, project.uuid(), 'ScreenShot.png'))
            item.setIcon(QtGui.QIcon(os.path.join(self.cachePath, project.uuid(), 'ScreenShot.png')))
            self.projectList.addItem(item)

        self.show()
        # except:
        #     self.cachePath = '/home/daeseok.chae/Music'
        #     for project in os.listdir(self.cachePath):
        #         # print(project.name(), i.uuid())
        #         # try:
        #         item = QtGui.QListWidgetItem()
        #         item.setText(project)
        #         if os.path.exists(os.path.join(self.cachePath, project, 'ScreenShot.png')):
        #             item.setIcon(QtGui.QIcon(os.path.join(self.cachePath, project, 'ScreenShot.png')))
        #             self.projectList.addItem(item)
        #         # except:
        #         #     pass

        # mari.projects.archive('0fef277b-3161-4b2e-a839-ed47d243b8a6', '/home/daeseok.chae/Desktop/babaolianhuankou')

    def rmbClicked(self, pos):
        menu = QtGui.QMenu(self)
        menu.addAction("== Export Archive ==", self.exportArchive)
        menu.addAction("== Rename Project ==", self.renameProject)
        menu.exec_(Qt.QtGui.QCursor.pos())

    def renameProject(self):
        oldName = self.projectList.currentItem().text()
        newName = ""

        # input new Name
        dialog = QtWidgets.QDialog(self)
        dialog.setGeometry(0, 0, 340, 40)
        dialog.setWindowTitle("Rename Project")

        label = QtGui.QLabel(dialog)
        label.setText("replace Name")
        label.setGeometry(10, 10, 90, 20)

        newNameLineEdit = QtGui.QLineEdit(dialog)
        newNameLineEdit.setGeometry(110, 10, 150, 20)

        okButton = QtGui.QPushButton(dialog)
        okButton.setText("Replace")
        okButton.clicked.connect(dialog.close)
        okButton.setGeometry(270, 10, 60, 20)

        dialog.exec_()

        mari.projects.rename(oldName, newName)

    def exportArchive(self):
        print(self.projectList.currentItem().text())
        DBIP = dxConfig.getConf("DB_IP")
        client = MongoClient(DBIP)
        dbName = "inventory"
        collName = 'assets'
        database = client[dbName]
        coll = database[collName]

        checkDB = {}
        checkDB["name"] = self.projectList.currentItem().text()
        checkDB["type"] = "ASSET_SRC"

        findItem = coll.find(checkDB)
        if findItem.count() == 1:
            item = findItem.__getitem__(0)
            print('objID :', item['_id'])

            # if exists mra file
            if item['files'].get('mari'):
                print('')
                dialog = QtGui.QMessageBox()
                dialog.setFont(QtGui.QFont("Cantarell", 13))
                dialog.setText('file is exists\ndo you want overwrite?')
                dialog.setWindowTitle('overwrite warning')
                dialog.setIcon(QtGui.QMessageBox.Warning)
                dialog.addButton("Yes", QtGui.QMessageBox.YesRole)
                dialog.addButton("No", QtGui.QMessageBox.NoRole)

                result = dialog.exec_()
                print(result, QtGui.QMessageBox.YesRole)
                # result => Yes : 1 | No : 0
                if result == 1:
                    return

            ProgressDialog.instance = ProgressDialog()
            self.Progress = ProgressDialog.instance
            self.Progress.show()

            self.Progress.progress_text.setText('Archive Project')
            self.Progress.pbar.setValue(0)
            mari.app.processEvents()

            # make Archive
            print(self.projectList.currentItem().data(QtCore.Qt.UserRole))
            tempArchivePath = '/dexter/Cache_DATA/RND/daeseok/.archiveTempRepository/{0}.mra'.format(self.projectList.currentItem().text())
            mari.projects.archive(self.projectList.currentItem().data(QtCore.Qt.UserRole), tempArchivePath)

            print(item['files']['preview'])
            assetLibDirPath = os.path.dirname(item['files']['preview'])

            archivePath = os.path.join(assetLibDirPath, "mari", os.path.basename(tempArchivePath))
            print(tempArchivePath + " => " + archivePath)

            # Move Archive Job
            job = author.Job()
            job.title = str('(Mari Archive) %s' % self.projectList.currentItem().text())
            job.service = 'Cache'
            job.maxactive = 1
            job.tier = 'cache'
            job.projects = ['export']

            # directory mapping
            job.newDirMap(src='S:/', dst='/show/', zone='NFS')
            job.newDirMap(src='N:/', dst='/netapp/', zone='NFS')
            job.newDirMap(src='R:/', dst='/dexter/', zone='NFS')

            mainTask = author.Task(title='main Task')
            mainTask.serialsubtasks = 1

            if not os.path.exists(os.path.dirname(archivePath)):
                command = "mkdir {0}".format(os.path.dirname(archivePath))
                print(command)
                mainTask.addCommand(author.Command(argv=command, service='Cache'))

            command = "mv -vf {0} {1}".format(tempArchivePath, archivePath)
            print(command)
            mainTask.addCommand(author.Command(argv=command, service='Cache'))

            # '/netapp/backstage/pub/bin/inventory/updateMariPath.py'
            command = "python /netapp/backstage/pub/bin/inventory/updateMariPath.py {0} {1} {2} {3}".format(dbName,
                                                                                                    collName,
                                                                                                    item['_id'],
                                                                                                     archivePath)

            print(command)
            # DB Write Archive
            mainTask.addCommand(author.Command(argv=command, service='Cache', tags=['py']))
            job.addChild(mainTask)
            job.priority = 999

            author.setEngineClientParam(hostname=dxConfig.getConf("TRACTOR_CACHE_IP"),
                                        port=dxConfig.getConf("TRACTOR_PORT"),
                                        user=getpass.getuser(),
                                        debug=True)
            job.spool()
            author.closeEngineClient()

            self.Progress.close()

            mari.utils.message('"%s" job spool\n' % self.projectList.currentItem().text())

        elif findItem.count() > 1:
            # select Item
            for item in findItem:
                print(item['project'] + '/' + item['category'])
        else:
            print("not exists, try archive export after inventory upload please")
            print("do you want rename?")
            dialog = QtGui.QMessageBox()
            dialog.setFont(QtGui.QFont("Cantarell", 13))
            dialog.setText('file not exists in inventory\n [checkList]\n  - file checked in inventory\n  - if file exists in inventory, rename project name')
            dialog.setWindowTitle('file exists warning')
            dialog.setIcon(QtGui.QMessageBox.Warning)
            dialog.addButton("Ok", QtGui.QMessageBox.AcceptRole)

            result = dialog.exec_()

class ProgressDialog( QtWidgets.QDialog ):
    instance = None
    def __init__( self ):
        super( ProgressDialog, self ).__init__()
        self.setWindowTitle( 'Archive Progress' )
        self.resize( 400, 50 )

        self.v_layout = QtGui.QVBoxLayout()
        self.setLayout( self.v_layout )

        #self.cancel_button = QtGui.QPushButton( 'Cancel' )
        #mari.utils.connect( self.cancel_button.clicked, lambda: self.cancel() )

        self.pbar = QtGui.QProgressBar( self )
        self.progress_text = QtGui.QLabel( self )
        self.progress_text.setText( 'Preparing to export ...' )
        self.pbar.setValue(0)

        self.v_layout.addWidget( self.pbar )
        self.v_layout.addWidget( self.progress_text )
        #self.v_layout.addWidget( self.cancel_button )

    def cancel( self ):
        global g_export_cancelled
        g_export_cancelled = True

def main():
    # app = QtGui.QApplication(sys.argv)
    global astPubMain
    astPubMain= projectViewDialog()
    print("wing")

    # global projectViewDialog
    # ArchiveDialogWin = projectViewDialog()
    # sys.exit(app.exec_())
