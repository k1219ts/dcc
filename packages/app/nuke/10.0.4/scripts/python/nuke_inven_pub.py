# -*- coding: utf-8 -*-

from pymodule import Qt
from pymodule.Qt import QtWidgets
from pymodule.Qt import QtGui
from pymodule.Qt import QtCore

try:
    import os, sys, nuke, subprocess, shutil
    from pymongo import MongoClient
    import pymongo

    import tractor.api.author as author
    import getpass, datetime, pprint

    import ui_nuke_inven_pub
    reload(ui_nuke_inven_pub)
except:
    pass
# ------------------------------------------------------------------------------

def nodeNumSort(node):
    nodeNum = node.name().replace(node.Class(), '')
    return int(nodeNum)

class PubWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        self.ui = ui_nuke_inven_pub.Ui_Form()
        self.ui.setupUi(self)
        self.ui.thumbLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.ui.allPub.setEnabled(False)
        self.ui.selectedPub.setEnabled(False)
        # ------------------------------------------------------------------------------
        #self.ui.versionSpinBox.setMinimumHeight(20)

        #self.basePath = '/dexter/Cache_DATA/comp/TD_hslth/asset_test/spool'
        self.basePath = '/assetlib/reference/fx'

        self.ffmpegPath = '/netapp/backstage/pub/apps/ffmpeg_for_exr/bin/ffmpeg_with_env'
        #self.exrOption = '-apply_trc gamma22'
        self.exrOption = '-apply_trc iec61966_2_1'
        self.dpxOption = '-apply_trc log_sqrt'

        self.DBIP = "10.0.0.12:27017, 10.0.0.13:27017"
        self.tmpThumbFile = '/tmp/test.jpg'

        #self.delay = QtCore.qtim

        self.ui.titleLineEdit.setText(os.path.basename(nuke.root().name()).replace('.nk', ''))

        self.ui.readCombo.currentIndexChanged.connect(self.changeReadNode)
        self.ui.frameSlider.sliderMoved.connect(self.refreshThumbnail)
        #self.ui.frameSlider.sliderMoved.connect(self.sliderDelay)

        self.ui.hipSearchButton.clicked.connect(self.hipSearchDialog)
        self.ui.closeButton.clicked.connect(self.close)
        self.ui.publishButton.clicked.connect(self.publish)

        # READ NODE SETTING
        for rn in sorted(nuke.allNodes('Read'), key=nodeNumSort):
            self.ui.readCombo.addItem(rn.name())

        # PREFIX SETTING
        self.rootName = ''
        if nuke.root().name().startswith('/netapp/show'):
            self.rootName = nuke.root().name().replace('/netapp/show', '')
        else:
            self.rootName = nuke.root().name()

        # CATEGORY SETTING
        self.ui.typeComboBox.addItems(['COMP_SRC', 'FX_REF'])
        self.ui.typeComboBox.currentIndexChanged.connect(self.categoryChanged)
        self.ui.prjComboBox.currentIndexChanged.connect(self.prjChanged)
        self.categoryChanged(0)
        self.ui.categoryComboBox.setEnabled(True)
        self.ui.categoryLineEdit.setEnabled(True)
        self.ui.hipLine.setEnabled(False)
        self.ui.hipSearchButton.setEnabled(False)

        # PROJECT MANUAL
        self.ui.prjLineEdit.setEnabled(False)


    def categoryChanged(self, index):
        self.ui.prjComboBox.clear()
        category = self.ui.typeComboBox.itemText(index)
        if category == 'COMP_SRC':
            self.ui.categoryComboBox.setEnabled(True)
            self.ui.categoryLineEdit.setEnabled(True)
            self.ui.hipLine.setEnabled(False)
            self.ui.hipSearchButton.setEnabled(False)

        elif category == 'FX_REF':
            self.ui.categoryComboBox.setEnabled(False)
            self.ui.categoryLineEdit.setEnabled(False)
            self.ui.hipLine.setEnabled(True)
            self.ui.hipSearchButton.setEnabled(True)


        client = MongoClient(self.DBIP)
        db = client['inventory']
        coll = db['assets']
        self.ui.prjComboBox.addItems(sorted(coll.find({'type': category})
                                            .distinct('project')))
        self.ui.prjComboBox.addItem(u'직접입력')

    def prjChanged(self, index):
        category = self.ui.typeComboBox.currentText()
        project = self.ui.prjComboBox.itemText(index)

        client = MongoClient(self.DBIP)
        db = client['inventory']
        coll = db['assets']
        self.ui.categoryComboBox.clear()
        self.ui.categoryComboBox.addItems(sorted(coll.find({'type': category,
                                                       'project':project})
                                            .distinct('category')))


        if project == u'직접입력':
            self.ui.prjLineEdit.setEnabled(True)
            self.ui.prjLineEdit.setFocus(QtCore.Qt.MouseFocusReason)
            self.ui.categoryComboBox.clear()

        self.ui.categoryComboBox.addItem(u'직접입력')

    def sliderDelay(self, frame):

        pass

    def hipSearchDialog(self):
        #'/'.join(nk.split('/')[:-3] + ['scenes', 'houdini'])

        if nuke.root().name().count('/') > 3:
            basePath = '/'.join(nuke.root().name().split('/')[:-3] + ['scenes'])
            if not(os.path.exists(basePath)):
                basePath = os.path.dirname(nuke.root().name())

        else:
            basePath = os.path.dirname(nuke.root().name())

        hipFile = QtWidgets.QFileDialog.getOpenFileName(self,
                                                        'Select Hip File',
                                                        basePath,
                                                        "Houdini (*.hip)")
        if hipFile[0]:
            self.ui.hipLine.setText(hipFile[0])

    def refreshThumbnail(self, frame):
        rn = nuke.toNode(self.ui.readCombo.currentText())

        rnFile = rn['file'].value()
        first = rn['first'].value()
        last = rn['last'].value()

        oc = nuke.OutputContext()
        oc.setFrame(frame)

        filename = rn['file'].getEvaluatedValue(oc)
        ext = os.path.splitext(rnFile)[-1]
        qtExt = ['.jpg', '.jpeg','.png', '.tiff', '.tif']
        videoExt = ['.mov', '.avi', '.mp4']

        #if ext == '.jpg':
        if ext in qtExt:
            self.ui.thumbLabel.setPixmap(QtGui.QPixmap(filename).scaled(QtCore.QSize(320,180)))

        elif ext in videoExt:
            # -ss thumbnail point
            # -s size
            time = frame / 24.0

            cmd =  '%s -i %s ' % (self.ffmpegPath, filename)
            cmd += '-f image2 -vframes 1 -s 320x180 '
            cmd += '-ss %f ' % time
            cmd += '-y %s' % self.tmpThumbFile
            ok = os.system(cmd)
            if ok == 0:
                pixmap = QtGui.QPixmap( self.tmpThumbFile)
                pixmap = pixmap.scaled(QtCore.QSize(320, 180),
                                       QtCore.Qt.KeepAspectRatio)
                self.ui.thumbLabel.setPixmap(pixmap)
            else:
                self.ui.thumbLabel.setPixmap(None)
                self.ui.thumbLabel.setText("Can't make thumbnail")

        elif ext == '.exr':
            cmd =  '%s' % self.ffmpegPath
            cmd += ' %s -i ' % self.exrOption
            cmd += filename
            cmd += ' -y %s' % self.tmpThumbFile
            isError = os.system(cmd)
            print isError
            if isError:
                cmd =  "convert -colorspace sRGB -quality 100 -resize 320x180 "
                cmd += "%s %s" % (filename, self.tmpThumbFile)
                os.system(cmd)

            pixmap = QtGui.QPixmap(self.tmpThumbFile)
            pixmap = pixmap.scaled(QtCore.QSize(320, 180),
                                   QtCore.Qt.KeepAspectRatio)
            self.ui.thumbLabel.setPixmap(pixmap)
        elif ext == '.dpx':
            cmd = "convert -resize 320x180 "
            cmd += "%s -set colorspace Log -colorspace sRGB %s" % (filename, self.tmpThumbFile)
            os.system(cmd)
            pixmap = QtGui.QPixmap(self.tmpThumbFile)
            pixmap = pixmap.scaled(QtCore.QSize(320, 180),
                                   QtCore.Qt.KeepAspectRatio)
            self.ui.thumbLabel.setPixmap(pixmap)

        # OTHER? TIFF, PNG, PSD
        else:
            cmd =  '%s ' % self.ffmpegPath
            cmd += '-i %s ' % filename
            cmd += '-s 320x180 '
            cmd += '-y %s' % self.tmpThumbFile
            ok = os.system(cmd)
            if ok == 0:
                pixmap = QtGui.QPixmap( self.tmpThumbFile)
                pixmap = pixmap.scaled(QtCore.QSize(320, 180),
                                       QtCore.Qt.KeepAspectRatio)
                self.ui.thumbLabel.setPixmap(pixmap)
            else:
                self.ui.thumbLabel.setPixmap(None)
                self.ui.thumbLabel.setText("Can't make thumbnail")
        self.ui.frameLabel.setText(str(frame))


    def changeReadNode(self, index):

        rn = nuke.toNode(self.ui.readCombo.currentText())

        # TAG REFRESH
        self.ui.tagTextEdit.clear()
        tags = os.path.dirname(nuke.root().name()).split('/')
        for el in os.path.dirname(rn['file'].value()).split('/'):
            if '_' in el:
                tags += el.split('_')
            else:
                tags.append(el)
        tags += self.ui.titleLineEdit.text().split('_')

        tags = list(set(tags))
        tagExceptList = ['netapp','show','script', 'render','shot', 'images',
                         'Cache_DATA','dexter', '',
                         '','']
        for tag in tags:
            if tag and not(tag in tagExceptList):
                self.ui.tagTextEdit.append(tag)
                #self.ui.tagTextEdit.insertPlainText('#'+tag)

        rn = nuke.toNode(self.ui.readCombo.currentText())

        rnFile = rn['file'].value()
        first = rn['first'].value()
        last = rn['last'].value()
        mid = first + ((last - first) / 2)
        self.ui.frameSlider.setMinimum(first)
        self.ui.frameSlider.setMaximum(last)

        self.ui.frameSlider.setValue(mid)
        self.ui.frameLabel.setText(str(mid))
        self.refreshThumbnail(mid)

    def getCPcommand(self, nodes, destRoot):
        copyRootTask = author.Task(title="COPY ROOT")

        # DUPLATECATE CHECK
        tempOrg = []
        duList = []

        for r in nodes:
            if r['file'].value() in tempOrg:
                duList.append(r)
            else:
                tempOrg.append(r['file'].value())

        nodes = [item for item in nodes if item not in duList]

        # START MAKING COPY NODES
        for n in nodes:
            # IF NO FILE KNOB: CONTINUE
            if not(n.knob('file')):
                continue
            filePath = n['file'].value()
            if filePath.startswith('/assetlib/'):
                continue
            fileName = os.path.basename(filePath)

            stf = n['first'].value()
            edf = n['last'].value()

            bracketStr = '[0-9]*'

            # COUNT # AND REPLACE # TO {FRAMES.FRAMES}
            if '#' in fileName:
                cpFileName = fileName.replace('#'*fileName.count('#'), bracketStr)
                cpFilePath = os.path.join(os.path.dirname(filePath), cpFileName)

            # CHECK IF %0? AND REPLACE IT TO {FRAMES.FRAMES}
            elif '%0' in fileName:
                point = fileName.find('%0')
                ptarget = fileName[point:point+4]
                cpFileName = fileName.replace(ptarget, bracketStr)
                cpFilePath = os.path.join(os.path.dirname(filePath), cpFileName)

            # SINGLE FILE COPY
            else:
                cpFilePath = filePath

            dest = os.path.join(destRoot, os.path.basename(os.path.dirname(cpFilePath)))

            #cmd = '/netapp/backstage/pub/bin/seqcp %s %s' % (cpFilePath, dest)
            cmd = 'python /netapp/backstage/pub/bin/inventory/seqcp.py %s %s' % (cpFilePath, dest)
            copyCmd = author.Command(argv=str(cmd))

            copyTask = author.Task(title='COPY %s' % n.name(),
                                   service='nuke')
            copyTask.addCommand(copyCmd)

            copyRootTask.addChild(copyTask)

        return copyRootTask


    def publish(self):
        print "publish"
        # ------------------------------------------------------------------------------
        # PREPARE

        rn = nuke.toNode(self.ui.readCombo.currentText())
        assetName = self.ui.titleLineEdit.text()
        assetType = self.ui.typeComboBox.currentText()

        if assetType == 'COMP_SRC':
            category = self.ui.categoryComboBox.currentText()
            if category == u'직접입력':
                category = self.ui.categoryLineEdit.text()
        else:
            category = None

        project = self.ui.prjLineEdit.text()
        if project == u'직접입력':
            project = self.ui.prjLineEdit.text()

        tags = self.ui.tagTextEdit.toPlainText()
        tagList = []
        for tag in tags.split('\n'):
            tagList.append(tag)

        if not(project):
            project = 'no_project'
        elif ' ' in project:
            project = project.replace(' ', '_')
        self.ui.thumbLabel.pixmap().save(self.tmpThumbFile)

        # ------------------------------------------------------------------------------
        projectPath = os.path.join(self.basePath, project)
        assetPath = os.path.join(projectPath, assetName)

        movPath = os.path.join(assetPath, assetName + '_mov.mov')
        thumbPath = os.path.join(assetPath, assetName + '_thumb.jpg')
        gifPath = os.path.join(assetPath, assetName + '_gif.gif')
        nkPath = os.path.join(assetPath, assetName + '_nk.nk')
        if self.ui.hipLine.text() and os.path.exists(self.ui.hipLine.text()):
            hipPath = os.path.join(assetPath, assetName + '_hip.hip')
        else:
            hipPath = ''

        tmpBase = '/dexter/Cache_DATA/comp/render_script/reference_publish'
        if not(os.path.exists(tmpBase)):
            os.makedirs(tmpBase)

        tmpNkPath = os.path.join(tmpBase, project + '_' + assetName + '_nk.nk')
        tmpThumbPath = os.path.join(tmpBase, project + '_' + assetName + '_thumb.jpg')
        dbRecord = {'name'    :assetName,
                    'project' :project,
                    'tags': tagList,
                    'files'    :{'nuke':nkPath,
                                 'houdini':hipPath,
                                 'thumbnail': thumbPath,
                                 'gif': gifPath,
                                 'mov': movPath,
                                 'preview':movPath
                                 },
                    'enabled' :False,
                    'user' : getpass.getuser(),
                    'time' : datetime.datetime.now().isoformat(),
                    'type' : 'FX_REF'
                    }
        if category:
            dbRecord['category'] = category

        client = MongoClient(self.DBIP)
        db = client.inventory
        coll = db.assets

        checkDB = {}
        checkDB['name'] = dbRecord['name']
        checkDB['project'] = dbRecord['project']
        #checkDB['enabled'] = True

        isDBExists = coll.find(checkDB).limit(1)
        if isDBExists.count():
            QtWidgets.QMessageBox.warning(self,
                                          'name duplicate',
                                          'name duplicate'
                                          )
            return
        else:
            result_id = coll.insert_one(dbRecord)

        svc = 'Cache'
        # ------------------------------------------------------------------------------
        # JOB CREATE
        jobTitle = '(Reference) Pub_%s' % str(assetName)

        job = author.Job(title=jobTitle,
                         priority=10000,
                         service=svc)
        job.tier = 'batch'
        job.projects = ['export']

        # ------------------------------------------------------------------------------
        # ROOT TASK FOR SERIAL
        emptyRootTask = author.Task(title='Empty for Serial')
        emptyRootTask.serialsubtasks = 1
        job.addChild(emptyRootTask)
        # ------------------------------------------------------------------------------

        # CP TASK
        fileNodes = []
        fileNodes += nuke.allNodes('Read')
        fileNodes += nuke.allNodes('DeepRead')
        # fileNodes += nuke.allNodes('ReadGeo')
        # fileNodes += nuke.allNodes('Camera2')
        # POSSIBLY MORE FILE KNOB??


        # REPLACE FILE KNOB VALUE
        nodePathDic = {}
        for i in fileNodes:
            nodePathDic[i.name()] = i['file'].value()

            if i['file'].value().startswith('/assetlib/'):
                continue

            targetPath = '/'.join(i['file'].value().split('/')[-2:])
            backupPath = os.path.join(assetPath, targetPath)
            i['file'].setValue(backupPath)

        # SCRIPT SAVE TO BACKUP NUKE PATH
        nuke.scriptSave(tmpNkPath)
        for i in fileNodes:
            i['file'].setValue(nodePathDic[i.name()])

        # CREATE DIR PATH IF NOT EXISTS
        dirTask = author.Task(title="directory create command",
                              argv=['install', '-d', '-m', '755', assetPath],
                              #argv=str('mkdir -p %s' % assetPath),
                              service=svc)
        emptyRootTask.addChild(dirTask)

        # MOVE NUKE FILE TO ASSET STORAGE
        nkMoveTask = author.Task(title="nk file move command",
                                 argv=str('mv %s %s' % (tmpNkPath, nkPath)),
                                 service=svc)

        # COPY HIP FILE TO ASSET STORAGE
        if hipPath:
            hipCopyCmd = author.Command(argv=str("cp %s %s" % (self.ui.hipLine.text(),
                                                               hipPath))
                                        )
            nkMoveTask.addCommand(hipCopyCmd)

        emptyRootTask.addChild(nkMoveTask)

        # COPY READ NODE TASK
        copyTasks = self.getCPcommand(fileNodes, assetPath)
        emptyRootTask.addChild(copyTasks)

        # THUMBNAIL CREATE TASK
        shutil.copy(self.tmpThumbFile, tmpThumbPath)
        thumbMoveTask = author.Task(title="thumbnail file move command",
                                 argv=str('mv %s %s' % (tmpThumbPath, thumbPath)),
                                 service=svc)
        emptyRootTask.addChild(thumbMoveTask)

        # MOV CREATE TASK
        if rn['file'].value().endswith('mov'):
            # IF SELECTED READ NODE IS MOV FILE:
            movCmd = ['cp', '-f', rn['file'].value(), movPath]

        else:
            movCmd =  [self.ffmpegPath, '-r', '24', '-start_number', str(rn['first'].value())]
            if rn['file'].value().endswith('exr'):
                movCmd += ['-apply_trc', 'gamma22']

            movCmd += ['-i',rn['file'].value(),'-r','24','-an','-vcodec','libx264']
            movCmd += ["-pix_fmt" ,"yuv420p","-preset","slow","-profile:v","baseline","-b","30000k","-tune","zerolatency"]
            movCmd += ['-y', movPath]

        movTask = author.Task(title="mov create command",
                              argv=str(' '.join(movCmd)),
                              service=svc)

        emptyRootTask.addChild(movTask)

        # GIF CREATE TASK
        gifCmd = '/netapp/backstage/pub/bin/inventory/gifCreate %s %s' % (movPath, gifPath)
        gifTask = author.Task(title="gif create command",
                              argv=str(gifCmd),
                              service=svc)
        emptyRootTask.addChild(gifTask)

        # DB RECORD TASK

        dbCmd = 'python /netapp/backstage/pub/bin/inventory/enableDBRecord.py %s %s %s' % ('inventory',
                                                                                 'assets',
                                                                                 result_id.inserted_id)
        dbTask = author.Task(title="db record command",
                             argv=dbCmd,
                             service=svc)
        emptyRootTask.addChild(dbTask)

        # SEND JOB TO TRACTOR
        print job.asTcl()

        renderAlfredFile = os.path.join(tmpBase, project + '_' + assetName + '_temp.alf')
        f = open(renderAlfredFile, 'w')
        f.write(job.asTcl())
        f.close()

        TRACTOR = '/netapp/backstage/pub/apps/tractor/linux/Tractor-2.2/bin/tractor-spool'
        #options = '--engine=10.0.0.106:80 --user=%s' % getpas.getuser()
        options = '--engine=10.0.0.30:80 --user=%s' % getpass.getuser()

        cmd = '%s %s --priority=%s %s' % ( TRACTOR,
                                           options,
                                           '10000',
                                           renderAlfredFile )
        print cmd
        print os.system(cmd)
        nuke.message('sent job.')
        self.close()

if __name__ == "__main__":
    try:
        import dev_cp
        dev_cp.pub()
    except:
        pass
"""
import nuke
from PySide2 import QtCore, QtGui
import nuke_inven_pub
reload(nuke_inven_pub)

def tata():
    pubWidget = nuke_inven_pub.PubWidget(QtWidgets.QApplication.activeWindow())
    pubWidget.setWindowFlags(QtCore.Qt.Dialog)
    pubWidget.show()

tata()
"""
