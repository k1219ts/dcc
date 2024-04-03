# -*- coding: utf-8 -*-
from PySide2 import QtWidgets, QtCore, QtGui

import sys, os, json, shutil
import nuke, nukescripts

from ui_renderHistory import Ui_Form

class RenderHistory(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(RenderHistory, self).__init__()
        self.setWindowTitle('Render History By Tae Hyung Lee, Dexter Digital')

        self.ui = Ui_Form()
        self.ui.setupUi(self)
        #------------------------------------------------------------------------------
        prjSeqFile = '/stdrepo/CMP/TD_hslth/Global_File/prj_seq.json'
        prjSeq = json.load(open(prjSeqFile, 'r'))
        self.prjDic = prjSeq['prj_code']
        self.prjStereoDic = prjSeq['isStereo']
        self.seqDic = prjSeq['prj_seq']

        self.fileNameDic = {'comp':'comp_v001.nk', 'lighting':'lgt_v01_w01.nk', 'fx':'fx_v010.nk'}

        self.ui.comboBox.currentIndexChanged.connect(self.setSeq)
        self.ui.comboBox_3.currentIndexChanged.connect(self.setImportPath)

        self.ui.pushButton.clicked.connect(self.searchShot)
        self.ui.pushButton_2.clicked.connect(self.importOpen)

        self.ui.lineEdit.returnPressed.connect(self.searchShot)
        self.ui.lineEdit.textEdited.connect(self.setImportPath)

        self.ui.treeWidget.itemDoubleClicked.connect(self.executeNuke)


        for i in sorted(self.prjDic.keys(), reverse=False):
            self.ui.comboBox.addItem(i, self.prjDic[i])
        #------------------------------------------------------------------------------
        self.uiSetting()


    def uiSetting(self):
        #------------------------------------------------------------------------------
        self.widgetFont = self.font()
        self.widgetFont.setPointSize(12)
        self.setFont(self.widgetFont)
        #------------------------------------------------------------------------------

        self.ui.treeWidget.setColumnCount(3)
        self.ui.treeWidget.headerItem().setText(0, 'File')
        self.ui.treeWidget.headerItem().setText(1, 'Version')
        self.ui.treeWidget.headerItem().setText(2, 'Date')

        self.ui.treeWidget.header().resizeSection(0, 420)
        self.ui.treeWidget.header().resizeSection(1, 100)
        self.ui.treeWidget.header().resizeSection(2, 85)

        self.ui.treeWidget.setSortingEnabled(True)
        self.ui.treeWidget.sortByColumn(2, QtCore.Qt.DescendingOrder)
        self.ui.treeWidget.setDragEnabled(True)
        #------------------------------------------------------------------------------
        self.ui.groupBox_2.setStyleSheet("""
        QGroupBox{
        margin-top: 8px;
        border: 2px solid green;
        }
        QGroupBox::title {
        subcontrol-origin: margin;
        subcontrol-position: top center;
        }
        """)

        self.ui.groupBox.setStyleSheet("""
        QGroupBox{
        margin-top: 8px;
        border: 2px solid rgb(182,73,38);
        }
        QGroupBox::title {
        subcontrol-origin: margin;
        subcontrol-position: top center;
        }
        """)
        #------------------------------------------------------------------------------
        self.ui.comboBox_3.addItem('FX', 'dev/precomp/nuke')
        self.ui.comboBox_3.addItem('Comp', 'comp/script')
        self.ui.comboBox_3.addItem('Lighting', 'dev/precomp/nuke')

    def setImportPath(self, index):
#       /show/tisf/shot/DTF/DTF_0110/fx/dev/precomp/nuke
#       /show/tisf/shot/DTF/DTF_0020/lighting/dev/precomp/nuke

        currentPrj = self.ui.comboBox.currentText()
        currentSeq = self.ui.comboBox_2.currentText()
        process = self.ui.comboBox_3.currentText().lower()
        shotNum = str(self.ui.lineEdit.text())

        shotPath = '/show/%s/shot/%s/%s' % (self.prjDic[currentPrj], currentSeq,
                                             currentSeq + '_' + shotNum)
        processPath = process + '/' + self.ui.comboBox_3.itemData(self.ui.comboBox_3.currentIndex())

        self.ui.lineEdit_2.setText('%s/%s' % (shotPath, processPath))
        self.ui.lineEdit_3.setText('%s_%s' % (currentSeq + '_' + shotNum,
                                              self.fileNameDic[process]))


    def setSeq(self, index):
        self.ui.comboBox_2.clear()
        self.ui.comboBox_2.addItems(self.seqDic[str(self.ui.comboBox.itemData(self.ui.comboBox.currentIndex()))])

    def searchShot(self):
        self.ui.treeWidget.clear()
        basePath = '/dexter/Cache_DATA/comp/render_script'
        currentPrj = self.ui.comboBox.currentText()
        currentSeq = self.ui.comboBox_2.currentText()
        shotNum = str(self.ui.lineEdit.text())

        shotPath = '%s/%s/%s/%s' % (basePath, self.prjDic[currentPrj], currentSeq,
                                    currentSeq + '_' + shotNum)

        nkFiles = QtCore.QDir(shotPath).entryInfoList(['*.nk'])
        contextDic = {}


        for nk in nkFiles:
            fileName = nk.fileName()
            rDate = nk.lastModified()
            if fileName.count('_') > 3:
                context = fileName.split('_')[2]
                version = fileName.split('_')[3]


                if contextDic.get(context):
                    contextItem = contextDic[context]

                elif context.startswith('v'):
                    contextItem = self.ui.treeWidget
                    version = context

                else:
                    contextItem = QtWidgets.QTreeWidgetItem(self.ui.treeWidget)
                    contextItem.setText(0, context)
                    tempFont = contextItem.font(0)
                    tempFont.setPointSize(20)
                    contextItem.setFont(0,tempFont)
                    contextDic[context] = contextItem
            else:
                if contextDic.get('etc'):
                    contextItem = contextItem = contextDic['etc']
                else:
                    contextItem = QtWidgets.QTreeWidgetItem(self.ui.treeWidget)
                    contextItem.setText(0, 'ETC')
                    tempFont = contextItem.font(0)
                    tempFont.setPointSize(20)
                    contextItem.setFont(0,tempFont)
                    contextDic['etc'] = contextItem

                version = ''

            item = QtWidgets.QTreeWidgetItem(contextItem)
            item.setText(0, nk.fileName())
            item.setText(1, version)
            item.setText(2, rDate.toString('yyyy/MM/dd hh:mm:ss'))
            item.setTextAlignment(1, QtCore.Qt.AlignCenter)
            item.setData(0, QtCore.Qt.UserRole, nk.absoluteFilePath())
        #------------------------------------------------------------------------------
#        compBasePath = '/show/%s/shot/%s/%s/comp/comp/script' % (self.prjDic[currentPrj], currentSeq,
#                                                                 currentSeq + '_' + shotNum)
        compBasePath = '/show/%s/shot/%s/%s/comp/pub/base_script' % (self.prjDic[currentPrj], currentSeq,
                                                                     currentSeq + '_' + shotNum)

        compNkFiles = QtCore.QDir(compBasePath).entryInfoList(['*.nk'])
        if compNkFiles:
            compNkRoot = QtWidgets.QTreeWidgetItem(self.ui.treeWidget)
            compNkRoot.setText(0, 'Comp Base')
            compNkRoot.setFirstColumnSpanned(True)
            compNkRoot.setBackground(0, QtGui.QBrush(QtCore.Qt.darkGreen))
            compNkRoot.setForeground(0, QtGui.QBrush(QtCore.Qt.black))
            compNkRoot.setFont(0, tempFont)


            for compNk in compNkFiles:
                fileName = compNk.fileName()
                context = fileName.split('_')[2]
                version = fileName.split('_')[3].split('.')[0]
                rDate = compNk.lastModified()

                item = QtWidgets.QTreeWidgetItem(compNkRoot)
                item.setText(0, compNk.fileName())
                item.setText(1, version)
                item.setText(2, rDate.toString('yyyy/MM/dd hh:mm:ss'))
                item.setData(0, QtCore.Qt.UserRole, compNk.absoluteFilePath())


    def executeNuke(self, item, column):
        pass
#        filePath = item.data(0, QtCore.Qt.UserRole)
#        if filePath:
#            if filePath.startswith('/show/'):
#                QtGui.QMessageBox.information(self, "Warning", u"합성팀 원본은 더블클릭으로 열 수 없습니다.")
#                return
#
#            nuke.scriptOpen(filePath)

    def importOpen(self):
        if (self.ui.treeWidget.selectedItems()) and (self.ui.treeWidget.selectedItems()[0].parent()):

            filePath = self.ui.treeWidget.selectedItems()[0].data(0, QtCore.Qt.UserRole)

            print('from : ', filePath)
            print('to   : ', self.ui.lineEdit_2.text() + '/' + self.ui.lineEdit_3.text())

            #descPath = '/home/taehyung.lee/Documents/temp/import_test/' + self.ui.lineEdit_3.text()
            descPath = self.ui.lineEdit_2.text() + '/' + self.ui.lineEdit_3.text()

            if not(os.path.exists(os.path.dirname(descPath))):
                os.makedirs(os.path.dirname(descPath))

            shutil.copy(filePath, descPath)
            nuke.scriptOpen(descPath)

        else:
            QtWidgets.QMessageBox.information(self, "Warning", u"아이템이 선택되지 않았거나 파일아이템이 아닙니다.")



    def dragEnterEvent(self, event):

        if event.mimeData().hasText():
            return
        mimeText = ''
        for item in self.selectedItems():
            mimeText += 'file://' + str(item.data(0, QtCore.Qt.UserRole).toString()) + '\r\n'
        event.accept()

    def dragMoveEvent(self, event):
        #event.accept()
        pass

def publish_base_script():
    savedPath = nuke.value('root.name')
    if savedPath.startswith('/netapp/dexter'):
        savedPath = savedPath.replace('/netapp/dexter', '')

    if (savedPath.count('/') > 5) and (savedPath.startswith('/show/')):
        print(savedPath)
        splitedPath = savedPath.split('/')
        project = splitedPath[2]
        sequence = splitedPath[4]
        shotName = splitedPath[5]

        pubPath = '/show/%s/shot/%s/%s/comp/pub/base_script/' % (project, sequence, shotName)

        if not(os.path.exists(pubPath)):
            os.makedirs(pubPath)

        version = len(os.listdir(pubPath)) + 1
        pubFile = '%s_pub_v%s.nk' % (shotName, str(version).zfill(3))

        if nuke.ask('File Pub?\n' + pubPath + pubFile):
            shutil.copy(savedPath, pubPath + pubFile)
            nuke.message('File Pub\n' + pubPath + pubFile)



if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    ms = RenderHistory()
    ms.show()
    sys.exit(app.exec_())
