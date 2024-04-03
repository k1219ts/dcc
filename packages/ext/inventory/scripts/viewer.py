# -*- coding: utf-8 -*-
#from PyQt4 import QtCore, QtGui
from pymodule import Qt
from pymodule.Qt import QtWidgets
from pymodule.Qt import QtGui
from pymodule.Qt import QtCore
import pymodule.Qt as Qt

import db_query
import getpass, os, sys, subprocess, socket
import taskThread

from detailDialog import DetailDialog
from download_dialog import DownloadDialog
from items import AlembicItem
from items import ContainerWidget
from items import ImageItem
from items import ThumbnailItem
from user_config import GlobalConfig

isHoudini = False
try:
    import hou
    isHoudini = True
except:
    pass

# 20170911 daeseok.chae
import AssetDataProcess

import math

INVEN_TAIL = '@inven&'


def makeItems(listWidget, record):
    # MAKE TYPE BASED ITEM
    # TODO: MAKE ITEM BASED ON ITEM ITSELF (EX: COMP_SRC WITH ONLY 1 IMAGE)

    if record['type'] in ['COMP_SRC','COMP_ASSET', 'FX_REF', 'VFX_REF', 'PRV_SRC', 'MAT_SRC']:
        # CASE NORMAL PREVIEW ITEM
        if record['files'].has_key('gif'):
            # PREVIEW ITEM
            imgItem = ThumbnailItem()
            listWidget.addItem(imgItem)
            citem = ContainerWidget(record['files']['gif'],
                                    record['files']['thumbnail'])
            listWidget.setItemWidget(imgItem, citem)
            title = ""
            if record['files'].has_key('nuke'):
                title += "<img src=':nuke.png' width='20', height='20'>"
            if (record['files'].has_key('houdini')) and record['files']['houdini']:
                title += "<img src=':houdini.png' width='20', height='20'>"
            title += " " + record['name']
            citem.titleLabel.setText(title)
            imgItem.setItemData(record)

        # CASE SINGLE IMAGE ITEM
        else:
            imgItem = ImageItem(record)
            listWidget.addItem(imgItem)
            #citem = ContainerWidget(thumbnail=record['files']['thumbnail'])

        # listWidget.setItemWidget(imgItem, citem)
        # title = ""
        # if record['files'].has_key('nuke'):
        #     title += "<img src=':nuke.png' width='20', height='20'>"
        # if (record['files'].has_key('houdini')) and record['files']['houdini']:
        #     title += "<img src=':houdini.png' width='20', height='20'>"
        # title += " " + record['name']
        # citem.titleLabel.setText(title)
        # imgItem.setItemData(record)

    elif record['type'] in ['HDRI_SRC']:
        imgItem = ImageItem(record)
        listWidget.addItem(imgItem)

    elif record['type'] in ['ASSET_SRC']:
        imgItem = AlembicItem(record)
        listWidget.addItem(imgItem)
    else:
        imgItem = ImageItem(record)
        listWidget.addItem(imgItem)
    return imgItem

class TabWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(TabWidget, self).__init__(parent)
        self.gridLayout = QtWidgets.QGridLayout(self)
        self.gridLayout.setSpacing(0)
        self.gridLayout.setMargin(0)

        self.statusLabel = QtWidgets.QLabel(self)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding,
                                           QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.statusLabel.sizePolicy().hasHeightForWidth())
        self.statusLabel.setSizePolicy(sizePolicy)
        self.statusLabel.setAlignment(QtCore.Qt.AlignVCenter|QtCore.Qt.AlignRight)
        self.statusLabel.setStyleSheet("""
        QLabel{ color: rgb(247.095,146.88,30.09)};
        """)

        self.listWidget = BaseList(self)
        self.gridLayout.addWidget(self.listWidget, 0, 0, 1, 2)
        self.gridLayout.addWidget(self.statusLabel, 1, 0, 1, 1)
        self.moreButton = QtWidgets.QPushButton(self)
        self.moreButton.setText("Next")
        self.moreButton.setText("Next")
        self.gridLayout.addWidget(self.moreButton, 1, 1, 1, 1)

        self.progressDialog = QtWidgets.QProgressDialog("PROGRESSING", "Cancel", 0, 100, self)
        self.progressDialog.setWindowModality(QtCore.Qt.WindowModal)
        self.progressDialog.setWindowTitle("Progressing")
        self.progressDialog.setStyleSheet("""
        QProgressBar{border:2px solid grey; text-align: center; color: rgb(200,200,200); background-color: rgb(48,48,48)}
        QProgressBar::chunk{background-color: rgb(247.095,146.88,30.09)};
        """)
        self.progressDialog.close()

        self.dbRecord = []
        self.currentIndex = 0

        self.moreButton.clicked.connect(self.insertThumbnailItem)

    def insertThumbnailItem(self):
        leftItemCount = len(self.dbRecord) - self.currentIndex
        user_config = GlobalConfig.instance().config_dic
        if leftItemCount < user_config['item_per_click']:
            itemCount = leftItemCount
        else:
            itemCount = user_config['item_per_click']

        self.progressDialog.setMinimum(0)
        self.progressDialog.setMaximum(itemCount)
        self.progressDialog.show()

        for index, i in enumerate(self.dbRecord[self.currentIndex:self.currentIndex+itemCount]):
            self.progressDialog.setValue(index + 1)
            makeItems(self.listWidget, i)

        # listWidget.sortItems()
        self.progressDialog.close()

        self.currentIndex += itemCount

        statusText = 'Found %d items / ' % len(self.dbRecord)
        statusText += 'Show %d items ' % self.currentIndex
        self.statusLabel.setText(statusText)

    def insertSingleItem(self, listWidget, record):
        makeItems(listWidget, record)

    def setRecord(self, record):
        self.dbRecord = record

    def getRecord(self):
        return self.dbRecord


class BaseList(QtWidgets.QListWidget):
    assetClicked = QtCore.Signal(list)

    def __init__(self, parent=None):
        super(BaseList, self).__init__(parent)
        # TO CLASSIFY DRAG ENTER FROM
        self.viewerId = 'viewer'

        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setAcceptDrops(True)
        self.setDragEnabled(True)
        self.setViewMode(QtWidgets.QListView.IconMode)

        # IMAGE SIZE OF EACH ITEM / MUST SMALLER THAN ACTUAL ITEM SIZE
        self.setIconSize(QtCore.QSize(200,130))
        self.setSelectionMode(QtWidgets.QListView.ExtendedSelection)
        self.setResizeMode(QtWidgets.QListView.Adjust)
        #self.viewOptions().displayAlignment = QtCore.Qt.AlignCenter
        #self.setUniformItemSizes(True)
        self.setStyleSheet("""
        QListView{border: none; background: rgb(48,48,48); color: rgb(200,200,200);}
        QListView::item:selected {background: rgb(68,150,210)}
        QListView::item:!selected {background: rgb(48,48,48)}
        QListView::item:hover{background-color: rgb(240,169,32);}
        """)

        scrollbarStyle = 'QScrollBar{border: 1px solid black; background: #484848;} '
        scrollbarStyle = scrollbarStyle + 'QScrollBar::add-page  {border: 1px solid grey; background: transparent;} '
        scrollbarStyle = scrollbarStyle + 'QScrollBar::sub-page {border: 1px solid grey; background: transparent;} '

        self.verticalScrollBar().setStyleSheet(scrollbarStyle)
        self.setFocusPolicy(QtCore.Qt.NoFocus)

        self.setMovement(QtWidgets.QListView.Static)

        #-----------------------------------------------------------------------
        # item container from query / have button

        self.itemSelectionChanged.connect(self.emitClicked)
        self.itemDoubleClicked.connect(self.doubleClick)
        self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.menuPopup)

        self.mousePressPos = QtCore.QPoint() # TO STORE DRAG START POINT

    def menuPopup(self, point):
        menu = QtWidgets.QMenu(self)
        #items = self.selectedItems()
        if self.itemAt(point):
            item = self.itemAt(point)
            openAction = menu.addAction(u'Open Folder')
            openAction.triggered.connect(lambda : self.openItem(item))

            detailAction = menu.addAction(u'Detail...')
            detailAction.triggered.connect(lambda : self.showDetail(item))

            if item.getItemData()['type'] == 'ASSET_SRC':
                if not "PyQt" in Qt.__binding__:
                    importAction = menu.addAction(u'Import data in Maya')
                    importAction.triggered.connect(lambda : self.importToMaya(item))

                exportAction = menu.addAction(u'Export Data in directory')
                exportAction.triggered.connect(lambda: self.exportToDirectory(item))

                invenConf = db_query.getInventoryConfig()

                if getpass.getuser() in invenConf['authorized_user'] and not self.viewerId == "bookmark":
                    renameAction = menu.addAction(u'Rename item')
                    renameAction.triggered.connect(lambda: self.renameItem(item))

                    deleteAction = menu.addAction(u'Delete item')
                    deleteAction.triggered.connect(lambda: self.deleteItem(item))

            elif item.getItemData()['type'] == 'BORA_SRC' and not "PyQt" in Qt.__binding__:
                BOPMeshAction = menu.addAction(u'Import Ocean')
                BOPMeshAction.triggered.connect(lambda: self.importBOPmesh(item))
                BOPsimulAction = menu.addAction(u'Import Params')
                BOPsimulAction.triggered.connect(lambda: self.importBOPparam(item))

            elif item.getItemData()['type'] == "HDRI_SRC" and isHoudini:
                exportAction = menu.addAction(u'Import hdri')
                exportAction.triggered.connect(lambda: self.importHdriToHoudini(item))

            elif item.getItemData()['type'] == "VFX_REF":
                downloadAction = menu.addAction(u'Download Selected Item')
                downloadAction.triggered.connect(lambda: self.downloadItems(self.selectedItems()))

            # TEST FOR DEBUG
            if getpass.getuser() in ['taehyung.lee', 'root', 'render']:
                recreateAction = menu.addAction(u'Recreate')
                recreateAction.triggered.connect(lambda: self.recreate(self.selectedItems()))

            menu.popup(self.mapToGlobal(point))

    def downloadItems(self, items):
        docs = []
        for i in items:
            docs.append(i.getItemData())
        dialog = DownloadDialog(self, docs)
        dialog.exec_()

    def recreate(self, items):
        i, ok = QtWidgets.QInputDialog.getInteger(self,
                "QInputDialog.getInteger()", "Percentage:", -1, -1, 100, 1)

        for item in items:
            slider = self.itemWidget(item).horizontalSlider

            info = item.getItemData()
            mov = info['files']['mov']
            thumb = info['files']['thumbnail']
            size = "180x125"
            if i == -1:
                point = float(slider.value()) / 24.0
            else:
                point = float(i) / 24.0
            ffmpegCmd = '/netapp/backstage/pub/apps/ffmpeg_for_exr/bin/ffmpeg_with_env'


            thumbnailCmd =  [ffmpegCmd, '-i', info['files']['mov'], '-vframes', '1']
            thumbnailCmd += ['-vf', 'scale=320x240', '-ss', str(point), '-y']
            thumbnailCmd += [info['files']['thumbnail']]
            print 'cmd : ', ' '.join(thumbnailCmd)

            subprocess.call(thumbnailCmd)

    def importBOPmesh(self, item):
        import BoraOceanProcess
        BoraOceanProcess.importMayaPreset(item).importScene()

    def importBOPparam(self, item):
        import BoraOceanProcess
        BoraOceanProcess.importMayaPreset(item).importSimul()

    def importToMaya(self, item):
        if item.getItemData()['type'] == 'ASSET_SRC':
            AssetDataProcess.importToMaya(item.getItemData()['files'])

    def importHdriToHoudini(self, item):
        hdriPath = item.getItemData()['files']['org']
        filename = os.path.basename(hdriPath)
        node = hou.node('/obj')
        lightNode = node.createNode("envlight", filename)
        lightNode.parm('env_map').set(hdriPath)

    def exportToDirectory(self, item):
        if item.getItemData()['type'] == 'ASSET_SRC':
            AssetDataProcess.exportAssetData(self, item.getItemData())

    def renameItem(self, item):
        if item.getItemData()['type'] == 'ASSET_SRC':
            itemName = item.getItemData()['name']
            dialog = QtWidgets.QInputDialog(self)
            dialog.setWindowTitle('Rename')
            dialog.setLabelText("input rename [current : %s]" % itemName)
            msg = dialog.exec_()
            if msg == 1:
                db_query.updateName(item.getItemData()['_id'], dialog.textValue())
                item.setText(' ' + dialog.textValue())

    def deleteItem(self, item):
        if item.getItemData()['type'] == 'ASSET_SRC':
            print item.getItemData()['_id']
            db_query.deleteItem(item.getItemData()['_id'])
            row = item.listWidget().row(item)
            item.listWidget().takeItem(row)

    def openItem(self, item):
        previewPath = item.getItemData()['files']['preview']
        #dirPath = os.path.dirname(previewPath)

        if sys.platform == 'darwin':
            subprocess.call(["open", "-R", unicode(previewPath)])
        else:
            subprocess.Popen(['nautilus', unicode(previewPath)])

    def editTags(self, item):
        # # CASE ITEM COUNT IS ONE
        # tag = item.getItemData()['tags']
        # dialog = TagEdit(tags=tag)
        # print dialog.exec_()
        # ## CASE MULTIPLE ITEM
        pass

    def showDetail(self, item):
        dw = DetailDialog(self, item)
        dw.exec_()

    def itemClickForPlay(self, item):
        try:
            # WAIT ICON UNTIL GIF LOADING
            itemId = item.getItemData()['_id']
            updateThread = taskThread.StatUpdateThread()
            updateThread.queryTerm = {'itemId' : itemId}
            updateThread.action = 'click'
            updateThread.updateTerm = {'$inc' : {updateThread.action:1}}
            updateThread.start()
            QtWidgets.QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.WaitCursor))
            updateThread.wait()
            item.click(self.itemWidget(item))
            QtWidgets.QApplication.restoreOverrideCursor()
        except e:
            print e
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    def doubleClick(self, item):
        try:
            itemId = item.getItemData()['_id']
            updateThread = taskThread.StatUpdateThread()
            updateThread.queryTerm = {'itemId': itemId}
            updateThread.action = 'doubleclick'
            updateThread.updateTerm = {'$inc': {updateThread.action: 1}}
            updateThread.start()
            updateThread.wait()
        except e:
            print e
            pass
        item.doubleClick(self.itemWidget(item))

    def emitClicked(self):
        #self.emit(QtCore.SIGNAL("assetClicked"), self.selectedItems())
        #self.emit(self.assetClicked, self.selectedItems())
        self.assetClicked.emit(self.selectedItems())

    def makeUrlsFromDict(self, fdata):
        urls = []
        for key in fdata:
            if fdata[key]:
                files = fdata[key]
                if type(files) == list:
                    for ff in files:
                        url = QtCore.QUrl()
                        #url.setPath('file://'+key + INVEN_HEADER + ff)
                        url.setPath('file://' + ff + INVEN_TAIL+ key)
                        urls.append(url)
                elif type(files) == unicode:
                    url = QtCore.QUrl()
                    #url.setPath('file://'+key + INVEN_HEADER + files)
                    url.setPath('file://' + files + INVEN_TAIL + key)
                    urls.append(url)
        return urls

    def mousePressEvent(self, QMouseEvent):
        super(BaseList, self).mousePressEvent(QMouseEvent)
        self.mousePressPos = QMouseEvent.pos()

    def mouseMoveEvent(self, QMouseEvent):
        if not QMouseEvent.buttons() & QtCore.Qt.LeftButton:
            return
        if (QMouseEvent.pos() - self.mousePressPos).manhattanLength() \
                < QtWidgets.QApplication.startDragDistance():
            return

        if self.selectedItems() and self.itemAt(self.mousePressPos):

            mimeData = QtCore.QMimeData()
            orgPath = ''
            urls = []
            for i in self.selectedItems():
                if i.getItemData()['type'] == 'ASSET_SRC':
                    urls = self.makeUrlsFromDict(i.getItemData()['files'])

                elif i.getItemData()['files'].has_key('nuke'):
                    orgPath += i.getItemData()['files']['nuke'] + '\n'
                    urls.append(QtCore.QUrl('file://'+i.getItemData()['files']['nuke']))
                elif i.getItemData()['files'].has_key('tx'):
                    orgPath += i.getItemData()['files']['tx'] + '\n'
                    urls.append(QtCore.QUrl('file://'+i.getItemData()['files']['tx']))
                elif i.getItemData()['files'].has_key('org'):
                    orgPath += i.getItemData()['files']['org'] + '\n'
                    urls.append(QtCore.QUrl('file://'+i.getItemData()['files']['org']))
                elif i.getItemData()['files'].has_key('model'):
                    orgPath += i.getItemData()['files']['model'] + '\n'
                    urls.append(QtCore.QUrl('file://'+i.getItemData()['files']['model']))
                else:
                    orgPath += i.getItemData()['files']['preview'] + '\n'
                    urls.append(QtCore.QUrl('file://'+i.getItemData()['files']['preview']))

            mimeData.setText(orgPath)
            mimeData.setUrls(urls)

            drag = QtGui.QDrag(self)
            drag.setMimeData(mimeData)
            drag.exec_()

            #super(BaseList, self).mouseReleaseEvent(QMouseEvent)
            #super(BaseList, self).mouseMoveEvent(QMouseEvent)
        else:
            super(BaseList, self).mouseMoveEvent(QMouseEvent)

    def mouseReleaseEvent(self, QMouseEvent):
        if QMouseEvent.button() & QtCore.Qt.LeftButton:
            if self.itemAt(QMouseEvent.pos()) and ((QMouseEvent.pos() - self.mousePressPos).manhattanLength() \
                     < QtWidgets.QApplication.startDragDistance()):

                item = self.itemAt(QMouseEvent.pos())
                self.itemClickForPlay(item)
        super(BaseList, self).mouseReleaseEvent(QMouseEvent)

    # def dragLeaveEvent(self, QMouseEvent):
    #     print "drag leave event!!!"
    #     super(BaseList, self).mouseReleaseEvent(QMouseEvent)
    #     super(BaseList, self).dragLeaveEvent(QMouseEvent)
    #
    # def dragEnterEvent(self, event):
    #     event.accept()
    #
    # def dragMoveEvent(self, event):
    #     event.accept()


class BookmarkList(QtWidgets.QListWidget):
    def __init__(self, parent=None):
        super(BookmarkList, self).__init__(parent)
        #BookmarkList.__init__(self)
        # TO CLASSIFY DRAG ENTER FROM
        self.viewerId = 'bookmark'
        self.objIds = []

        self.deleteKey = QtWidgets.QShortcut(self)
        #self.deleteKey.setKey(QtCore.Qt.Key_Delete)
        self.deleteKey.setKey(QtGui.QKeySequence.Delete)
        self.deleteKey.activated.connect(self.deleteItem)

        self.setAcceptDrops(True)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setDragEnabled(True)
        self.setViewMode(QtWidgets.QListView.IconMode)

        # IMAGE SIZE OF EACH ITEM / MUST SMALLER THAN ACTUAL ITEM SIZE
        self.setIconSize(QtCore.QSize(200, 130))
        self.setSelectionMode(QtWidgets.QListView.ExtendedSelection)
        self.setResizeMode(QtWidgets.QListView.Adjust)
        # self.viewOptions().displayAlignment = QtCore.Qt.AlignCenter
        # self.setUniformItemSizes(True)
        self.setStyleSheet("""
                QListView{border: none; background: rgb(48,48,48); color: rgb(200,200,200);}
                QListView::item:selected {background: rgb(68,150,210)}
                QListView::item:!selected {background: rgb(48,48,48)}
                QListView::item:hover{background-color: rgb(240,169,32);}
                """)

        scrollbarStyle = 'QScrollBar{border: 1px solid black; background: #484848;} '
        scrollbarStyle = scrollbarStyle + 'QScrollBar::add-page  {border: 1px solid grey; background: transparent;} '
        scrollbarStyle = scrollbarStyle + 'QScrollBar::sub-page {border: 1px solid grey; background: transparent;} '

        self.verticalScrollBar().setStyleSheet(scrollbarStyle)
        self.setFocusPolicy(QtCore.Qt.NoFocus)

        self.itemDoubleClicked.connect(self.doubleClick)
        self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.menuPopup)

        self.mousePressPos = QtCore.QPoint() # TO STORE DRAG START POINT


    def menuPopup(self, point):
        menu = QtWidgets.QMenu(self)
        #items = self.selectedItems()
        if self.itemAt(point):
            item = self.itemAt(point)
            openAction = menu.addAction(u'Open Folder')
            openAction.triggered.connect(lambda : self.openItem(item))

            detailAction = menu.addAction(u'Detail...')
            detailAction.triggered.connect(lambda : self.showDetail(item))

            menu.popup(self.mapToGlobal(point))

    def openItem(self, item):
        previewPath = item.getItemData()['files']['preview']
        if sys.platform == 'darwin':
            subprocess.call(["open", "-R", unicode(previewPath)])
        else:
            subprocess.Popen(['nautilus', unicode(previewPath)])

    def showDetail(self, item):
        dw = DetailDialog(self, item)
        dw.exec_()

    def itemClickForPlay(self, item):
        try:
            # WAIT ICON UNTIL GIF LOADING
            itemId = item.getItemData()['_id']
            updateThread = taskThread.StatUpdateThread()
            updateThread.queryTerm = {'itemId' : itemId}
            updateThread.action = 'click'
            updateThread.updateTerm = {'$inc' : {updateThread.action:1}}
            updateThread.start()
            updateThread.wait()

            QtWidgets.QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.WaitCursor))
            item.click(self.itemWidget(item))
            QtWidgets.QApplication.restoreOverrideCursor()
        except e:
            print e
            pass
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    def doubleClick(self, item):
        try:
            itemId = item.getItemData()['_id']
            updateThread = taskThread.StatUpdateThread()
            updateThread.queryTerm = {'itemId': itemId}
            updateThread.action = 'doubleclick'
            updateThread.updateTerm = {'$inc': {updateThread.action: 1}}
            updateThread.start()
            updateThread.wait()
        except e:
            print e
            pass
        item.doubleClick(self.itemWidget(item))

    def emitClicked(self):
        self.assetClicked.emit(self.selectedItems())

    def makeUrlsFromDict(self, fdata):
        urls = []
        for key in fdata:
            if fdata[key]:
                files = fdata[key]
                if type(files) == list:
                    for ff in files:
                        url = QtCore.QUrl()
                        #url.setPath('file://'+key + INVEN_HEADER + ff)
                        url.setPath('file://' + ff + INVEN_TAIL+ key)
                        urls.append(url)
                elif type(files) == unicode:
                    url = QtCore.QUrl()
                    #url.setPath('file://'+key + INVEN_HEADER + files)
                    url.setPath('file://' + files + INVEN_TAIL + key)
                    urls.append(url)
        return urls

    def mousePressEvent(self, QMouseEvent):
        super(BookmarkList, self).mousePressEvent(QMouseEvent)
        self.mousePressPos = QMouseEvent.pos()

    def mouseMoveEvent(self, QMouseEvent):
        if not QMouseEvent.buttons() & QtCore.Qt.LeftButton:
            return
        if (QMouseEvent.pos() - self.mousePressPos).manhattanLength() \
                < QtWidgets.QApplication.startDragDistance():
            return

        if self.selectedItems() and self.itemAt(self.mousePressPos):

            mimeData = QtCore.QMimeData()
            orgPath = ''
            urls = []
            for i in self.selectedItems():
                if i.getItemData()['type'] == 'ASSET_SRC':
                    urls = self.makeUrlsFromDict(i.getItemData()['files'])

                elif i.getItemData()['files'].has_key('nuke'):
                    orgPath += i.getItemData()['files']['nuke'] + '\n'
                    urls.append(QtCore.QUrl('file://'+i.getItemData()['files']['nuke']))
                elif i.getItemData()['files'].has_key('tx'):
                    orgPath += i.getItemData()['files']['tx'] + '\n'
                    urls.append(QtCore.QUrl('file://'+i.getItemData()['files']['tx']))
                elif i.getItemData()['files'].has_key('org'):
                    orgPath += i.getItemData()['files']['org'] + '\n'
                    urls.append(QtCore.QUrl('file://'+i.getItemData()['files']['org']))
                elif i.getItemData()['files'].has_key('model'):
                    orgPath += i.getItemData()['files']['model'] + '\n'
                    urls.append(QtCore.QUrl('file://'+i.getItemData()['files']['model']))
                else:
                    orgPath += i.getItemData()['files']['preview'] + '\n'
                    urls.append(QtCore.QUrl('file://'+i.getItemData()['files']['preview']))

            mimeData.setText(orgPath)
            mimeData.setUrls(urls)

            drag = QtGui.QDrag(self)
            drag.setMimeData(mimeData)
            drag.exec_()
        else:
            super(BookmarkList, self).mouseMoveEvent(QMouseEvent)

    def mouseReleaseEvent(self, QMouseEvent):
        if QMouseEvent.button() & QtCore.Qt.LeftButton:
            if self.itemAt(QMouseEvent.pos()) and ((QMouseEvent.pos() - self.mousePressPos).manhattanLength()
                    < QtWidgets.QApplication.startDragDistance()):

                item = self.itemAt(QMouseEvent.pos())
                self.itemClickForPlay(item)
        super(BookmarkList, self).mouseReleaseEvent(QMouseEvent)

    def deleteItem(self):
        print "delete key pressed!!"
        oids = [self.item(i).getItemData()['_id'] for i in range(self.count())]
        print "sefl.objIds : ", self.objIds
        print oids

        for i in self.selectedItems():
            self.objIds.remove(i.getItemData()['_id'])
            self.takeItem(self.row(i))
            del i

        db_query.updateBookmark(getpass.getuser(), self.objIds)

    def dragEnterEvent(self, event):
        if event.mimeData().hasText():
            event.accept()
            return
        orgPath = ''
        for i in self.selectedItems():
            if i.getItemData()['files'].has_key('nuke'):
                orgPath += i.getItemData()['files']['nuke'] + '\n'
            elif i.getItemData()['files'].has_key('tx'):
                orgPath += i.getItemData()['files']['tx'] + '\n'
            elif i.getItemData()['files'].has_key('org'):
                orgPath += i.getItemData()['files']['org'] + '\n'
            elif i.getItemData()['files'].has_key('model'):
                orgPath += i.getItemData()['files']['model'] + '\n'
        event.mimeData().setText(orgPath)
        event.accept()

    def dragMoveEvent(self, event):
        event.accept()

    def dragLeaveEvent(self, event):
        event.accept()

    def dropEvent(self, event):
        if event.source().viewerId == "viewer":
            for i in event.source().selectedItems():
                # DUPLICATION CHECK
                if i.getItemData()['_id'] in self.objIds:
                    continue
                makeItems(self, i.getItemData())
                self.objIds.append(i.getItemData()['_id'])
            # record DB
            oids = [self.item(i).getItemData()['_id'] for i in range(self.count())]
            db_query.updateBookmark(getpass.getuser(), oids)

            event.source().clearSelection()
            event.accept()
        event.accept()



class CategoryTree(QtWidgets.QTreeWidget):
    def __init__(self, parent=None):
        super(CategoryTree, self).__init__(parent)
        self.setFocusPolicy(QtCore.Qt.NoFocus)
        self.setContextMenuPolicy(QtCore.Qt.DefaultContextMenu)
        self.header().setVisible(False)

        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        print "dragEnterEvent!!!"
        event.accept()

    def dragMoveEvent(self, event):
        event.accept()

    def dropEvent(self, event):
        print "dropped!!"
        catItem = self.itemAt(event.pos())
        if catItem:
            # CHECK IF ITEM IS UNDER VFX_REF
            if catItem.parent() and (catItem.parent().text(0) == 'VFX_REF'):

                catName = unicode(catItem.text(0))
                items = event.source().selectedItems()
                #try:
                for i in items:
                    itemId = i.getItemData()['_id']
                    cPrj = i.getItemData()['project']
                    nPrj = None
                    if cPrj == u'etc':
                        nPrj = [catName]
                        print "etc", nPrj

                    elif type(cPrj) == unicode:
                        print repr(cPrj)
                        nPrj = [cPrj, catName]
                        print "unicode", nPrj

                    else:
                        nPrj = cPrj + [catName]
                        print "else", nPrj
                    #print repr(cPrj), repr(nPrj)
                    db_query.updateProject(itemId, nPrj)
                # except:
                #     QtGui.QMessageBox.information(self, "Error", "Error")
                # finally:
                #     QtGui.QMessageBox.information(self, "Done", "Done")

        event.accept()
