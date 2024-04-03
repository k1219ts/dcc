# -*- coding: utf-8 -*-
import getpass
import os

from pymodule.Qt import QtCore
from pymodule.Qt import QtWidgets

from controllers.itemController import ItemController
from widgets.itemform import ItemForm
from libs.utils import error_message
from libs.utils import open_directory
from libs.ustr import ustr

try:
    import dxsUsd
    import maya.cmds as cmds
    MAYA = True
except:
    MAYA = False

class ItemView(QtWidgets.QListView):

    changed = QtCore.Signal()
    item_changed = QtCore.Signal(object)
    item_clicked = QtCore.Signal(object)

    def __init__(self, parent=None):
        QtWidgets.QListView.__init__(self, parent)

        # instance variables
        self.advanced_mode = True
        self.item = []
        self.checkCategory = ''
        self.assetName = ''
        self.filePath = ''

        self.item_controller = ItemController(self)

        self.setAcceptDrops(True)
        self.setDragDropMode(QtWidgets.QAbstractItemView.DragDrop)
        self.setDragEnabled(True)
        self.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.setIconSize(QtCore.QSize(160, 150))
        self.setResizeMode(QtWidgets.QListView.Adjust)
        self.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.setViewMode(QtWidgets.QListView.IconMode)

        # QAbstractItemView
        self.clicked.connect(self.listview_clicked)
        self.doubleClicked.connect(self.listview_double_clicked)

        # Connect the contextMenu
        self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.item_rmbClicked)

        self.menu = QtWidgets.QMenu(self)
        self.menu.setStyleSheet("\
            QMenu::item {color: #CCCCCC;}\
            QMenu::item:selected {background-color: #81CF3E; color: #404040;}")

        self.tag_input_action = QtWidgets.QAction("Input Tag", self)
        self.tag_input_action.triggered.connect(self.tag_input)
        self.delete_action = QtWidgets.QAction("Delete", self)
        self.delete_action.triggered.connect(self.deleteItems)
        self.edit_action = QtWidgets.QAction("Edit", self)
        self.edit_action.triggered.connect(self.show_edit_window)
        # MAYA
        self.usd_viewer_action = QtWidgets.QAction("USD Viewer", self)
        self.usd_viewer_action.triggered.connect(self.usd_window)
        self.import_usd_action = QtWidgets.QAction("import USD", self)
        self.import_usd_action.triggered.connect(self.referenceImport)
        self.import_geom_action = QtWidgets.QAction("import Geom", self)
        self.import_geom_action.triggered.connect(self.importGeom_clicked)
        self.replace_usds_action = QtWidgets.QAction("Replace USDs", self)
        self.replace_usds_action.triggered.connect(self.replace_clicked)

    # OK
    def listview_clicked(self, model_index):
        document = self.find_document((model_index))
        self.item_clicked.emit(document)

    def is_feature_enabled(self):
        return self.advanced_mode

    def set_advanced_mode(self, mode):
        self.advanced_mode = mode

    def clear(self):
        self.item_controller.clear()

    def add_items(self, items, return_count=False):
        count = 0
        for item in items:
            self.item_controller.add_item(item)
            count += 1

        if return_count:
            return count

        return None

    def document(self, list_widget_item):
        return self.item_controller.document(list_widget_item)

    def get_selectedItem(self):
        itemList = []

        model_indexes = self.selectedIndexes()
        for model_index in model_indexes:
            document = self.find_document(model_index)

            self.checkCategory = document.category
            self.assetName = document.name
            self.filePath = document.dir_path

            itemList.append(document)

        self.item = itemList

    def dragMoveEvent(self, event):
        event.accept()

    # TODO: dragEnterEvent - Texture
    def dragEnterEvent(self, event):
        event.ignore()
        # urlList = []
        # if event.mimeData().hasUrls():
        #     for index, i in enumerate(self.selectedItems()):
        #         source_name = i.text()
        #         self.get_selectedItem()
        #         if self.checkCategory == 'Texture':
        #             item = Database.gDB.source.find_one({"name": source_name})
        #             path = item['files']['filePath']
        #             itemList = []
        #             if os.path.exists(path):
        #                 if 'source' in path:  # single texture file
        #                     for i in os.listdir(path):
        #                         if source_name in i:
        #                             if not 'preview' in i:
        #                                 itemList.append(i)
        #                 else: # PBR folder
        #                     getFiles = os.listdir(path)
        #                     for i in getFiles:
        #                         if 'jpg' in i or 'tif' in i or 'exr' in i:
        #                             if not 'preview' in i:
        #                                 itemList.append(i)
        #                         else:
        #                             pass
        #                 for fileName in itemList:
        #                     urlLink = ''
        #                     urlLink += 'file://'
        #                     urlLink += path
        #                     urlLink += '/'
        #                     urlLink += fileName
        #                     url = QtCore.QUrl(urlLink)
        #                     urlList.append(url)
        #                 event.mimeData().setUrls(urlList)
        #                 event.accept()

        #         else: #USD Asset
        #             pass
        # else:
        #     event.ignore()

    def dropEvent(self, event):
        event.ignore()

    def item_rmbClicked(self, point):
        self.menu.clear()

        is_bookmark = True if type(self).__name__ == "BookmarkView" else False

        index = self.indexAt(point)
        if not index.isValid():
            return

        select_count =  len(self.selectedIndexes())

        self.get_selectedItem()

        if not self.is_feature_enabled():
            if select_count != 1:
                return

            self.menu.addAction(self.usd_viewer_action)
            if MAYA:
                self.menu.addSeparator()
                self.menu.addAction(self.import_usd_action)
                self.menu.addAction(self.import_geom_action)
                self.menu.addAction(self.replace_usds_action)
            self.menu.exec_(self.mapToGlobal(point))
            return

        if self.checkCategory == "Texture":
            if self.item[0].sub_category == "Trash":
                self.menu.addAction(self.delete_action)
            else:
                if is_bookmark:
                    pass
                else:
                    if select_count == 1:
                        # self.menu.addAction(self.tag_input_action)
                        self.menu.addAction(self.edit_action)

        else:
            if self.item[0].sub_category == "Trash":
                self.menu.addAction(self.delete_action)
            else:
                if select_count == 1:
                    self.menu.addAction(self.usd_viewer_action)
                    if MAYA:
                        self.menu.addSeparator()
                        self.menu.addAction(self.import_usd_action)
                        self.menu.addAction(self.import_geom_action)
                        self.menu.addAction(self.replace_usds_action)
                        self.menu.addSeparator()

                    if is_bookmark:
                        pass
                    else:
                        # self.menu.addAction(self.tag_input_action)
                        self.menu.addAction(self.edit_action)

        self.menu.exec_(self.mapToGlobal(point))

    def tag_input(self):
        self.get_selectedItem()

        selection_ranges = self.selectionModel().selection()
        getTagName = ",".join(self.item[0].tag)
        Qinput = QtWidgets.QInputDialog()
        lineEdit = QtWidgets.QLineEdit()
        lineEdit.setStyleSheet('''QLineEdit{
                                            color: rgb(255,255,255);
                                            background-color: white;
                                            }
                                           ''')
        text, ok = Qinput.getText(self, ' Tag Input Dialog {} item/s'.format(len(selection_ranges)), 'Tag',
                                  lineEdit.Normal, text=getTagName)

        if ok:
            tag_name = ustr(text).lower()
            self.item_controller.deprecated_edit_tags(selection_ranges, tag_name)

            self.changed.emit()

    def deleteItems(self):
        msg = QtWidgets.QMessageBox.question(
            self, "Delete", u"Are you sure want to delete items?",
            QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel)

        if msg != QtWidgets.QMessageBox.Ok:
            return

        self.get_selectedItem()

        selection_ranges = self.selectionModel().selection()
        self.item_controller.delete_items(selection_ranges)

        self.changed.emit()

    def listview_double_clicked(self, model_index):
        self.statusText("Open Directory")
        self.get_selectedItem()

        # if self.checkCategory == 'Texture':
        #     path = self.filePath
        # else:
        #     path = os.path.dirname(self.filePath)

        document = self.find_document(model_index)

        try:
            open_directory(document.dir_path)
        except OSError as error:
            error_message(str(error))

    def usd_window(self):
        self.statusText("Open USD Viewer")
        self.get_selectedItem()
        for i in os.environ:
            if 'DEV_LOCATION' in i:
                cmd = '/backstage/dcc/DCC rez-env usdtoolkit -- usdviewer %s' % self.filePath
                print('Developer Mode')
            else:
                cmd = '/backstage/dcc/DCC dev rez-env usdtoolkit -- usdviewer %s' % self.filePath
        os.system(cmd)

    def show_edit_window(self):
        model_index = self.currentIndex()
        document = self.find_document(model_index)
        dialog = ItemForm(self, document)
        dialog.changed.connect(self.edit_item_changed)
        dialog.tag_changed.connect(self.edit_item_tag_changed)
        dialog.update_form()
        dialog.show()

    def edit_item_changed(self):
        """단일 아이템이 변경되면 item_changed 시그널 발생하는 메소드"""
        model_index = self.currentIndex()
        document = self.find_document(model_index)
        item = self.find_item(model_index)

        # 썸네일과 어셋명, 상태 변경
        item.setText(document.name)
        item.setData(None, QtCore.Qt.DecorationRole)
        item.setData(document.preview_path)
        item.status = document.status

        self.item_changed.emit(document)

    def edit_item_tag_changed(self):
        """전체 아이템이 갱신되도록 changed 시그널 발생하는 메소드"""
        self.changed.emit()

    def statusText(self, text=''):
        self.statusLabel.clear()
        self.statusLabel.setText(text)

    def itemConnect(self, status):
        self.statusLabel = status

    def find_item(self, model_index):
        return self.item_controller.find_item(model_index)

    def find_document(self, model_index):
        return self.item_controller.find_document(model_index)

    #
    # MAYA
    #
    def referenceImport(self):
        self.statusText("Import USD")
        self.get_selectedItem()
        assetName = self.assetName
        filename = self.filePath
        if cmds.pluginInfo('TaneForMaya', q=True, l=True):
            selected = cmds.ls(sl=True, dag=True, type='TN_Tane')
            if selected:
                self.referenceTane(selected[0], assetName, filename)
                self.statusText("Imported Successfully")
            else:
                dxsUsd.dxsMayaUtils.UsdAssemblyImport(filename)
                self.statusText("Imported Successfully")
        else:
            dxsUsd.dxsMayaUtils.UsdAssemblyImport(filename)
            self.statusText("Imported Successfully")

    def referenceTane(self, taneShape, assetName, sourceFile):
        environmentNode = cmds.listConnections(taneShape, s=True, d=False, type='TN_Environment')
        assert environmentNode, '# msg : tane setup error'
        environmentNode = environmentNode[0]
        index = 0
        for idx in range(0, 100):
            if not cmds.connectionInfo("%s.inSource[%d]" % (environmentNode, idx), ied=True):
                index = idx
                break
        proxyShape = cmds.TN_CreateNode(nt='TN_UsdProxy')
        proxyTrans = cmds.listRelatives(proxyShape, p=True)[0]
        cmds.setAttr('%s.visibility' % proxyTrans, 0)
        cmds.setAttr('%s.renderFile' % proxyShape, sourceFile, type='string')
        cmds.connectAttr('%s.outSource' % proxyShape, '%s.inSource[%d]' % (environmentNode, index))
        proxyTrans = cmds.rename(proxyTrans, 'TN_%s' % assetName)
        taneTrans = cmds.listRelatives(taneShape, p=True)[0]
        cmds.parent(proxyTrans, taneTrans)
        cmds.select(taneTrans)

    def importGeom_clicked(self):
        self.statusText("Import Geometry")
        self.get_selectedItem()
        filePath = self.filePath
        model_dir = os.path.join(os.path.dirname(filePath), 'model')
        print('model_dir: {}'.format(model_dir))
        current_versions = []

        for f in os.listdir(model_dir):
            if os.path.isdir(os.path.join(model_dir, f)):
                if "v" in f:
                    current_versions.append(f)
        last_version = current_versions[-1]
        last_dir_path = os.path.join(model_dir, last_version)
        print('last_version: {}'.format(last_version))
        print('last_dir_path: {}'.format(last_dir_path))

        for fileName in os.listdir(last_dir_path):
            if fileName in ["high_geom.usd", "mid_geom.usd", "low_geom.usd"]:
                dxsUsd.dxsMayaUtils.UsdImport(os.path.join(last_dir_path, fileName))
                self.statusText("Imported Successfully")
            else:
                pass

    def replace_clicked(self):
        self.statusText("Replace :Please Select maya nodes")
        selectedObj = cmds.ls(sl=True)
        self.get_selectedItem()
        filePath = self.filePath
        if not selectedObj:
            msgBox = QtWidgets.QMessageBox()
            msgBox.setWindowTitle("Error")
            msgBox.setText("Please Select maya nodes.")
            return msgBox.exec_()
        replaced = []
        for i in selectedObj:
            cmds.setAttr("%s.filePath" % i, filePath, type="string")
            replaced.append(i)
        replaced = sorted(replaced)
        num = 0
        for i in replaced:
            num += 1
            newName = self.assetName+str(num)
            cmds.rename(i, newName)
        self.statusText("Replaced Successfully")
