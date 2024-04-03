#coding=utf-8
from PySide2 import QtWidgets, QtGui, QtCore
import subprocess
from core import Database
import os
import getpass
import datetime



try:
    import maya.cmds as cmds
    MAYA = True
    plugins = ['backstageMenu', 'pxrUsd','DXUSD_Maya']
    for p in plugins:
        if not cmds.pluginInfo(p, q=True, l=True):
            cmds.loadPlugin(p)
    import dxBlockUtils

except:
    MAYA = False
    pass

userName = getpass.getuser()

class ItemListWidget(QtWidgets.QListWidget):
    def __init__(self, parent=None):
        QtWidgets.QListWidget.__init__(self, parent)

        self.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.setViewMode(QtWidgets.QListView.IconMode)
        self.setIconSize(QtCore.QSize(160, 150))
        self.setResizeMode(QtWidgets.QListView.Adjust)
        self.setAcceptDrops(True)
        self.setDragEnabled(True)
        self.setDragDropMode(QtWidgets.QAbstractItemView.DragDrop)
        self.itemDoubleClicked.connect(self.openDir_clicked)
        self.clicked.connect(self.item_cliked)
        # Connect the contextMenu
        self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.item_rmbClicked)

    def get_selectedItem(self):
        itemList = []
        selectedItems = self.selectedItems()
        for item in selectedItems:
            itemName = item.text()
            SCcursor = Database.gDB.source.find({'name': {'$exists': True}})

            cursor = Database.gDB.item.find({'name': {'$exists': True}})


            cursorList = [SCcursor, cursor]
            for i in cursorList:
                for itr in i:
                    if itemName==itr['name']:
                        if not itr in itemList:
                            itemList.append(itr)


        self.item = itemList
        for i in itemList:
            item = i
            self.thumbnailPath = item['files']['preview']
            reply = item['reply'][0]
            self.user = reply['user']
            self.time = reply['time'].split('T')[0]
            self.checkCategory = item['category']

            if self.checkCategory == 'Texture':
                self.assetName = item['name']
                self.filePath = item['files']['filePath']
                t = os.path.getmtime(self.filePath)
                self.mtime= datetime.datetime.fromtimestamp(t)


            else:
                self.assetName = item['name']
                self.filePath = item['files']['usdfile']
                t = os.path.getmtime(self.filePath)
                self.mtime= str(datetime.datetime.fromtimestamp(t)).split('.')[0]



    def dragEnterEvent(self, event):
        current= self.itemAt(event.pos())
        urlList = []
        if event.mimeData().hasUrls:
            for index, i in enumerate(self.selectedItems()):
                source_name = i.text()
                self.get_selectedItem()
                if self.checkCategory == 'Texture':
                    item = Database.gDB.source.find_one({"name": source_name})
                    path = item['files']['filePath']
                    itemList = []
                    if os.path.exists(path):
                        if 'source' in path:  # single texture file
                            for i in os.listdir(path):
                                if source_name in i:
                                    if not 'preview' in i:
                                        itemList.append(i)
                        else: # PBR folder
                            getFiles = os.listdir(path)
                            for i in getFiles:
                                if 'jpg' in i or 'tif' in i or 'exr' in i:
                                    if not 'preview' in i:
                                        itemList.append(i)
                                else:
                                    pass
                        for fileName in itemList:
                            urlLink = ''
                            urlLink += 'file://'
                            urlLink += path
                            urlLink += '/'
                            urlLink += fileName
                            url = QtCore.QUrl(urlLink)
                            urlList.append(url)
                        event.mimeData().setUrls(urlList)
                        event.accept()

                else: #USD Asset
                    pass
        else:
            # event.accept()
            event.ignore()

    def dragMoveEvent(self, event):
        event.accept()

    def dropEvent(self, event):
        # print(event.mimeData().urls())
        event.ignore()

    def item_rmbClicked(self,point):

        if self.itemAt(point):
            index = self.indexAt(point)
            if not index.isValid():
                return

            menu = QtWidgets.QMenu()
            menu.setStyleSheet('''
                                            QMenu::item:selected {
                                            background-color: #81CF3E;
                                            color: #404040; }
                                           ''')

            self.get_selectedItem()

            if self.checkCategory == 'Texture':
                if self.item[0]['subCategory'] == 'Trash':
                    add_action6 = menu.addAction("Delete Item")
                    action = menu.exec_(self.mapToGlobal(point))
                    if action == add_action6:
                        self.deleteItem()
                        for item in self.selectedItems():
                            self.takeItem(self.row(item))
                    else:
                        pass

                else:
                    add_action5 = menu.addAction("Input Tag")
                    action = menu.exec_(self.mapToGlobal(point))
                    if action == add_action5:
                        self.tag_input()
                    else:
                        pass

            else:
                if self.item[0]['subCategory'] == 'Trash':
                    add_action6 = menu.addAction("Delete Item")
                    action = menu.exec_(self.mapToGlobal(point))
                    if action == add_action6:
                        self.deleteItem()
                        for item in self.selectedItems():
                            self.takeItem(self.row(item))
                    else:
                        pass

                else:
                    if MAYA == True:

                        add_action = menu.addAction("USD Viewer")
                        add_action2 = menu.addAction("import USD")
                        add_action3 = menu.addAction("import Geom")
                        add_action4 = menu.addAction("Replace USDs")
                        add_action5 = menu.addAction("Input Tag")
                        action = menu.exec_(self.mapToGlobal(point))
                        if action == add_action:
                            self.usd_window()
                        if action == add_action2:
                            self.referenceImport()
                        if action == add_action3:
                            self.importGeom_clicked()
                        if action == add_action4:
                            self.replace_clicked()
                        if action == add_action5:
                            self.tag_input()

                    elif MAYA == False:
                        add_action = menu.addAction("USD Viewer")
                        add_action5 = menu.addAction("Input Tag")
                        action = menu.exec_(self.mapToGlobal(point))
                        if action == add_action:
                            self.usd_window()
                        if action == add_action5:
                            self.tag_input()
                    else:
                        pass

        else:
            pass

    def tag_input(self):
        self.get_selectedItem()
        getTagName = ",".join(self.item[0]['tag'])
        Qinput =QtWidgets.QInputDialog()
        lineEdit =QtWidgets.QLineEdit()
        lineEdit.setStyleSheet('''QLineEdit{
                                            color: rgb(255,255,255);
                                            background-color: white;
                                            }
                                           ''')
        text, ok = Qinput.getText(self, ' Tag Input Dialog', 'Tag',
                                  lineEdit.Normal,text = getTagName)

        if ok:
            tagName = str(text).lower()
            for i in self.item:
                assetName =i['name']
                category = i['category']
                Database.AddTag(assetName, tagName, category)
        else:
            pass

    def deleteItem(self):
        self.get_selectedItem()
        for i in self.item:
            itemName= i['name']
            category= i['category']
            Database.AddDeleteItem(itemName, userName)
            Database.DeleteDocument(itemName, category)
            print('deleted')

    def openDir_clicked(self,item):
        self.statusText("Open Directory")
        self.get_selectedItem()

        if self.checkCategory == 'Texture':
            subprocess.Popen(['xdg-open', str(self.filePath)])

        else:
            path = os.path.dirname(self.filePath)
            subprocess.Popen(['xdg-open', str(path)])


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
                dxBlockUtils.ImportPxrReference(filename)
                self.statusText("Imported Successfully")
        else:
            print(filename)
            dxBlockUtils.ImportPxrReference(filename)
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
        print('model_dir:',model_dir)
        current_versions = []

        for f in os.listdir(model_dir):
            if os.path.isdir(os.path.join(model_dir, f)):
                if "v" in f:
                    current_versions.append(f)
        last_version = current_versions[-1]
        last_dir_path = os.path.join(model_dir, last_version)
        print('last_version:',last_version)
        print('last_dir_path:',last_dir_path)

        for fileName in os.listdir(last_dir_path):
            if "high_geom.usd" in fileName or "mid_geom.usd" in fileName or "low_geom.usd" in fileName:
                dxBlockUtils.UsdImport(os.path.join(last_dir_path, fileName)).doIt()
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
        replaced =[]
        for i in selectedObj:
            cmds.setAttr("%s.filePath" % i, filePath, type="string")
            replaced.append(i)
        replaced.sort()
        num = 0
        for i in replaced:
            num += 1
            newName = self.assetName+str(num)
            cmds.rename(i, newName)
        self.statusText("Replaced Successfully")

    def statusText(self,text=''):
        self.statusLabel.clear()
        self.statusLabel.setText(text)

    def itemConnect(self,status, preview, label, category):
        self.statusLabel = status
        self.previewLabel = preview
        self.commentLabel = label
        self.category = category

    def item_cliked(self, event):
        self.get_selectedItem()
        thumbnailPath = self.item[0]['files']['preview']
        reply = self.item[0]['reply'][0]
        user = reply['user']
        time = reply['time'].split('T')[0]


        pixmap = QtGui.QPixmap(thumbnailPath)
        self.previewLabel.setPixmap(pixmap)
        getTagName = ",".join(self.item[0]['tag'])

        if self.item[0]['category'] == 'Texture':
            if len(self.filePath) > 40:
                name = self.filePath.split("/") #[u'', u'assetlib', u'Texture', u'RealDisplacement', u'AUTUMN-LEAVES-01']
                self.filePath = "/"+name[1]+"/"+name[2]+"/"+name[3]+"/"+"\n"+"                             "+name[4]
                self.commentLabel.setText('File Summary\n\n%s \n     %s\n\n  Tag: %s \n\n\n\n' % (self.item[0]['name'], self.item[0]['files']['filePath'], getTagName))
            else:
                self.commentLabel.setText('File Summary\n\n%s \n     %s\n\n\n  Tag: %s \n\n\n\n' % (self.item[0]['name'], self.item[0]['files']['filePath'], getTagName))

        else:

            if len(self.assetName) > 10:
                name = self.filePath.split("/")
                self.filePath = "/"+name[1]+"/"+name[2]+"/"+name[3]+"/"+name[4]+"\n"+"           "+"/"+name[5]
                self.commentLabel.setText('File Summary\n\n%s \n     %s\n\n  Tag: %s \n\n  Uploaded by %s \n  %s \n\n' % (self.item[0]['name'], self.item[0]['files']['usdfile'], getTagName, user, self.mtime))
            else:
                self.commentLabel.setText('File Summary\n\n%s \n     %s\n\n\n  Tag: %s \n\n  Uploaded by %s \n  %s \n\n' % (self.item[0]['name'], self.item[0]['files']['usdfile'],getTagName ,user,self.mtime))
