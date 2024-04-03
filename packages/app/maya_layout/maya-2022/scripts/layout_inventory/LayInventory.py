# -*- coding: utf-8 -*-
####################################################
########## coding by RND youkyoung.kim #############
####################################################
import sys, os
import Qt
from Qt import QtWidgets
from Qt import QtGui
from Qt import QtCore
from inventory import inventory #inventory window
from LayInventory_ui import Ui_Form
from LayMongo import LayInventorydb
# from dxname import tag_parser

CURRENTPATH = os.path.dirname(os.path.abspath(__file__))
try:
    import maya.cmds as cmds
    import maya.mel as mel
except ImportError:
    pass

class InventoryMain(QtWidgets.QWidget):
    def __init__(self, parent = None):
        QtWidgets.QWidget.__init__(self, parent)
        self.setWindowFlags(QtCore.Qt.Window)
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.settingWindow()
        self.settingImage()
        self.itemInsert()
        self.connection()

    def connection(self):
        self.ui.close_btn.clicked.connect(lambda : self.close())
        self.ui.inventory_btn.clicked.connect(self.inventoryOpen)
        self.ui.type_combo.currentIndexChanged.connect(self.changeCategory)
        self.ui.source_btn.clicked.connect(self.openSource)
        self.ui.reset_btn.clicked.connect(self.allClear)
        self.ui.snapshot_btn.clicked.connect(self.snapShot)
        self.ui.send_btn.clicked.connect(self.dbSend)
        self.ui.mov_btn.clicked.connect(self.movInvenOpen)

    def settingWindow(self):
        ## reset value
        self.getlist = []
        self.sourcename = ""
        self.source = ""
        self.defaultimage = os.path.join(CURRENTPATH, 'img/preview.png')
        self.sourcepath = "/dexter/Cache_DATA/LAY/001_asset"
        self.ui.title_txt.clear()
        self.ui.tag_txt.clear()
        self.ui.source_txt.setText(self.sourcepath)
        self.ui.file_txt.clear()
        self.ui.snapshot_btn.setEnabled(0)
        self.ui.send_btn.setEnabled(0)
        self.loadpath = cmds.file(q=True, location=True)
        self.titleSet(self.loadpath)

    def titleSet(self, dirs = None):
        if dirs.find('.mb') > 0:
            self.ui.snapshot_btn.setEnabled(1)
            self.ui.send_btn.setEnabled(1)
            self.sourcepath = os.path.dirname(dirs)
            self.sourcename = os.path.basename(dirs)
            self.ui.source_txt.setText(self.sourcepath)
            self.ui.file_txt.setText(self.sourcename)
            if not self.loadpath == dirs:
                cmds.file(dirs, open=True, force=True)
                self.loadpath = dirs

    def settingImage(self):
        ## window image setting
        self.setButtonImage(self.ui.reset_btn, 'reset')
        self.setButtonImage(self.ui.source_btn, 'open')
        self.setButtonImage(self.ui.inventory_btn, 'inventory')
        self.setPixmapImage(self.defaultimage)

        self.ui.type_combo.setView(QtWidgets.QListView())
        self.ui.type_combo.setStyleSheet('''
        QComboBox QAbstractItemView::item { min-height: 20px; min-width: 120px;}
        QComboBox QAbstractItemView { font-size: 10pt;}''')

        self.ui.category_combo.setView(QtWidgets.QListView())
        self.ui.category_combo.setStyleSheet('''
        QComboBox QAbstractItemView::item { min-height: 20px; min-width: 120px;}
        QComboBox QAbstractItemView { font-size: 10pt;}''')

    def setButtonImage(self, button, image):
        button.setIcon(
            QtGui.QIcon(QtGui.QPixmap(os.path.join(CURRENTPATH, "img/%s.png" % image))))

    def setPixmapImage(self, img):
        ## snapshot view setting
        image = QtGui.QPixmap(img)
        image = image.scaled(320, 240, QtCore.Qt.KeepAspectRatioByExpanding, QtCore.Qt.SmoothTransformation)
        self.ui.snapshot_img.setPixmap(image)

    def inventoryOpen(self):
        # inventory window open
        window = inventory.Inventory()
        window.show()

    def movInvenOpen(self):
        # self.close()
        import LayInvenMov
        LayInvenMov.main()

    def itemInsert(self):
        ## type, category combo box setting
        self.category = {}
        assettype = ['camera', 'char', 'env', 'prop', 'vehicle', 'reference']
        self.category['camera'] = ['etc']
        self.category['char'] = ['animal', 'human', 'etc']
        self.category['env'] = ['grass', 'mountain', 'rock', 'tree', 'etc']
        self.category['prop'] = ['building', 'interior', 'machine', 'weapon', 'etc']
        self.category['vehicle'] = ['car', 'ship', 'train', 'airplain', 'etc']
        self.category['reference'] = ['clip', 'nuke', 'etc']
        self.ui.type_combo.addItems(assettype)
        self.changeCategory()

    def changeCategory(self):
        self.ui.category_combo.clear()
        imsi = self.ui.type_combo.currentText()
        self.ui.category_combo.addItems(self.category[imsi])

    def openSource(self):
        ## open file dialog loading
        dirs = self.openDialog("Open Scene File", self.sourcepath,
                                "Scene files (*.mb)")
        self.titleSet(dirs)

    def openDialog(self, title = None, path = None, filter = None):
        self.dir = ""
        loadpath = QtWidgets.QFileDialog.getOpenFileName(self, title, path, filter)
        return str(loadpath[0])

    def snapShot(self):
        ## type thumb file create
        img = self.loadpath.replace(".mb", "_thumb.jpg")
        self.newSnapshot(img)
        self.setPixmapImage(img)
        self.thumbnail = img

    def newSnapshot(self, path = None):
        ## mb snapshot new make
        currFrame = cmds.currentTime(query=True)
        format = cmds.getAttr("defaultRenderGlobals.imageFormat")
        cmds.setAttr("defaultRenderGlobals.imageFormat", 8)
        cmds.playblast(frame=currFrame, format="image", completeFilename=path, showOrnaments=True,
                       viewer=False, widthHeight=[1280, 720], percent=80)
        cmds.setAttr("defaultRenderGlobals.imageFormat", format)

    def allClear(self):
        ## reset btn click all clear
        self.setPixmapImage(self.defaultimage)
        self.settingWindow()

    def texFind(self):
        ## mb texture file find
        notlist = ""
        for texture in cmds.ls(type="file"):
            filename = cmds.getAttr('%s.fileTextureName' % texture)
            if filename.find('netapp') > 0:
                filename = filename.replace("/netapp/dexter", "")
            elif filename.find('X:') > 0:
                filename = filename.replace("X:", "/show")

            if os.path.exists(filename):
                self.getlist.append(filename)
            else:
                notlist = "%s %s not exists" % (notlist,
                                                filename.split('/')[-1])
                return notlist

    def dbSend(self):
        ## dbsend list make and send
        name = self.ui.title_txt.text()
        project = self.ui.type_combo.currentText()
        category = self.ui.category_combo.currentText()

        if name and project and category:
            ## name, project, category null check
            dbsend = {}
            texpaths = ''
            notlist = ''
            notlist = self.texFind()
            if notlist: ## texture file link not found warning dialog
                messageBox(">> Warning : not exits Texture file", notlist, 'warning', ['OK'])
            else:
                orgfilename = self.sourcename.split('.')[0]
                tag_txt = self.ui.tag_txt.toPlainText()
                dbtag = tag_txt.split('\n')
                dbtag.append(orgfilename)
                dbsend['org'] = self.loadpath
                dbsend['org_name'] = self.sourcename
                dbsend['org_path'] = self.sourcepath
                dbsend['org_file'] = orgfilename
                dbsend['name'] = name
                dbsend['project'] = project
                dbsend['category'] = category
                dbsend['texture'] = self.getlist
                dbsend['thumbnail'] = self.thumbnail
                dbsend['tags'] = dbtag
                print dbsend
                Laydb = LayInventorydb()
                Laydb.dbImport(dbsend)
        else:
            ## name, project, category null warning dialog
            messageBox(">> Warning : null value ",
                       "name, project, category, list not null!!", 'warning', ['OK'])

def messageBox(titles = 'Layout for Inventory UpLoad', messages = None,
               icons= 'warning',  # warning, question, information, critical
               buttons=['OK', 'Cancel']):
    bgcolor = [0.9, 0.6, 0.6]
    cmds.confirmDialog(title=titles, message=messages,
                       messageAlign='center', icon=icons,
                       button=buttons, backgroundColor=bgcolor)


#{'org_name': 'C0010_v03_w01_cam.mb', 'category': u'etc',
# 'org_path': '/dexter/Cache_DATA/LAY/006_Project/1987/shotPrv/END/_dev/ENDprv_110',
# 'name': u'test', 'tags': [u'testdddddd', u'aaaaaa', 'C0010_v03_w01_cam'],
# 'texture': [], 'project': u'camera',
# 'org': '/dexter/Cache_DATA/LAY/006_Project/1987/shotPrv/END/_dev/ENDprv_110/C0010_v03_w01_cam.mb',
# 'org_file': 'C0010_v03_w01_cam_preview',
# 'thumbnail': '/dexter/Cache_DATA/LAY/006_Project/1987/shotPrv/END/_dev/ENDprv_110/C0010_v03_w01_cam_thumb.jpg'}

