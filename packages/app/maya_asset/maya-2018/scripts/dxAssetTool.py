import dxAsset
# import dxAssetExport

import dxCommon

import maya.cmds as cmds
import maya.mel as mel

# import Qt.QtGui as QtGui
# import Qt.QtCore as QtCore
# import Qt.QtWidgets as QtWidgets

from PySide2 import QtWidgets
from PySide2 import QtCore
from PySide2 import QtGui

import os
import string
import time
import json

from dxAssetUI import Ui_assetCheckup_window


currentScript = os.path.abspath( __file__ )
scriptRoot = os.path.dirname( currentScript )
uiFile = os.path.join( scriptRoot, 'dxAssetUI.ui' )

class AssetCheckWin(QtWidgets.QMainWindow):
    def __init__( self, parent = dxCommon.getMayaWindow()):
        QtWidgets.QMainWindow.__init__(self, parent)

        self.ui = Ui_assetCheckup_window()
        self.ui.setupUi(self)

        # center of the screen
        mayaWindow = parent
        self.move( mayaWindow.frameGeometry().center() - self.frameGeometry().center() )

        self.m_currentSelection = cmds.ls( sl=True )
        # log
        self.m_log = {}

        # ui element
        self.m_geoCheckupUi = [
                'threeside', 'nside', 'concave', 'holed',
                'twisted', 'lamina', 'overlapface', 'overlapmesh',
                'zeroface', 'zerouv', 'outsideuv', 'multiuvset',
                'zeroedge', 'twoedgevtx', 'nonmanifold'
                ]
        self.m_geoSetupUi = [
                'freezetransform', 'zeropivot', 'zeromovevertex', 'deletehistory',
                'mergevertex', 'normaifoldgeometry'
                ]
        self.m_geoAttrUi = [
                'objectname', 'texture', 'multiuv','materialSet'
                ]
        self.m_categoryUi = {
                'geoCheckup': self.m_geoCheckupUi,
                'geoSetup': self.m_geoSetupUi,
                'geoAttr': self.m_geoAttrUi,
                }
        self.m_uilabelMap = {
                'threeside': '3 Sides',
                'nside': 'more than 4 Sides',
                'concave': 'Concave',
                'holed': 'Holed',
                'twisted': 'Twisted',
                'lamina': 'Lamina',
                'overlapface': 'Overlap Face',
                'overlapmesh': 'Overlap Mesh',
                'zeroface': 'Zero Area Face',
                'zerouv': 'Zero UV',
                'outsideuv': 'OutSide UV',
                'multiuvset': 'multi UVSet',
                'zeroedge': 'Zero length Edge',
                'twoedgevtx': '2Edge Vertex',
                'nonmanifold': 'Non-Manifold',
                'freezetransform': 'Freeze Transform',
                'conformnormal': 'Conform Normal',
                'unlocknormal': 'Unlock Normal',
                'zeropivot': 'Zero Pivot',
                'zeromovevertex': 'Zero Move Vertex',
                'mergevertex': 'Merge Vertex',
                'deletehistory': 'Delete History',
                'uninstance': 'UnInstance all instances',
                'initialshader': 'Initialize Shader',
                'uniquename': 'Unique Name',
                'objectname': 'objectName',
                'texture': 'Display Layer',
                'multiuv': 'Multi UV',
                'materialSet' : 'MaterialSet',
                'renderstats': 'RenderStats',
                'foursidedfaces' : '4-sided faces',
                'normaifoldgeometry' : 'Nonmanifold geometry'
                }
        self.m_uilabelInvertMap = {}
        for i in self.m_uilabelMap:
            self.m_uilabelInvertMap[ self.m_uilabelMap[i] ] = i

        self.progDialog = QtWidgets.QProgressDialog( self )
        self.progDialog.setRange( 0, 100 )
        self.progDialog.setWindowTitle( 'processing...' )
        self.progDialog.close()

        # set
        self.setWindowTitle( 'Asset Checkup Tool' )
        self.setAssetName()

        # geometry checkup popup-menu
        self.createGeoPopup()

        # messageLog popup-menu
        self.ui.messageLog_listWidget.setContextMenuPolicy( QtCore.Qt.CustomContextMenu )
        self.ui.messageLog_listWidget.customContextMenuRequested.connect( self.messageLogPopup )

        # command binding
        # self.ui.fileDialog_pushButton.clicked.connect( self.openFileDialog )
        self.ui.fileDialog_pushButton.clicked.connect(self.reloadAssetName)
        self.ui.geocheckup_pushButton.clicked.connect( self.geometryCheckupAction )
        self.ui.geosetup_pushButton.clicked.connect( self.geometrySetupAction )
        self.ui.geoattr_pushButton.clicked.connect( self.geometryAttributesAction )
        self.ui.checkupAll_pushButton.clicked.connect( self.checkupAll )
        # self.ui.export_pushButton.clicked.connect( self.exportMesh )
        # self.ui.close_pushButton.clicked.connect( self.closeWindow )
        self.ui.processLog_listWidget.itemSelectionChanged.connect( self.processLogAction )
        self.ui.messageLog_listWidget.itemDoubleClicked.connect( self.messageLogAction )

        setColor = 'QWidget {color: rgb(219,126,0)}'
        self.ui.overlapface_checkBox.setStyleSheet( setColor )
        self.ui.overlapmesh_checkBox.setStyleSheet( setColor )
        self.ui.normaifoldgeometry_checkBox.setStyleSheet( setColor )

        # load ui value
        valuefile = os.path.join( os.getenv('HOME'), 'tmp', 'asset_checkup_tool.preset' )
        print valuefile
        if os.path.exists( valuefile ):
            f = open( valuefile, 'r' )
            try:
                data = json.load( f )
                for i in data:
                    if self.m_uilabelMap.has_key(i):
                        if data[i]:
                            exec( 'self.ui.%s_checkBox.setCheckState( QtCore.Qt.Checked )' % i )
                        else:
                            exec( 'self.ui.%s_checkBox.setCheckState( QtCore.Qt.Unchecked )' % i )
            except:
                pass

        self.ui.normaifoldgeometry_checkBox.setChecked(False)

        self.show()

    def keyPressEvent( self, event ):
        if event.key() == QtCore.Qt.Key_Escape:
            self.close()

    def reloadAssetName(self):
        selected =cmds.ls(sl=1)[0]
        if selected:
            assetName = selected.split('_model_')[0]
            if '_' in assetName:
                splitName = assetName.split('_')
                basePath = 'asset/%s/branch/%s/texture' % (splitName[0], splitName[-1])
            else:
                basePath = 'asset/%s/texture' % assetName
            self.ui.assetname_lineEdit.setText(basePath)
        else:
            pass

    def assetPath( self, dir ):
        src = dir.split( '/' )
        if 'show' in src and 'asset' in src:
            if not "branch" in src:
                return "/".join(["asset", src[src.index('asset')+2], "texture"])
            else:
                return "/".join(["asset", src[src.index('asset') + 2], "branch", src[src.index("branch") + 1], "texture"])
        elif 'show' in src and 'shot' in src:
            basePath = "/".join( ["shot", src[src.index('shot') + 1], src[src.index('shot') + 2]] )
            if not "branch" in src:
                return "/".join([basePath, src[src.index('model') + 2], "texture"])
            else:
                return "/".join([basePath, src[src.index('model') + 2], "branch", src[src.index("branch") + 1], "texture"])

    def setAssetName( self ):
        sceneName= cmds.file( q=True, sn=True )
        if sceneName:
            getValue = self.assetPath( sceneName )
        else:
            projPath = cmds.workspace( q=True, rd=True )
            getValue = self.assetPath( projPath )
        if getValue:
            self.ui.assetname_lineEdit.setText( getValue )

    def openFileDialog( self ):
        dir = cmds.fileDialog2( fileMode = 3,
                                caption = 'Select Asset Name',
                                okCaption = 'select' )
        if not dir:
            return
        getValue = self.assetPath( dir[0] )
        if getValue:
            self.ui.assetname_lineEdit.setText( getValue )

    #------------------------------------------------------------------------
    # add popup-menu
    def createGeoPopup( self ):
        print self.m_categoryUi.items()
        for key, values in self.m_categoryUi.items():
            for i in values:
                exec( 'self.ui.%s_checkBox.setContextMenuPolicy( QtCore.Qt.CustomContextMenu )' % i )
                exec( 'self.ui.%s_checkBox.customContextMenuRequested.connect( self.addGeoPopupMenu )' % i )

    def addGeoPopupMenu( self ):
        sender = self.sender()
        uiname = sender.objectName().split('_')[0]
        category = None
        for key, values in self.m_categoryUi.items():
            if uiname in values:
                category = key
        if not category:
            return
        menu = QtWidgets.QMenu()
        menu.addAction( 'just select one', lambda: self.uiCategorySelectOne( sender, category ) )
        menu.addAction( 'select all', lambda: self.uiCategorySelectAll( sender, category ) )
        menu.exec_( QtGui.QCursor.pos() )

    def uiCategorySelectOne( self, parent, category ):
        for i in self.m_categoryUi[category]:
            exec( 'self.ui.%s_checkBox.setCheckState( QtCore.Qt.Unchecked )' % i )
        parent.setCheckState( QtCore.Qt.Checked )

    def uiCategorySelectAll( self, parent, category ):
        for i in self.m_categoryUi[category]:
            exec( 'self.ui.%s_checkBox.setCheckState( QtCore.Qt.Checked )' % i )
    #------------------------------------------------------------------------


    #------------------------------------------------------------------------
    # geometry checkup
    def geometryCheckup( self ):
        options = {}
        for i in self.m_geoCheckupUi:
            getValue = int( eval('self.ui.%s_checkBox.isChecked()' % i) )
            options[i] = getValue

        start = time.time()
        assetGeo = dxAsset.AssetGeometry()
        self.m_currentSelection = assetGeo.m_currentSelection
        # option control
        for i in options:
            exec( 'assetGeo.%s = %s' % (i, options[i]) )
        dataDict = assetGeo.check( options )
        for i in dataDict:
            self.m_log[i] = dataDict[i]
        for i in options:
            if options[i]:
                if self.m_log[i]:
                    item = QtWidgets.QListWidgetItem()
                    item.setForeground( QtCore.Qt.red )
                    item.setText( self.m_uilabelMap[i] )
                    self.ui.processLog_listWidget.addItem( item )
                else:
                    exec( 'self.ui.%s_checkBox.setCheckState( QtCore.Qt.Unchecked )' % i )
                    self.ui.processLog_listWidget.addItem( '%s .... clear' % self.m_uilabelMap[i] )

        print 'Elapsed time : %s' % (time.time() - start)

    def geometryCheckupAction( self ):
        # progress dialog
        self.progDialog.show()

        # clear log
        self.m_log.clear()
        self.ui.messageLog_listWidget.clear()
        self.ui.processLog_listWidget.clear()

        # geometry checkup
        item = QtWidgets.QListWidgetItem()
        item.setForeground( QtGui.QColor(72,165,255) )
        item.setText( '------>> Geometry Checkup <<------' )
        self.ui.processLog_listWidget.addItem( item )

        #
        self.geometryCheckup()
        self.progDialog.close()
    #------------------------------------------------------------------------


    #------------------------------------------------------------------------
    # geometry setup
    def geometrySetup( self ):
        processes = []
        # default process
        defaults = [
                'conformnormal', 'unlocknormal', 'uninstance',
                'initialshader'
                ]
        for i in defaults:
            self.ui.processLog_listWidget.addItem( self.m_uilabelMap[i] )
        processes += defaults
        for i in self.m_geoSetupUi:
            getValue = int( eval( 'self.ui.%s_checkBox.isChecked()' % i ) )
            if getValue:
                self.ui.processLog_listWidget.addItem( self.m_uilabelMap[i] )
                processes.append( i )
        if not processes:
            return
        assetGeo = dxAsset.AssetGeometry()
        assetGeo.setupObjects( processes )

    def geometrySetupAction( self ):
        # clear log
        self.m_log.clear()
        self.ui.messageLog_listWidget.clear()
        self.ui.processLog_listWidget.clear()
        # geometry setup
        item = QtWidgets.QListWidgetItem()
        item.setForeground( QtGui.QColor(72,165,255) )
        item.setText( '------>> Geometry Setup <<------' )
        self.ui.processLog_listWidget.addItem( item )
        self.geometrySetup()
    #------------------------------------------------------------------------


    #------------------------------------------------------------------------
    # geometry attributes
    def geometryAttributes( self ):
        # asset name
        assetName = self.ui.assetname_lineEdit.text()
        if not assetName:
            m = cmds.confirmDialog( title = 'Warning : AssetName',
                                    message = 'Not find the name of the asset.',
                                    messageAlign = 'center',
                                    button = ['Skip', 'Setup', 'Break'],
                                    defaultButton = 'Setup',
                                    cancelButton = 'Skip',
                                    icon = 'warning' )
            if m == 'Setup':
                self.openFileDialog()
            if m == 'Break':
                self.ui.processLog_listWidget.addItem( '>>> Geometry Attributes Breaked !!!' )
                return

        processes = []
        defaults = ['uniquename', 'renderstats']
        for i in defaults:
            self.ui.processLog_listWidget.addItem( self.m_uilabelMap[i] )
        processes += defaults
        for i in self.m_geoAttrUi:
            getValue = int( eval('self.ui.%s_checkBox.isChecked()' % i) )
            if getValue:
                processes.append( i )
                self.ui.processLog_listWidget.addItem( self.m_uilabelMap[i] )
        if not processes:
            return

        assetAttr = dxAsset.AssetAttribute()
        self.m_currentSelection = assetAttr.m_currentSelection
        assetAttr.m_assetname = self.ui.assetname_lineEdit.text()

        result = assetAttr.add(processes)


        for i in result:
            self.m_log[i] = result[i]
            items = self.ui.processLog_listWidget.findItems( self.m_uilabelMap[i], QtCore.Qt.MatchWrap )
            for w in items:
                w.setForeground( QtCore.Qt.red )

    def geometryAttributesAction( self ):
        # clear log
        self.m_log.clear()
        self.ui.messageLog_listWidget.clear()
        self.ui.processLog_listWidget.clear()
        # attributes
        item = QtWidgets.QListWidgetItem()
        item.setForeground( QtGui.QColor(72,165,255) )
        item.setText( '------>> Geometry Attributes <<------' )
        self.ui.processLog_listWidget.addItem( item )
        #
        state = self.geometryAttributes()

    #------------------------------------------------------------------------


    def checkupAll( self ):
        # progress dialog
        self.progDialog.show()

        # clear log
        self.m_log.clear()
        self.ui.messageLog_listWidget.clear()
        self.ui.processLog_listWidget.clear()

        # attributes
        item = QtWidgets.QListWidgetItem()
        item.setForeground( QtGui.QColor(72,165,255) )
        item.setText( '------>> Geometry Attributes <<------' )
        self.ui.processLog_listWidget.addItem( item )
        #
        self.geometryAttributes()

        # checkup
        item = QtWidgets.QListWidgetItem()
        item.setForeground( QtGui.QColor(72,165,255) )
        item.setText( '------>> Geometry Checkup <<------' )
        self.ui.processLog_listWidget.addItem( item )
        #
        self.geometryCheckup()

        # maya cleanup
        item = QtWidgets.QListWidgetItem()
        item.setForeground( QtGui.QColor(72,165,255) )
        item.setText( '------>> Maya Cleanup <<------' )
        self.ui.processLog_listWidget.addItem( item )
        #

        # setup
        item = QtWidgets.QListWidgetItem()
        item.setForeground( QtGui.QColor(72,165,255) )
        item.setText( '------>> Geometry Setup <<------' )
        self.ui.processLog_listWidget.addItem( item )
        #
        self.geometrySetup()

        self.progDialog.close()



    #------------------------------------------------------------------------
    # log widget
    def processLogAction( self ):
        self.ui.messageLog_listWidget.clear()
        items = self.ui.processLog_listWidget.selectedItems()
        for i in items:
            citem = str( i.text() )
            if citem in self.m_uilabelInvertMap.keys():
                logname = self.m_uilabelInvertMap[citem]
                if citem == 'OutSide UV':
                    if logname in self.m_log.keys():
                        self.add_outsideuv_messageLog()
                else:
                    if logname in self.m_log.keys():
                        getMessage = self.m_log[logname]
                        if getMessage:
                            self.ui.messageLog_listWidget.addItems( getMessage )

    def add_outsideuv_messageLog( self ):
        # attrname = 'rman__riattr__user_txmultiUV'
        attrname = 'txmultiUV'
        getMessage = self.m_log['outsideuv']
        if getMessage:
            for i in getMessage:
                shape = cmds.ls( i, dag=True, s=True )
                item = QtWidgets.QListWidgetItem()
                item.setText( i )
                if cmds.attributeQuery( attrname, n=shape[0], ex=True ):
                    getValue = cmds.getAttr( '%s.%s' % (shape[0], attrname) )
                    if getValue:
                        item.setForeground( QtGui.QColor(72,165,255) )
                self.ui.messageLog_listWidget.addItem( item )

    def messageLogAction( self ):
        self.messageLogSelectComponents()

    def messageLogPopup( self, pos ):
        menu = QtWidgets.QMenu()
        menu.addAction( 'Select Components', self.messageLogSelectComponents )
        menu.addAction( 'Select Objects', self.messageLogSelectObjects )
        menu.addAction( 'IsolateSelect Objects', self.messageLogIsolateSelectObjects )
        items = self.ui.processLog_listWidget.selectedItems()
        if items:
            if str(items[0].text()) == 'OutSide UV':
                menu.addAction( 'Add Multi-Tile Attributes', self.messageLogAddMultiTileAttribute )
        menu.exec_( QtGui.QCursor.pos() )

    def messageLogSelectComponents( self ):
        selCompList = []
        selObjList = []
        items = self.ui.messageLog_listWidget.selectedItems()
        for i in items:
            citem = str( i.text() )
            if cmds.objExists( citem ):
                selCompList.append( citem )
                selObjList.append( citem.split('.')[0] )
        # select object
        cmds.selectMode( object=True )
        cmds.select( selObjList )
        # select component
        src = selCompList[0].split('.')
        if len(src) > 1:
            cmds.selectMode( component=True )
            if src[-1][0] == 'f':
                cmds.selectType( polymeshFace=True, alo=True )
            elif src[-1][0] == 'e':
                cmds.selectType( polymeshEdge=True, alo=True )
            elif src[-1][0] == 'v':
                cmds.selectType( polymeshVertex=True, alo=True )
            else:
                cmds.selectType( alc=True, alo=True )
            cmds.select( selCompList )

    def messageLogSelectObjects( self ):
        selectlist = []
        items = self.ui.messageLog_listWidget.selectedItems()
        for i in items:
            citem = str( i.text() )
            if cmds.objExists( citem ):
                selectlist += cmds.ls( citem.split('.')[0] )
        mel.eval( 'changeSelectMode -object' )
        cmds.select( selectlist )

    def messageLogIsolateSelectObjects( self ):
        modelpanels = cmds.getPanel( type='modelPanel' )
        focuspanel  = cmds.getPanel( wf=True )
        if focuspanel in modelpanels:
            mel.eval( 'enableIsolateSelect %s 0' % focuspanel )
            self.messageLogSelectObjects()
            mel.eval( 'enableIsolateSelect %s 1' % focuspanel )

    def messageLogAddMultiTileAttribute( self ):
        selectlist = []
        items = self.ui.messageLog_listWidget.selectedItems()
        for i in items:
            citem = str( i.text() )
            if cmds.objExists( citem ):
                selectlist.append( citem )
        attrname = 'txmultiUV'
        for i in cmds.ls(selectlist, dag=True, type='surfaceShape', ni=True):
            if not cmds.attributeQuery( attrname, n=i, ex=True ):
                cmds.addAttr( i, ln=attrname, at='long' )
            cmds.setAttr( '%s.%s' % (i, attrname),  1 )
    #------------------------------------------------------------------------

def show_ui():
    if cmds.window('assetCheckup_window', exists=True, q=True):
        cmds.deleteUI( 'assetCheckup_window' )

    app = AssetCheckWin()
    app.show()
#     cmds.showWindow(winDialog)
