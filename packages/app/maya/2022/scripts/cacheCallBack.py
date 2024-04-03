#encoding=utf-8
#!/usr/bin/env python

#-------------------------------------------------------------------------------
#
#   Dexter CG Supervisor
#
#		sanghun.kim		rman.td@gmail.com
#
#	2017.02.24	$3
#-------------------------------------------------------------------------------

import os, sys
import json

import maya.cmds as cmds
import maya.mel as mel

from Qt import QtCore, QtGui, QtWidgets, load_ui

import dxUI
import dxDB

currentScript = os.path.abspath( __file__ )
scriptRoot    = os.path.dirname( currentScript )
pyuiPath      = os.path.join( scriptRoot, 'pyui' )

#def setup_ui( uifile, base_instance=None ):
#	ui = load_ui( uifile )
#	if not base_instance:
#		return ui
#	else:
#		for member in dir(ui):
#			if not member.startswith('__') and member is not 'staticMetaObject':
#				setattr( base_instance, member, getattr(ui, member) )
#		return ui

rcss = '''
QHeaderView {
	font-size: 10pt;
}
QTableWidget {
	font-size: 11pt;
}
QTableWidgetItem {
	font-size: 11pt;
}
QComboBox {
	font-size: 12pt;
}
QDialog {
	background: rgb(10,10,10);
}
QPushButton {
	background-color: rgb(48,48,48);
}
'''

class CacheCheckupWindow( QtWidgets.QMainWindow ):
    def __init__( self, parent=None, show=None, seq=None, shot=None ):
        super( CacheCheckupWindow, self ).__init__( parent )
        uifile = os.path.join( pyuiPath, 'db_cache_view_v01.ui' )
        dxUI.setup_ui( uifile, self )
        self.setStyleSheet( rcss )

        # variable member
        self.DB = None	# db show documents
        self.m_showName  = show
        self.m_seqName   = seq
        self.m_shotName  = shot
        self.m_cacheData = None

        # MongoDB
        self.connection = dxDB._CONNECT

        # icon
        self.cameraIcon = QtGui.QIcon()
        self.cameraIcon.addFile( os.path.join(pyuiPath, 'camera.png') )
        self.meshIcon = QtGui.QIcon()
        self.meshIcon.addFile( os.path.join(pyuiPath, 'alembic.png') )
        self.zennIcon = QtGui.QIcon()
        self.zennIcon.addFile( os.path.join(pyuiPath, 'zenn.png') )
        self.layoutIcon = QtGui.QIcon()
        self.layoutIcon.addFile( os.path.join(pyuiPath, 'group.png') )
        self.sublayoutIcon = QtGui.QIcon()
        self.sublayoutIcon.addFile( os.path.join(pyuiPath, 'out_transform.png') )

        # logo
        logo_map = QtGui.QPixmap( os.path.join(pyuiPath, 'DEXTER_DIGITAL_W.png') )
        self.logo_label.setPixmap( logo_map )

        # UI Setup
        self.cache_tableWidget.setContextMenuPolicy( QtCore.Qt.CustomContextMenu )
        self.cache_tableWidget.customContextMenuRequested.connect( self.cacheContextMenu )
        self.cache_tableWidget.setColumnCount( 3 )
        self.cache_tableWidget.setHorizontalHeaderLabels(
                ['Element', 'Type', 'Version'] )
        self.cache_tableWidget.verticalHeader().setVisible( False )
        self.cache_tableWidget.horizontalHeader().resizeSection( 0, 300 )
        self.cache_tableWidget.horizontalHeader().resizeSection( 1, 100 )
#        self.cache_tableWidget.horizontalHeader().setResizeMode( 2, QtWidgets.QHeaderView.Stretch )
        self.cache_tableWidget.horizontalHeader().setSectionResizeMode( 2, QtWidgets.QHeaderView.Stretch )

        self.showUI_setup()

        # command bind
        self.show_comboBox.activated.connect( self.seqUI_setup )
        self.seq_comboBox.activated.connect( self.shotUI_setup )
        self.shot_comboBox.activated.connect( self.cacheUI_setup )
        self.cancel_pushButton.clicked.connect( self.closeDialog )
        self.update_pushButton.clicked.connect( self.updateProcess )

        self.show()

    def closeDialog( self ):
        self.close()

    #---------------------------------------------------------------------------
    def showUI_setup( self ):
        shows = self.connection.SHOT.collection_names()
        shows.sort()
        self.show_comboBox.addItems( shows )
        if self.m_showName:
            showIndex = self.show_comboBox.findText( self.m_showName )
            self.show_comboBox.setCurrentIndex( showIndex )
        self.seqUI_setup()

    def seqUI_setup( self ):
        self.m_showName = self.show_comboBox.currentText()
        self.getShotData( self.m_showName )
        seqs = self.m_shotData.keys()
        seqs.sort()
        self.seq_comboBox.clear()
        self.seq_comboBox.addItems( seqs )
        if self.m_seqName:
            seqIndex = self.seq_comboBox.findText( self.m_seqName )
            if seqIndex == -1:
                seqIndex = 0
            self.seq_comboBox.setCurrentIndex( seqIndex )
        self.shotUI_setup()

    def shotUI_setup( self ):
        self.m_seqName = self.seq_comboBox.currentText()
        self.shot_comboBox.clear()
        self.shot_comboBox.addItems( self.m_shotData[self.m_seqName] )
        if self.m_shotName:
            shotIndex = self.shot_comboBox.findText( self.m_shotName )
            if shotIndex == -1:
                shotIndex = 0
            self.shot_comboBox.setCurrentIndex( shotIndex )
        self.cacheUI_setup()


    def cacheUI_setup( self ):
        self.cache_tableWidget.clear()

        self.m_shotName  = self.shot_comboBox.currentText()
        self.m_cacheData = dxDB.getCacheData( self.m_showName, self.m_shotName )

        if self.m_cacheData and self.m_cacheData.has_key( 'zenn' ):
            headers = ['Element', 'Type', 'Version', 'Type', 'Version']
        else:
            headers = ['Element', 'Type', 'Version']
        self.cache_tableWidget.setColumnCount( len(headers) )
        self.cache_tableWidget.setHorizontalHeaderLabels( headers )
        if len(headers) > 3:
#            self.cache_tableWidget.horizontalHeader().setResizeMode( 4, QtWidgets.QHeaderView.Stretch )
			self.cache_tableWidget.horizontalHeader().setSectionResizeMode( 4, QtWidgets.QHeaderView.Stretch )
        self.cache_tableWidget.setRowCount( 0 )

        if not self.m_cacheData:
            return

        rowSize = 0
        if self.m_cacheData.has_key( 'camera' ):
            rowSize += 1
        if self.m_cacheData.has_key( 'mesh' ):
            rowSize += len( self.m_cacheData['mesh'].keys() )
        if self.m_cacheData.has_key( 'layout' ):
            verList = dbTimeSort( self.m_cacheData['layout'] )
            layoutFiles = self.m_cacheData['layout'][verList[0]]['file']
            rowSize += 1
            if type(layoutFiles).__name__ == 'list':
                if len(layoutFiles) > 1:
                    rowSize += len( layoutFiles )

        self.cache_tableWidget.setRowCount( rowSize )

        rowStart = 0
        # Camera
        if self.m_cacheData.has_key( 'camera' ):
            self.camera_tableWidgetItem( rowStart )
            rowStart += 1
        # Mesh
        if self.m_cacheData.has_key( 'mesh' ):
            elementList = self.m_cacheData['mesh'].keys()
            elementList.sort()
            for m in elementList:
                self.mesh_tableWidgetItem( m, rowStart )
                # Zenn
                if self.m_cacheData.has_key('zenn') and self.m_cacheData['zenn'].has_key( m ):
                    self.zenn_tableWidgetItem( m, rowStart )
                rowStart += 1
        # Layout
        if self.m_cacheData.has_key( 'layout' ):
            self.layout_tableWidgetItem( rowStart )
            rowStart += 1
            if type(layoutFiles).__name__ == 'list' and len(layoutFiles) > 1:
                for f in layoutFiles:
                    self.sublayout_tableWidgetItem( f, rowStart )
                    rowStart += 1


    def camera_tableWidgetItem( self, row ):
        type_item = QtWidgets.QTableWidgetItem()
        type_item.setText( 'Camera' )
        type_item.setTextAlignment( QtCore.Qt.AlignVCenter | QtCore.Qt.AlignHCenter )
        type_item.setIcon( self.cameraIcon )
        type_item.setFlags( QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled )
        self.cache_tableWidget.setItem( row, 1, type_item )

        verList = dbTimeSort( self.m_cacheData['camera'] )
        ver_comboBox = QtWidgets.QComboBox( self )
        ver_lineEdit = QtWidgets.QLineEdit()
        ver_lineEdit.setReadOnly( True )
        ver_lineEdit.mouseReleaseEvent = lambda _ : ver_comboBox.showPopup()
        ver_comboBox.setLineEdit( ver_lineEdit )
        ver_comboBox.lineEdit().setAlignment( QtCore.Qt.AlignCenter )
        ver_comboBox.addItems( verList )
        self.cache_tableWidget.setCellWidget( row, 2, ver_comboBox )

    def mesh_tableWidgetItem( self, element, row ):
        element_item = QtWidgets.QTableWidgetItem()
        element_item.setText( element )
        element_item.setTextAlignment( QtCore.Qt.AlignVCenter | QtCore.Qt.AlignHCenter )
        element_item.setFlags( QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled )
        self.cache_tableWidget.setItem( row, 0, element_item )

        type_item = QtWidgets.QTableWidgetItem()
        type_item.setText( 'Mesh' )
        type_item.setTextAlignment( QtCore.Qt.AlignVCenter | QtCore.Qt.AlignHCenter )
        type_item.setIcon( self.meshIcon )
        type_item.setFlags( QtCore.Qt.ItemIsEnabled )
        self.cache_tableWidget.setItem( row, 1, type_item )

        verList = dbTimeSort( self.m_cacheData['mesh'][element] )
        ver_comboBox = QtWidgets.QComboBox( self )
        ver_comboBox.setObjectName( 'meshComboBox_%s' % row )
        ver_lineEdit = QtWidgets.QLineEdit()
        ver_lineEdit.setReadOnly( True )
        ver_lineEdit.mouseReleaseEvent = lambda _ : ver_comboBox.showPopup()
        ver_comboBox.setLineEdit( ver_lineEdit )
        ver_comboBox.lineEdit().setAlignment( QtCore.Qt.AlignCenter )
        ver_comboBox.addItems( verList )
        ver_comboBox.currentIndexChanged.connect( self.meshVersionChange )
        self.cache_tableWidget.setCellWidget( row, 2, ver_comboBox )

    def zenn_tableWidgetItem( self, element, row ):
        type_item = QtWidgets.QTableWidgetItem()
        type_item.setText( 'Zenn' )
        type_item.setTextAlignment( QtCore.Qt.AlignVCenter | QtCore.Qt.AlignHCenter )
        type_item.setIcon( self.zennIcon )
        type_item.setFlags( QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled )
        self.cache_tableWidget.setItem( row, 3, type_item )

        verList = dbTimeSort( self.m_cacheData['zenn'][element] )
        ver_comboBox = QtWidgets.QComboBox( self )
        ver_lineEdit = QtWidgets.QLineEdit()
        ver_lineEdit.setReadOnly( True )
        ver_lineEdit.mouseReleaseEvent = lambda _ : ver_comboBox.showPopup()
        ver_comboBox.setLineEdit( ver_lineEdit )
        ver_comboBox.lineEdit().setAlignment( QtCore.Qt.AlignCenter )
        ver_comboBox.addItems( verList )
        self.cache_tableWidget.setCellWidget( row, 4, ver_comboBox )

        # version checkup
        elementItem   = self.cache_tableWidget.item( row, 0 )
        elementName   = str( elementItem.text() )

        mesh_comboBox = self.cache_tableWidget.cellWidget( row, 2 )
        meshVersion   = mesh_comboBox.currentText()
        meshFile	  = self.m_cacheData['mesh'][elementName][meshVersion]['file']

        zennFile	  = self.m_cacheData['zenn'][elementName][verList[0]]['ref-cache']
        if zennFile != meshFile:
            cmds.warning( 'Zenn cache version mismatch.' )
            ver_comboBox.setStyleSheet( 'QComboBox {color: rgb(255,95,98)}' )

    def layout_tableWidgetItem( self, row ):
        type_item = QtWidgets.QTableWidgetItem()
        type_item.setText( 'Layout' )
        type_item.setTextAlignment( QtCore.Qt.AlignVCenter | QtCore.Qt.AlignHCenter )
        type_item.setIcon( self.layoutIcon )
        type_item.setFlags( QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled )
        self.cache_tableWidget.setItem( row, 1, type_item )

        verList = dbTimeSort( self.m_cacheData['layout'] )
        ver_comboBox = QtWidgets.QComboBox( self )
        ver_comboBox.setObjectName( 'layoutComboBox_%s' % row )
        ver_lineEdit = QtWidgets.QLineEdit()
        ver_lineEdit.setReadOnly( True )
        ver_lineEdit.mouseReleaseEvent = lambda _ : ver_comboBox.showPopup()
        ver_comboBox.setLineEdit( ver_lineEdit )
        ver_comboBox.lineEdit().setAlignment( QtCore.Qt.AlignCenter )
        ver_comboBox.addItems( verList )
        ver_comboBox.activated.connect( self.layoutVersionChange )
        self.cache_tableWidget.setCellWidget( row, 2, ver_comboBox )

    def sublayout_tableWidgetItem( self, fileName, row ):
        type_item = QtWidgets.QTableWidgetItem()
        type_item.setText( 'group' )
        type_item.setTextAlignment( QtCore.Qt.AlignVCenter | QtCore.Qt.AlignHCenter )
        type_item.setIcon( self.sublayoutIcon )
        self.cache_tableWidget.setItem( row, 1, type_item )

        name_item = QtWidgets.QTableWidgetItem()
        name_item.setText( os.path.basename(fileName) )
        name_item.setTextAlignment( QtCore.Qt.AlignVCenter | QtCore.Qt.AlignHCenter )
        self.cache_tableWidget.setItem( row, 2, name_item )


    #---------------------------------------------------------------------------
    def cacheContextMenu( self ):
        items = self.cache_tableWidget.selectedItems()
        if items:
            menu = QtWidgets.QMenu( self )
            menu.addAction( 'Remove SelectItem', self.removeCacheWidgetItem )
            menu.exec_( QtWidgets.QCursor.pos() )

    def removeCacheWidgetItem( self ):
        items = self.cache_tableWidget.selectedItems()
        row = items[0].row()
        typeItem = self.cache_tableWidget.item( row, 1 )
        typeName = str( typeItem.text() )

        if typeName == 'Zenn' or typeName == 'Camera' or typeName == 'group':
            self.cache_tableWidget.removeRow( row )
        if typeName == 'Mesh':
            under_elementItem = self.cache_tableWidget.item( row+1, 0 )
            under_typeItem	  = self.cache_tableWidget.item( row+1, 1 )
            under_typeName	  = None
            if under_typeItem:
                under_typeName= str( under_typeItem.text() )
            if not under_elementItem and under_typeName == 'Zenn':
                self.cache_tableWidget.removeRow( row+1 )
            self.cache_tableWidget.removeRow( row )
        if typeName == 'Layout':
            rowCount = self.cache_tableWidget.rowCount()
            delrows  = range( row, rowCount )
            delrows.reverse()
            for i in delrows:
                self.cache_tableWidget.removeRow( i )

    def meshVersionChange( self ):
        sender = self.sender()
        row    = int( str( sender.objectName() ).split('_')[-1] )

        elementItem = self.cache_tableWidget.item( row, 0 )
        elementName = str( elementItem.text() )

        meshVersion = sender.currentText()
        meshFile    = self.m_cacheData['mesh'][elementName][meshVersion]['file']

        typeItem	= self.cache_tableWidget.item( row+1, 1 )
        if not typeItem:
            return
        typeName	= str( typeItem.text() )
        if typeName == 'Zenn':
            zenn_comboBox = self.cache_tableWidget.cellWidget( row+1, 2 )
            zennVersion = getZennVersion_by_abcFile(
                            self.m_cacheData['zenn'][elementName], meshFile )
            if zennVersion:
                zenn_comboBox.setStyleSheet( 'QComboBox {color: rgb(230,230,230)}' )
                zennIndex = zenn_comboBox.findText( zennVersion )
                zenn_comboBox.setCurrentIndex( zennIndex )
            else:
                cmds.warning( 'Not found zenn cache for alembic mesh version' )
                zenn_comboBox.setStyleSheet( 'QComboBox {color: rgb(255,95,98)}' )

    def layoutVersionChange( self ):
        sender = self.sender()
        row    = int( str(sender.objectName()).split('_')[-1] )

        layoutVersion = sender.currentText()
        layoutFiles   = self.m_cacheData['layout'][layoutVersion]['file']

        rowSize = row+1
        if type(layoutFiles).__name__ == 'list' and len(layoutFiles) > 1:
            rowSize += len(layoutFiles)
            self.cache_tableWidget.setRowCount( rowSize )

            # sub-layout
            rowStart = row+1
            for f in layoutFiles:
                self.sublayout_tableWidgetItem( f, rowStart )
                rowStart += 1
        else:
            self.cache_tableWidget.setRowCount( rowSize )


    #---------------------------------------------------------------------------
    def getShotData( self, showName ):
        self.DB = self.connection.SHOT[ showName ]
        docs = self.DB.find( {'show':showName, 'shot': {'$exists':True}} )

        self.m_shotData = dict()
        for i in range( docs.count() ):
            doc = docs[i]
            shotName = doc['shot']
            seqName  = shotName.split('_')[0]
            if not self.m_shotData.has_key( seqName ):
                self.m_shotData[seqName] = list()
            self.m_shotData[seqName].append( shotName )

        # sort
        for seq in self.m_shotData.keys():
            self.m_shotData[seq].sort()

    def getUiTableData( self ):
        # {
        #	'mesh': list(), # [alembic file list]
        #	'zenn': list( tuple() ), # [ (zenn path, isConstant) ]
        #	'camera': list(), # [alembic file list]
        # }
        data = dict()

        rowCount = self.cache_tableWidget.rowCount()
        for i in range(rowCount):
            elementItem = self.cache_tableWidget.item( i, 0 )
            if elementItem:
                elementName = str( elementItem.text() )

                mesh_comboBox = self.cache_tableWidget.cellWidget( i, 2 )
                meshVersion   = mesh_comboBox.currentText()
                meshFile	  = self.m_cacheData['mesh'][elementName][meshVersion]['file']

                if not data.has_key( 'mesh' ):
                    data['mesh'] = list()
                data['mesh'].append( meshFile )

                typeItem = self.cache_tableWidget.item( i, 3 )
                if typeItem:
                    typeName = str( typeItem.text() )
                    if typeName == 'Zenn':
                        zenn_comboBox = self.cache_tableWidget.cellWidget( i, 4 )
                        zennVersion   = zenn_comboBox.currentText()
                        zennFile	  = self.m_cacheData['zenn'][elementName][zennVersion]['file']
                        isConstant	  = False
                        if self.m_cacheData['zenn'][elementName][zennVersion].has_key('constant'):
                            isConstant= self.m_cacheData['zenn'][elementName][zennVersion]['constant']
                        if not data.has_key( 'zenn' ):
                            data['zenn'] = list()
                        data['zenn'].append( (zennFile, isConstant) )

            else:
                typeItem = self.cache_tableWidget.item( i, 1 )
                if typeItem:
                    typeName = str( typeItem.text() )
                    if typeName == 'Camera':
                        camera_comboBox = self.cache_tableWidget.cellWidget( i, 2 )
                        cameraVersion   = camera_comboBox.currentText()
                        cameraFile = self.m_cacheData['camera'][cameraVersion]['file']
                        if not data.has_key( 'camera' ):
                            data['camera'] = list()
                        data['camera'].append( cameraFile )
                        render_camera = self.m_cacheData['camera'][cameraVersion]['render_camera']
                        if render_camera:
                            if not data.has_key( 'render_camera' ):
                                data['render_camera'] = list()
                            data['render_camera'] += render_camera
                    if typeName == 'Layout':
                        layout_comboBox = self.cache_tableWidget.cellWidget( i, 2 )
                        layoutVersion   = layout_comboBox.currentText()
                        layoutFiles = self.m_cacheData['layout'][layoutVersion]['file']
                        if not data.has_key( 'layout' ):
                            data['layout'] = list()
                        if type(layoutFiles).__name__ == 'list':
                            if len(layoutFiles) > 1:
                                for x in range( i+1, i+len(layoutFiles)+1 ):
                                    nameItem = self.cache_tableWidget.item( x, 2 )
                                    if nameItem:
                                        baseName = str(nameItem.text())
                                        for f in layoutFiles:
                                            name = os.path.basename( f )
                                            if baseName == name:
                                                data['layout'].append( f )
                            else:
                                data['layout'].append( layoutFiles[0] )
                        else:
                            data['layout'].append( layoutFiles )
        return data

    #---------------------------------------------------------------------------
    # Update Process
    def updateProcess( self ):
        data = self.getUiTableData()
        if not data:
            return

        for i in data:
            print(i, data[i])
        # time unit setup( fps )
        if self.m_cacheData.has_key( 'fps' ):
            cmds.currentUnit( time=self.m_cacheData['fps'] )

        # variation data
        variationData = dict()
        variationFile = '/show/%s/asset/shaders/variation.json' % self.m_showName
        if os.path.exists( variationFile ):
            variationData = json.loads( open(variationFile, 'r').read() )

        # mesh import
        if data.has_key( 'mesh' ):
            import sgUI
            ciClass = sgUI.ComponentImport( Files=data['mesh'], World=1 )
            ciClass.m_mode = 1
            ciClass.m_display = 3
            ciClass.doIt()

        # zenn import
        if data.has_key( 'zenn' ):
            if cmds.pluginInfo( 'RenderMan_for_Maya', q=True, l=True ):
                if not cmds.pluginInfo( 'backstageLight', q=True, l=True ):
                    cmds.loadPlugin( 'backstageLight' )
                import lgtUI
                # zenn attributes
                zennAttrData = dict()
                zennAttrFile = '/show/%s/asset/shaders/zennAttributes.json' % self.m_showName
                if os.path.exists( zennAttrFile ):
                    zennAttrData = json.loads( open(zennAttrFile, 'r').read() )

                for zn, const in data['zenn']:
                    baseName = os.path.basename( zn )
                    baseNode = '%s_rig_GRP' % baseName
                    if cmds.objExists( baseNode ):
                        shapeName = cmds.createNode( 'zennArchive', p=baseNode, n='%s_zennArc' % baseName )
                        cmds.addAttr( shapeName, ln='rman__torattr___preShapeScript', dt='string' )
                        cmds.setAttr( '%s.rman__torattr___preShapeScript' % shapeName, 'dxarc', type='string' )
                        lgtUI.zennArchive_setCache( '%s.cachePath' % shapeName, zn )
                        # frame setup
                        if const:
                            cmds.setAttr( '%s.useSequence' % shapeName, 0 )
                            cmds.setAttr( '%s.frameOffset' % shapeName, int(const) )
                        # zenn attributes
                        if zennAttrData and zennAttrData.has_key( baseName.split(':')[-1] ):
                            setZennAttributes( shapeName, zennAttrData[baseName.split(':')[-1]] )
                    else:
                        cmds.warning( 'Zenn Archive Error : Not found parent node' )

        # camera import
        if data.has_key( 'camera' ):
            import dxCameraUI
            for i in data['camera']:
                dxcam = cmds.createNode( 'dxCamera' )
                dxCameraUI.import_cameraFile( '%s.fileName' % dxcam, i )

        # layout import
        if data.has_key( 'layout' ):
            import sgAssembly
            for f in data['layout']:
                sgAssembly.importAssemblyFile( f )

        # plackback
        cmds.playbackOptions( minTime=self.m_cacheData['start'] )
        cmds.playbackOptions( maxTime=self.m_cacheData['end'] )
        cmds.playbackOptions( animationStartTime=self.m_cacheData['start'] )
        cmds.playbackOptions( animationEndTime=self.m_cacheData['end'] )
        cmds.currentTime( self.m_cacheData['start'] )
        self.close()



def showCacheInfo():
    if not dxDB._CONNECT:
        cmds.warning( 'Not found Dexter DB!' )
        return

    global cacheWindow
    if cmds.window( 'cacheCheckUpWindow', exists=True, q=True ):
        cmds.deleteUI( 'cacheCheckUpWindow' )

    show = None; seq = None; shot = None
    current = cmds.file( q=True, sn=True )
    if not current:
        current = cmds.workspace( q=True, rd=True )
    src = current.split('/')
    if 'show' in src:
        show = src[ src.index('show') + 1 ]
    if 'shot' in src:
        seq  = src[ src.index('shot') + 1 ]
        shot = src[ src.index('shot') + 2 ]

    cacheWindow = CacheCheckupWindow( show=show, seq=seq, shot=shot )



#-------------------------------------------------------------------------------
def dbTimeSort( data ):
    verMap = dict()
    for v in data:
        dt = data[v]['time']
        if not verMap.has_key( dt ):
            verMap[dt] = list()
        verMap[dt].append( v )
    dtList = verMap.keys()
    dtList.sort( reverse=True )
    verList = list()
    for t in dtList:
        vers = verMap[t]
        vers.sort( reverse=True )
        verList += vers
    return verList


def getZennVersion_by_abcFile( versionData, abcFile ):
    versions = dict()
    for ver in versionData:
        if versionData[ver]['ref-cache'] == abcFile:
            dt = versionData[ver]['time']
            if not versions.has_key( dt ):
                versions[dt] = list()
            versions[dt].append( ver )
    if versions:
        dtList = versions.keys()
        dtList.sort( reverse=True )
        return versions[dtList[0]][0]


AttributeTypeMap = {
        'rman__riattr__user_txAssetName': 'string',
        'rman__riattr__user_txLayerName': 'string',
        'rman__riattr__user_txVarNum': 'long'
    }
def setZennAttributes( shapeName, attributeData ):
    for attr in attributeData:
        if AttributeTypeMap.has_key( attr ):
            atype = AttributeTypeMap[attr]
            if atype == 'string':
                if not cmds.attributeQuery( attr, n=shapeName, ex=True ):
                    cmds.addAttr( shapeName, ln=attr, dt='string' )
                cmds.setAttr( '%s.%s' % (shapeName, attr), attributeData[attr], type='string' )
            if atype == 'long':
                if not cmds.attributeQuery( attr, n=shapeName, ex=True ):
                    cmds.addAttr( shapeName, ln=attr, at='long' )
                cmds.setAttr( '%s.%s' % (shapeName, attr), attributeData[attr] )
