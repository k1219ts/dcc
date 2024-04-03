#encoding=utf-8
#!/usr/bin/env python

#-------------------------------------------------------------------------------
#
#   Dexter CG Supervisor
#
#		sanghun.kim		rman.td@gmail.com
#
#	2017.01.23	$2
#-------------------------------------------------------------------------------

import os
import sys
import re
import datetime
import dateutil.parser

from pymongo import MongoClient
import pymongo

import maya.cmds as cmds
import maya.mel as mel

from Qt import QtCore, QtGui, QtWidgets, load_ui

import dxUI


currentScript = os.path.abspath( __file__ )
scriptRoot    = os.path.dirname( currentScript )
pyuiPath	  = os.path.join( scriptRoot, 'pyui' )
#-------------------------------------------------------------------------------
#
#	UI
#
#-------------------------------------------------------------------------------
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
	font-size: 11pt;
}
QDialog {
	background: rgb(10,10,10);
}
QPushButton {
	background-color: rgb(48,48,48);
}
'''

class VersionCheckupWindow( QtWidgets.QMainWindow ):
    def __init__( self, parent=None, rigData=None, layoutData=None ):
        super( VersionCheckupWindow, self ).__init__( parent )
        uifile = os.path.join( pyuiPath, 'db_version_view_v01.ui' )
        dxUI.setup_ui( uifile, self )
        self.setStyleSheet( rcss )

        # current scene data
        self.rigData 	= rigData		# {asset: [(node, version), (node, version)], asset: []}
        self.layoutData = layoutData	# {asset: version, asset: version}

        current_file  = cmds.file( q=True, sn=True )
        source 		  = current_file.split('/')
        self.showName = source[ source.index('show')+1 ]

        # icon
        self.ok_map		= QtGui.QPixmap( os.path.join(pyuiPath, 'okPaths.png') )
        self.broken_map = QtGui.QPixmap( os.path.join(pyuiPath, 'brokenPaths.png') )
        self.rigIcon	= QtGui.QIcon()
        self.rigIcon.addFile( os.path.join(pyuiPath, 'out_dxRig.png') )
        self.assetIcon	= QtGui.QIcon()
        self.assetIcon.addFile( os.path.join(pyuiPath, 'out_dxAbcArchive.png') )

        # logo
        logo_map = QtGui.QPixmap( os.path.join(pyuiPath, 'DEXTER_DIGITAL_W.png') )
        self.logo_label.setPixmap( logo_map )

#		self.move( parent.frameGeometry().center() - self.frameGeometry().center() )

        self.version_tableWidget.setColumnCount( 4 )
        self.version_tableWidget.setHorizontalHeaderLabels(
                ['Node', 'Current Version', 'Target Version', 'Update'] )
        self.version_tableWidget.verticalHeader().setVisible( False )
        self.version_tableWidget.horizontalHeader().resizeSection( 0, 350 )
        self.version_tableWidget.horizontalHeader().setSectionResizeMode(
                                        1, QtWidgets.QHeaderView.Stretch )
        self.version_tableWidget.horizontalHeader().setSectionResizeMode(
                                        2, QtWidgets.QHeaderView.Stretch )
        self.version_tableWidget.horizontalHeader().resizeSection( 3, 80 )

        self.setDataUI()

        # command bind
        self.cancel_pushButton.clicked.connect( self.closeDialog )
        self.update_pushButton.clicked.connect( self.updateProcess )

        self.show()

    def closeDialog( self ):
        self.close()

    #---------------------------------------------------------------------------
    def setDataUI( self ):
        rowSize = 0
        # rigData
        for i in self.rigData:
            rowSize += len(self.rigData[i])
        # layoutData
        rowSize += len(self.layoutData.keys())

        self.version_tableWidget.setRowCount( rowSize )

        rowStart = 0
        # rigData
        for a in self.rigData:
            for node, version in self.rigData[a]:
                node_item, cver_item, tver_comboBox, update_widget = self.rig_tableWidgetItem( a, node, version )
                self.version_tableWidget.setItem( rowStart, 0, node_item )
                self.version_tableWidget.setItem( rowStart, 1, cver_item )
                if tver_comboBox:
                    self.version_tableWidget.setCellWidget( rowStart, 2, tver_comboBox )
                if update_widget:
                    self.version_tableWidget.setCellWidget( rowStart, 3, update_widget )
                rowStart += 1
        # layoutData
        for a in self.layoutData:
            asset_item, cver_item, tver_comboBox, update_widget = self.layout_tableWidgetItem( a, self.layoutData[a] )
            self.version_tableWidget.setItem( rowStart, 0, asset_item )
            self.version_tableWidget.setItem( rowStart, 1, cver_item )
            if tver_comboBox:
                self.version_tableWidget.setCellWidget( rowStart, 2, tver_comboBox )
            if update_widget:
                self.version_tableWidget.setCellWidget( rowStart, 3, update_widget )
            rowStart += 1

    def rig_tableWidgetItem( self, assetName, nodeName, version ):
        node_item = QtWidgets.QTableWidgetItem()
        node_item.setText( nodeName )
        node_item.setTextAlignment( QtCore.Qt.AlignVCenter | QtCore.Qt.AlignHCenter )
        node_item.setIcon( self.rigIcon )

        cver_item = QtWidgets.QTableWidgetItem()
        cver_item.setText( version )
        cver_item.setTextAlignment( QtCore.Qt.AlignVCenter | QtCore.Qt.AlignHCenter )

        tver_comboBox = None; update_widget = None
        tverList = getDB_assetData( self.showName, assetName, 'rig.pub' )
        if tverList:
            tver_comboBox = QtWidgets.QComboBox( self )
            tver_lineEdit = QtWidgets.QLineEdit()
            tver_lineEdit.setReadOnly( True )
            tver_comboBox.setLineEdit( tver_lineEdit )
            tver_comboBox.lineEdit().setAlignment( QtCore.Qt.AlignCenter )
            tver_comboBox.addItems( tverList )

            update_widget = QtWidgets.QLabel( self )
            update_widget.setAlignment( QtCore.Qt.AlignCenter )
            if version == tverList[0]:
                update_widget.setPixmap( self.ok_map )
            else:
                update_widget.setPixmap( self.broken_map )

        return node_item, cver_item, tver_comboBox, update_widget

    def layout_tableWidgetItem( self, assetName, version ):
        asset_item = QtWidgets.QTableWidgetItem()
        asset_item.setText( assetName )
        asset_item.setTextAlignment( QtCore.Qt.AlignVCenter | QtCore.Qt.AlignHCenter )
        asset_item.setIcon( self.assetIcon )

        cver_item = QtWidgets.QTableWidgetItem()
        cver_item.setText( version )
        cver_item.setTextAlignment( QtCore.Qt.AlignVCenter | QtCore.Qt.AlignHCenter )

        tver_comboBox = None; update_widget = None;
        tverList = getDB_assetData( self.showName, assetName, 'model.pub' )
        if tverList:
            tver_comboBox = QtWidgets.QComboBox( self )
            tver_lineEdit = QtWidgets.QLineEdit()
            tver_lineEdit.setReadOnly( True )
            tver_comboBox.setLineEdit( tver_lineEdit )
            tver_comboBox.lineEdit().setAlignment( QtCore.Qt.AlignCenter )
            tver_comboBox.addItems( tverList )

            update_widget = QtWidgets.QLabel( self )
            update_widget.setAlignment( QtCore.Qt.AlignCenter )
            if version == tverList[0]:
                update_widget.setPixmap( self.ok_map )
            else:
                update_widget.setPixmap( self.broken_map )
                tver_comboBox.setStyleSheet( 'QComboBox {color: rgb(255,95,98)}' )

        return asset_item, cver_item, tver_comboBox, update_widget


    #---------------------------------------------------------------------------
    def updateProcess( self ):
        cmds.warning( 'Not yet working!' )
        self.close()



#-------------------------------------------------------------------------------
#
#	Core
#
#-------------------------------------------------------------------------------
def getVersionInfo():
    global verWindow
    if cmds.window( 'verCheckUpWindow', exists=True, q=True ):
        cmds.deleteUI( 'verCheckUpWindow' )

    rdata = getRigInfo()
    ldata = getLayoutInfo()
    print rdata, ldata
    if not rdata and not ldata:
        return
    verWindow = VersionCheckupWindow( rigData=rdata, layoutData=ldata )


def getRigInfo():
    # asset : [ (node, version), (node, version) ]
    data = dict()
    for i in cmds.ls( rn=True, type='dxRig' ):
        fileName = cmds.referenceQuery( i, filename=True )
        version  = os.path.splitext( os.path.basename(fileName) )[0]
        assetName= version.split('_rig')[0]
        if not data.has_key( assetName ):
            data[assetName] = list()
        data[assetName].append( (i, version) )
    return data

def getLayoutInfo():
    # asset : version
    data = dict()
#	for i in cmds.ls( type='dxAbcArchive' ):
#		action = cmds.getAttr( '%s.action' % i )
#		if action == 2:	# Layout Export
#			abcFile  = cmds.getAttr( '%s.abcFileName' % i )
#			source   = abcFile.split('/')
#			if 'asset' in source:
##				assetName = source[ source.index('asset')+2 ]
#				assetName = get_abc_assetName( abcFile )
#				version   = os.path.splitext( os.path.basename(abcFile) )[0]
#				version   = get_versionName( version )
#				data[assetName] = version
    return data


#-------------------------------------------------------------------------------
def get_versionName( name ):
    version = name
    # low, mid remove
    version = version.replace('_low', '')
    version = version.replace('_mid', '')
    return version

def get_abc_assetName( fileName ):
    baseName = os.path.basename( fileName )
    baseName = os.path.splitext( baseName )[0]

    assetName = baseName
    # version split
    vers = re.compile(r'_v\d+').findall( baseName )
    if vers:
        assetName = assetName.split(vers[0])[0]
    # low, mid remove
    assetName = assetName.replace( '_low', '' )
    assetName = assetName.replace( '_mid', '' )
    # task remove ( model )
    assetName = assetName.replace( '_model', '' )

    return assetName

def getDB_assetData( showName, assetName, myData ):
    connection = MongoClient( '10.0.0.12:27017, 10.0.0.13:27017' )
    db = connection.ASSET[showName]

    docs = db.find( {'show':showName, 'name':assetName, myData: {'$exists':True}} )
    if docs.count() == 1:
        src = myData.split('.')
        data = docs[0][src[0]][src[1]]
        return dbTimeSort(data)

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

