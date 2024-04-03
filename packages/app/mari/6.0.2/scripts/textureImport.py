#encoding=utf-8
#--------------------------------------------------------------------------------
#
#    Dexter CG-Supervisor
#
#        Sanghun Kim, rman.td@gmail.com
#
#    rman.td 2016.04.02 $2
#
#-------------------------------------------------------------------------------

import os, sys
import json
import glob
import mari
from PySide2 import QtWidgets, QtGui, QtCore

# from pymodule.Qt import QtWidgets
# from pymodule.Qt import QtGui
# from pymodule.Qt import QtCore
import dxMari as dxm

#-------------------------------------------------------------------------------
#    Core
class TexturePreview:
    def __init__( self, txpath ):
        self.txpath        = txpath
        self.marichannels = []
        self.geo = mari.current.geo()

    # create channels and optimized
    def createChannels( self ):
        # create channel
        current_channel = []
        for i in self.geo.channelList():
            current_channel.append( i.name() )
        for i in self.marichannels:
            if not i in current_channel:
                self.geo.createChannel( i, 2048, 2048, 8 )

    # create layer
    def createLayers( self ):
        layername = os.path.basename( self.txpath )
        self.txinfo = dxm.getMetadata_textureNameInfo()
        uvindexList = list( self.txinfo.keys() )
        uvindexList.remove( 'multiuv' )
        for i in self.marichannels:
            channelObj = self.geo.channel( i )
            try:
                channelObj.layer(layername)
            except:
                channelObj.createPaintableLayer( layername )

            layerObj = channelObj.layer(layername)
            imgSet   = layerObj.imageSet()
            import_image_paths = list()
            for x in uvindexList:
                img  = imgSet.image( x )
                name = self.txinfo[x] + '_' + i
                if x in self.txinfo['multiuv'].keys():
                    name += '.' + str(self.txinfo['multiuv'][x])
                source = glob.glob( os.path.join( self.txpath, '%s.*' % name ) )
                if source:
                    import_image_paths.append( [img, source[0]] )

            if import_image_paths:
                imgSet.importImages( import_image_paths, imgSet.SCALE_THE_PATCH )

    def doIt( self ):
        if not self.marichannels:
            self.marichannels = dxm.getTextureChannels( self.txpath )
        self.createChannels()
        self.createLayers()


#-------------------------------------------------------------------------------
#    UI
class ImportExportedTextureDialog( QtWidgets.QDialog ):
    def __init__( self, parent=None ):
        super( ImportExportedTextureDialog, self ).__init__( parent )
        self.setWindowTitle( 'Import Channel Textre' )

        # ETC
        icons = mari.resources.path('ICONS')
        self.m_channels = None
        self.m_channelWidgetList = list()
        self.m_txinfo = dxm.getMetadata_textureNameInfo()

        # dialog main layout
        mlayout = QtWidgets.QVBoxLayout( self )

        # import file
        imp_layout = QtWidgets.QGridLayout()
        mlayout.addLayout( imp_layout )

        layerfile_label = QtWidgets.QLabel( 'LayerInfo File' )
        imp_layout.addWidget( layerfile_label, 0, 0 )
        self.layerfile_lineEdit = QtWidgets.QLineEdit()
        self.layerfile_lineEdit.setFixedWidth( 400 )
        self.layerfile_lineEdit.setFixedHeight( 32 )
        imp_layout.addWidget( self.layerfile_lineEdit, 0, 1 )
        self.layerfile_pushButton = QtWidgets.QPushButton( QtGui.QIcon(icons + os.sep + 'OpenFile.png'), '' )
        imp_layout.addWidget( self.layerfile_pushButton, 0, 2 )
        # bind command
        self.layerfile_pushButton.clicked.connect( self.layerfileSetProc )
        # ui disable
        if self.m_txinfo:
            self.layerfile_lineEdit.setDisabled( True )
            self.layerfile_pushButton.setDisabled( True )

        txpath_label = QtWidgets.QLabel( 'Texture Path' )
        imp_layout.addWidget( txpath_label, 1, 0 )
        self.txpath_lineEdit = QtWidgets.QLineEdit()
        self.txpath_lineEdit.setFixedWidth( 400 )
        self.txpath_lineEdit.setFixedHeight( 32 )
        imp_layout.addWidget( self.txpath_lineEdit, 1, 1 )
        txpath_pushButton = QtWidgets.QPushButton( QtGui.QIcon(icons + os.sep + 'OpenFile.png') , '' )
        imp_layout.addWidget( txpath_pushButton, 1, 2 )
        # bind command
        txpath_pushButton.clicked.connect( self.txpathSetProc )

        channel_label = QtWidgets.QLabel( 'Channel List' )
        channel_label.setAlignment( QtCore.Qt.AlignCenter )
        mlayout.addWidget( channel_label )

        # scroll area
        channelArea = QtWidgets.QScrollArea()
        channelArea.setMinimumHeight( 250 )
        channelArea.setWidgetResizable( True )
        channelArea.setEnabled( True )
        channelAreaWidget = QtWidgets.QWidget( channelArea )
        # contextmenu
        channelAreaWidget.setContextMenuPolicy( QtCore.Qt.CustomContextMenu )
        channelAreaWidget.customContextMenuRequested.connect( self.channelCheck_Popup )
        # vertical layout
        channelVerticalLayout = QtWidgets.QVBoxLayout( channelAreaWidget )
        # channel grid layout
        self.channelGridLayout = QtWidgets.QGridLayout()
        self.channelGridLayout.setContentsMargins( 5, 5, 5, 5 )
        self.channelGridLayout.setSpacing( 10 )
        channelVerticalLayout.addLayout( self.channelGridLayout )
        # vertical spacing
        spacer = QtWidgets.QSpacerItem(10,10,QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        channelVerticalLayout.addItem( spacer )

        channelArea.setWidget( channelAreaWidget )
        mlayout.addWidget( channelArea )

        # import
        import_pushButton = QtWidgets.QPushButton( 'Import Texture' )
        import_pushButton.setFixedHeight( 40 )
        mlayout.addWidget( import_pushButton )
        # bind command
        import_pushButton.clicked.connect( self.importProc )

        # copyright
        font = QtGui.QFont()
        font.setPointSize(8)
        copyright = QtWidgets.QLabel( '@DexterDigital Texture' )
        copyright.setAlignment( QtCore.Qt.AlignRight )
        copyright.setFont( font )
        mlayout.addWidget( copyright )

        #self.show()

    def channelCheck_Popup( self ):
        menu = QtWidgets.QMenu()
        menu.addAction( 'Select All', self.channelCheck_selectAll )
        menu.addAction( 'Clear All' , self.channelCheck_clearAll )
        menu.exec_( QtWidgets.QCursor.pos() )

    def channelCheck_selectAll( self ):
        if not self.m_channels:
            return
        for c in self.m_channels:
            widgetName = '%s_checkBox' % c.replace(' ', '_')
            exec( 'self.%s.setChecked( True )' % widgetName )

    def channelCheck_clearAll( self ):
        if not self.m_channels:
            return
        for c in self.m_channels:
            widgetName = '%s_checkBox' % c.replace(' ', '_')
            exec( 'self.%s.setChecked( False )' % widgetName )

    # find channels
    def findTxChannels( self ):
        layerfile = self.layerfile_lineEdit.text()
        if layerfile:
            self.m_txinfo = dxm.readTxLayer_byFile( layerfile )
            dxm.geoMetadata_txname_update( self.m_txinfo )

        txpath = self.txpath_lineEdit.text()
        if txpath:
            self.m_channels = dxm.getTextureChannels( txpath )

    # add channels
    def add_channelWidget( self ):
        # clear channelWidget
        if self.m_channelWidgetList:
            for w in self.m_channelWidgetList:
                self.channelGridLayout.removeWidget( w )
                w.deleteLater()
                del w
        self.m_channelWidgetList = list()

        if not self.m_channels:
            return
        for c in self.m_channels:
            layout_index = self.m_channels.index(c)
            row = layout_index / 2
            if layout_index == layout_index / 2 * 2:
                column = 1
            else:
                column = 3
            # horizontal spacing
            spacer1 = QtWidgets.QSpacerItem(10,10,QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
            self.channelGridLayout.addItem( spacer1, row, 0 )
            spacer2 = QtWidgets.QSpacerItem(10,10,QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
            self.channelGridLayout.addItem( spacer2, row, 2 )
            spacer3 = QtWidgets.QSpacerItem(10,10,QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
            self.channelGridLayout.addItem( spacer3, row, 4 )
            widgetName = '%s_checkBox' % c.replace(' ', '_')
            exec( 'self.%s = QtWidgets.QCheckBox( "%s" )' % (widgetName, c) )
            exec( 'self.m_channelWidgetList.append( self.%s )' % widgetName )
            exec( 'self.%s.setChecked( True )' % widgetName )
            exec( 'self.channelGridLayout.addWidget( self.%s, row, column )' % widgetName )

    def layerfileSetProc( self ):
        startpath = mari.resources.path( 'MARI_DEFAULT_IMAGE_PATH' )
        filename = QtWidgets.QFileDialog.getOpenFileName( None, 'Texture LayerInfo File', startpath, '*.json' )
        if filename:
            self.layerfile_lineEdit.setText( filename[0] )
            self.findTxChannels()
            self.add_channelWidget()

    def txpathSetProc( self ):
        startpath = mari.resources.path( 'MARI_DEFAULT_IMAGE_PATH' )
        dir = QtWidgets.QFileDialog.getExistingDirectory( None, 'Texture Import Path', startpath )
        dir = dir.replace( '\\', '/' ) # to linux path
        if dir:
            self.txpath_lineEdit.setText( str(dir) )
            self.findTxChannels()
            self.add_channelWidget()

    def importChannelList( self ):
        result = list()
        if self.m_channels:
            for c in self.m_channels:
                widgetName = '%s_checkBox' % c.replace(' ', '_')
                value = eval( 'self.%s.isChecked()' % widgetName )
                if value:
                    result.append( c )
            return result

    def importProc( self ):
        txinfofile = self.layerfile_lineEdit.text()
        txpath       = self.txpath_lineEdit.text()
        txClass = TexturePreview( txpath )
        txClass.marichannels = self.importChannelList()
        # close window
        self.close()
        # run process
        txClass.doIt()

def show_ui():
    dlg = ImportExportedTextureDialog()
    if mari.projects.current() is None:
        mari.utils.message( 'Open Project' )
        return
    mari.utils.execDialog( dlg )
