#encoding=utf-8
#--------------------------------------------------------------------------------
#
#    RenderMan TD
#
#        Sanghun Kim, rman.td@gmail.com
#
#    rman.td 2015.06.24 $1
#
#-------------------------------------------------------------------------------

import os, sys
import json
import mari
from PySide2 import QtWidgets, QtGui, QtCore

# from pymodule.Qt import QtWidgets
# from pymodule.Qt import QtGui
# from pymodule.Qt import QtCore

import sceneDescription

class SceneImportDialog( QtWidgets.QDialog ):
    def __init__( self, parent=None ):
        super( SceneImportDialog, self ).__init__( parent )
        self.setWindowTitle( 'Mari Scene Import' )

        # ETC
        icons = mari.resources.path('ICONS')
        self.m_data = None

        # dialog main layerout
        mlayout = QtWidgets.QVBoxLayout( self )

        # channel info file
        imp_layout = QtWidgets.QGridLayout()
        mlayout.addLayout( imp_layout )

        infofile_label = QtWidgets.QLabel( 'Channel InfoFile' )
        imp_layout.addWidget( infofile_label, 0, 0 )
        self.infofile_lineEdit = QtWidgets.QLineEdit()
        self.infofile_lineEdit.setFixedWidth( 400 )
        self.infofile_lineEdit.setFixedHeight( 32 )
        imp_layout.addWidget( self.infofile_lineEdit, 0, 1 )
        infofile_pushButton = QtWidgets.QPushButton( QtGui.QIcon(icons + os.sep + 'OpenFile.png'), '' )
        infofile_pushButton.clicked.connect( self.set_infofile )
        imp_layout.addWidget( infofile_pushButton, 0, 2 )

        meshfile_label = QtWidgets.QLabel( 'Geometry File' )
        meshfile_label.setAlignment( QtCore.Qt.AlignCenter )
        imp_layout.addWidget( meshfile_label, 1, 0 )
        self.meshfile_lineEdit = QtWidgets.QLineEdit()
        self.meshfile_lineEdit.setFixedWidth( 400 )
        self.meshfile_lineEdit.setFixedHeight( 32 )
        imp_layout.addWidget( self.meshfile_lineEdit, 1, 1 )
        meshfile_pushButton = QtWidgets.QPushButton( QtGui.QIcon(icons + os.sep + 'OpenFile.png'), '' )
        imp_layout.addWidget( meshfile_pushButton, 1, 2 )

        channel_label = QtWidgets.QLabel( 'Channel List' )
        channel_label.setAlignment( QtCore.Qt.AlignCenter )
        mlayout.addWidget( channel_label )

        # scroll area
        channelArea = QtWidgets.QScrollArea()
        channelArea.setMinimumHeight( 250 )
        channelArea.setWidgetResizable( True )
        channelArea.setEnabled( True )
        channelWidget = QtWidgets.QWidget( channelArea )
        # contextmenu
        channelWidget.setContextMenuPolicy( QtCore.Qt.CustomContextMenu )
        channelWidget.customContextMenuRequested.connect( self.channelCheck_Popup )
        # vertical layout
        channelVerticalLayout = QtWidgets.QVBoxLayout( channelWidget )
        # channel grid layout
        self.channelGridLayout = QtWidgets.QGridLayout()
        self.channelGridLayout.setContentsMargins( 5, 5, 5, 5 )
        self.channelGridLayout.setSpacing( 10 )
        channelVerticalLayout.addLayout( self.channelGridLayout )
        # vertical spacing
        spacer = QtWidgets.QSpacerItem(10,10,QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        channelVerticalLayout.addItem( spacer )

        channelArea.setWidget( channelWidget )
        mlayout.addWidget( channelArea )

        # import
        import_pushButton = QtWidgets.QPushButton( 'Import Scene' )
        import_pushButton.setFixedHeight( 40 )
        import_pushButton.clicked.connect( self.import_data )
        mlayout.addWidget( import_pushButton )

        # copyright
        font = QtGui.QFont()
        font.setPointSize( 8 )
        copyright = QtWidgets.QLabel( '@DexterStudios' )
        copyright.setAlignment( QtCore.Qt.AlignRight )
        copyright.setFont( font )
        mlayout.addWidget( copyright )

        self.show()

    def channelCheck_Popup( self ):
        menu = QtWidgets.QMenu()
        menu.addAction( 'Select All', self.channelCheck_selectAll )
        menu.addAction( 'Clear All' , self.channelCheck_clearAll )
        menu.exec_( QtWidgets.QCursor.pos() )

    def channelCheck_selectAll( self ):
        if not self.m_data:
            return
        for c in self.m_data['channels'].keys():
            widgetName = '%s_checkBox' % c.replace(' ', '_')
            exec( 'self.%s.setChecked( True )' % widgetName )

    def channelCheck_clearAll( self ):
        if not self.m_data:
            return
        for c in self.m_data['channels'].keys():
            widgetName = '%s_checkBox' % c.replace(' ', '_')
            exec( 'self.%s.setChecked( False )' % widgetName )

    def set_infofile( self ):
        filepath = QtWidgets.QFileDialog.getOpenFileName( None, 'Select File', '/show' )
        if filepath[0]:
            self.infofile_lineEdit.setText( filepath[0] )
            f = open( filepath[0], 'r' )
            data = json.load( f )
            f.close()
            self.m_data = data
            self.set_meshfile()
            self.add_channelWidget()

    def set_meshfile( self ):
        self.meshfile_lineEdit.setText( self.m_data['objects'][0] )

    def add_channelWidget( self ):
        channels = self.m_data['channels'].keys()
        channels.sort()
        for c in channels:
            layout_index = channels.index(c)
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
            exec( 'self.%s.setChecked( True )' % widgetName )
            exec( 'self.channelGridLayout.addWidget( self.%s, row, column )' % widgetName )

    def getChannelState( self ):
        result = dict()
        channels = self.m_data['channels'].keys()
        for c in channels:
            widgetName = '%s_checkBox' % c.replace(' ', '_')
            value = eval( 'self.%s.isChecked()' % widgetName )
            result[c] = value
        return result

    def import_data( self ):
        mari.projects.close()
        channelState = self.getChannelState()

        infofile = self.infofile_lineEdit.text()
        basename = os.path.basename(infofile).split('.')[0]
        projName = basename.replace( 'channels', 'rebuild' )
        mari.projects.create( projName, self.meshfile_lineEdit.text(), [] )
        geo = mari.geo.current()
        geo.removeChannel( geo.channel('diffuse') )
        # create channel
        for i in channelState:
            if channelState[i]:
                info = self.m_data['channels'][i]
                geo.createChannel( i, info['width'], info['height'], info['depth'] )
        self.close()
        # buildScene
        sceneDescription.buildScene( os.path.dirname(infofile), channelState )


def show_ui():
    global SceneImportWin
    SceneImportWin = SceneImportDialog()
