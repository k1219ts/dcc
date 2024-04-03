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

class SceneExportDialog( QtWidgets.QDialog ):
    def __init__( self, parent=None ):
        super( SceneExportDialog, self ).__init__( parent )
        self.setWindowTitle( 'Mari Scene Export' )

        if mari.projects.current() is None:
            mari.utils.message( 'Open Project' )
            return

        # Mari Scene
        self.channels = mari.geo.current().channelList()

        # ETC
        icons = mari.resources.path('ICONS')

        # dialog main layerout
        mlayout = QtWidgets.QVBoxLayout( self )

        # channel info file
        imp_layout = QtWidgets.QGridLayout()
        mlayout.addLayout( imp_layout )

        exppath_label = QtWidgets.QLabel( 'Export Path' )
        imp_layout.addWidget( exppath_label, 0, 0 )
        self.exppath_lineEdit = QtWidgets.QLineEdit()
        self.exppath_lineEdit.setFixedWidth( 400 )
        self.exppath_lineEdit.setFixedHeight( 32 )
        imp_layout.addWidget( self.exppath_lineEdit, 0, 1 )
        exppath_pushButton = QtWidgets.QPushButton( QtGui.QIcon(icons + os.sep + 'OpenFile.png'), '' )
        exppath_pushButton.clicked.connect( self.set_export_dir )
        imp_layout.addWidget( exppath_pushButton, 0, 2 )

        # scroll area
        channelArea = QtWidgets.QScrollArea()
        channelArea.setMinimumHeight( 250 )
        channelArea.setWidgetResizable( True )
        channelArea.setEnabled( True )
        channelWidget = QtWidgets.QWidget( channelArea )
        # contextmenu
        channelWidget.setContextMenuPolicy( QtCore.Qt.CustomContextMenu )
        channelWidget.customContextMenuRequested.connect( self.channelCheck_Popup )
        # channel vertical layout
        channelVerticalLayout = QtWidgets.QVBoxLayout( channelWidget )
        # channel grid layout
        self.channelGridLayout = QtWidgets.QGridLayout()
        self.channelGridLayout.setContentsMargins( 5, 5, 5, 5 )
        self.channelGridLayout.setSpacing( 10 )
        # add channels
        self.add_channelWidget()
        channelVerticalLayout.addLayout( self.channelGridLayout )
        # vertical spacing
        spacer = QtWidgets.QSpacerItem(10,10,QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        channelVerticalLayout.addItem( spacer )

        channelArea.setWidget( channelWidget )
        mlayout.addWidget( channelArea )

        # export
        export_pushButton = QtWidgets.QPushButton( 'Export Scene' )
        export_pushButton.setFixedHeight( 40 )
        export_pushButton.clicked.connect( self.export_data )
        mlayout.addWidget( export_pushButton )

        # copyright
        font = QtGui.QFont()
        font.setPointSize( 8 )
        copyright = QtWidgets.QLabel( '@DexterStudios' )
        copyright.setAlignment( QtCore.Qt.AlignRight )
        copyright.setFont( font )
        mlayout.addWidget( copyright )

        self.show()

    def add_channelWidget( self ):
        nameList = list()
        for c in self.channels:
            nameList.append( c.name() )
        nameList.sort()
        for c in nameList:
            layout_index = nameList.index(c)
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
            exec( 'self.%s.setObjectName( "%s" )' % (widgetName, c) )
            exec( 'self.%s.setChecked( True )' % widgetName )
            exec( 'self.channelGridLayout.addWidget( self.%s, row, column )' % widgetName )

    def channelCheck_Popup( self ):
        menu = QtWidgets.QMenu()
        menu.addAction( 'Select All', self.channelCheck_selectAll )
        menu.addAction( 'Clear All' , self.channelCheck_clearAll )
        menu.exec_( QtWidgets.QCursor.pos() )

    def channelCheck_selectAll( self ):
        for c in self.channels:
            widgetName = '%s_checkBox' % c.name().replace(' ', '_')
            exec( 'self.%s.setChecked( True )' % widgetName )

    def channelCheck_clearAll( self ):
        for c in self.channels:
            widgetName = '%s_checkBox' % c.name().replace(' ', '_')
            exec( 'self.%s.setChecked( False )' % widgetName )

    def set_export_dir( self ):
        dirpath = QtWidgets.QFileDialog.getExistingDirectory( None, 'Export Path', '/show' )
        if dirpath:
            self.exppath_lineEdit.setText( str(dirpath) )

    def getChannelState( self ):
        result = dict()
        for c in self.channels:
            widgetName = '%s_checkBox' % c.name().replace(' ', '_')
            value = eval( 'self.%s.isChecked()' % widgetName )
            result[c.name()] = value
        return result

    def export_data( self ):
        channelState = self.getChannelState()
        exportPath   = self.exppath_lineEdit.text()
        if exportPath:
            sceneDescription.exportScene( exportPath, channelState )
            self.close()
            mari.utils.message( 'Export Complete!' )


def show_ui():
    global SceneExportWin
    SceneExportWin = SceneExportDialog()
