#encoding=utf-8
#--------------------------------------------------------------------------------
#
#    RenderMan TD
#
#        Sanghun Kim, rman.td@gmail.com
#
#    rman.td 2016.04.02 $2
#
#-------------------------------------------------------------------------------

import os, sys
import string
import glob

# from PySide import QtGui, QtCore
from pymodule.Qt import QtWidgets
from pymodule.Qt import QtGui
from pymodule.Qt import QtCore
import mari

import dxMari as dxm


class ProjectDialog( QtWidgets.QDialog ):
    def __init__( self, parent=None ):
        super( ProjectDialog, self ).__init__( parent )

        self.setWindowTitle( 'New Project' )
        self.resize( 700, 650 )
        # dialog main layout
        self.mlayout = QtWidgets.QVBoxLayout( self )

        # show, asset
        self.show_layout = QtWidgets.QHBoxLayout()
        self.mlayout.addLayout( self.show_layout )
        label_font = QtGui.QFont()
        label_font.setPointSize(12)
        label_font.setBold( True )
        combo_font = QtGui.QFont()
        combo_font.setPointSize(11)

        self.show_label        = QtWidgets.QLabel( ' Show : ' )
        self.show_label.setFont( label_font )
        self.show_layout.addWidget( self.show_label )
        self.show_comboBox    = QtWidgets.QComboBox()
        self.show_comboBox.setObjectName( 'show' )
        self.show_comboBox.setMinimumSize( QtCore.QSize(0,32) )
        self.show_comboBox.setFont( combo_font )
        self.show_layout.addWidget( self.show_comboBox )
        self.shot_label        = QtWidgets.QLabel( ' Shot : ' )
        self.shot_label.setFont( label_font )
        self.show_layout.addWidget( self.shot_label )
        self.shot_comboBox    = QtWidgets.QComboBox()
        self.shot_comboBox.setObjectName( 'shot' )
        self.shot_comboBox.setMinimumSize( QtCore.QSize(0,32) )
        self.shot_comboBox.setFont( combo_font )
        self.show_layout.addWidget( self.shot_comboBox )
        self.subdir_lineEdit= QtWidgets.QLineEdit()
        self.subdir_lineEdit.setFixedHeight( 32 )
        self.subdir_lineEdit.setFont( combo_font )
        self.show_layout.addWidget( self.subdir_lineEdit )
        # tree view
        self.dir_treeView = QtWidgets.QTreeView()
        self.dir_treeView.setFont( combo_font )
        self.mlayout.addWidget( self.dir_treeView )
        # set geometry
        self.setgeo_pushButton = QtWidgets.QPushButton(
                QtGui.QIcon(mari.resources.path('ICONS') + os.sep + 'Bottom.png'), ''
            )
        self.mlayout.addWidget( self.setgeo_pushButton )
        # project setup
        self.proj_layout = QtWidgets.QGridLayout()
        self.mlayout.addLayout( self.proj_layout )

        self.projName_label    = QtWidgets.QLabel( '  Name :' )
        self.projName_label.setAlignment( QtCore.Qt.AlignRight|QtCore.Qt.AlignVCenter )
        self.proj_layout.addWidget( self.projName_label, 0, 0 )
        self.projName_lineEdit = QtWidgets.QLineEdit()
        self.projName_lineEdit.setFixedHeight( 32 )
        self.proj_layout.addWidget( self.projName_lineEdit, 0, 1 )
        self.projGeo_label    = QtWidgets.QLabel( '  Geometry :' )
        self.projGeo_label.setAlignment( QtCore.Qt.AlignRight|QtCore.Qt.AlignVCenter )
        self.proj_layout.addWidget( self.projGeo_label, 1, 0 )
        self.projGeo_lineEdit = QtWidgets.QLineEdit()
        self.projGeo_lineEdit.setFixedHeight( 32 )
        self.proj_layout.addWidget( self.projGeo_lineEdit, 1, 1 )
        self.projGeo_pushButton = QtWidgets.QPushButton(
                QtGui.QIcon(mari.resources.path('ICONS') + os.sep + 'OpenFile.png'), ''
            )
        self.proj_layout.addWidget( self.projGeo_pushButton, 1, 2 )
        self.projRoot_label = QtWidgets.QLabel( '  Root Path :' )
        self.projRoot_label.setAlignment( QtCore.Qt.AlignRight|QtCore.Qt.AlignVCenter )
        self.proj_layout.addWidget( self.projRoot_label, 2, 0 )
        self.projRoot_lineEdit = QtWidgets.QLineEdit()
        self.projRoot_lineEdit.setFixedHeight( 32 )
        self.proj_layout.addWidget( self.projRoot_lineEdit, 2, 1 )
        self.projRoot_pushButton = QtWidgets.QPushButton(
                QtGui.QIcon(mari.resources.path('ICONS') + os.sep + 'OpenFile.png'), ''
            )
        self.proj_layout.addWidget( self.projRoot_pushButton, 2, 2 )
        self.setup_layout = QtWidgets.QHBoxLayout()
        self.mlayout.addLayout( self.setup_layout )

        self.ok_pushButton = QtWidgets.QPushButton( 'OK' )
        self.ok_pushButton.setFont( combo_font )
        self.ok_pushButton.setFixedHeight( 40 )
        self.setup_layout.addWidget( self.ok_pushButton )
        self.cancel_pushButton = QtWidgets.QPushButton( 'Cancel' )
        self.cancel_pushButton.setFont( combo_font )
        self.cancel_pushButton.setFixedHeight( 40 )
        self.setup_layout.addWidget( self.cancel_pushButton )
        self.cancel_pushButton.clicked.connect( self.close )
        # command binding
        self.setgeo_pushButton.clicked.connect( self.setProject_proc )
        self.projGeo_pushButton.clicked.connect( self.setProjectGeo_file )
        self.projRoot_pushButton.clicked.connect( self.setProjectRoot_path )
        self.ok_pushButton.clicked.connect( self.setProject_mari )
        # copyright
        font = QtGui.QFont()
        font.setPointSize(8)
        copyright = QtWidgets.QLabel( '@DexterDigital Texture' )
        copyright.setAlignment( QtCore.Qt.AlignRight )
        copyright.setFont( font )
        self.mlayout.addWidget( copyright )

        self.setupUi()

        self.show()

    def setupUi( self ):
        # show comboBox
        show_list = []
        showdir = '/show'
        showdir = dxm.dirReMap( showdir )
        for i in os.listdir( showdir ):
            if os.path.isdir( os.path.join(showdir, i) ):
                show_list.append( i )
        show_list.sort()
        self.show_comboBox.addItems( show_list )
        # init show
        startpath = mari.resources.path('MARI_DEFAULT_GEOMETRY_PATH')
        if startpath:
            source = startpath.split( os.sep )
            if 'show' in source:
                find_index = source.index('show') + 1
                if source[find_index] in show_list:
                    self.show_comboBox.setCurrentIndex( show_list.index(source[find_index]) )
            #self.show_comboBox.setCurrentIndex( show_list.index(source[2]) )

        self.fileModel = QtWidgets.QFileSystemModel()
        self.add_shotItems()

        # subdir line edit
        self.subdirFilterModel = QtCore.QSortFilterProxyModel()
        self.dirCompleter = QtWidgets.QCompleter()
        self.dirCompleter.setModel( self.subdirFilterModel )
        self.dirCompleter.setCompletionMode( QtWidgets.QCompleter.UnfilteredPopupCompletion )
        self.dirCompleter.setCaseSensitivity( QtCore.Qt.CaseInsensitive )
        self.dirCompleter.setMaxVisibleItems( 10 )
        self.subdir_lineEdit.setCompleter( self.dirCompleter )
        self.subdir_lineEdit.textEdited.connect( self.subdirFilterModel.setFilterFixedString )
        self.subdir_lineEdit.editingFinished.connect( self.changeTreeView_proc )
        self.subdirCompleter()

        # file tree view
        self.dir_treeView.setModel( self.fileModel )
        self.dir_treeView.hideColumn(1)
        self.dir_treeView.hideColumn(2)
        self.dir_treeView.setColumnWidth( 0, 500 )
        self.changeTreeView_proc()

        # command binding
        self.show_comboBox.activated.connect( self.changeComboBox_proc )
        self.shot_comboBox.activated.connect( self.changeComboBox_proc )

    def add_shotItems( self ):
        current_show = self.show_comboBox.currentText()
        showdir = string.join( ['', 'show', current_show], '/' )
        showdir = dxm.dirReMap( showdir )
        shot_list = []
        for i in os.listdir( showdir ):
            if os.path.isdir( os.path.join(showdir, i) ):
                shot_list.append( i )
        shot_list.sort()
        self.shot_comboBox.clear()
        self.shot_comboBox.addItems( shot_list )

    def subdirCompleter( self ):
        dirRoot = os.path.join( '/show',
                                self.show_comboBox.currentText(),
                                self.shot_comboBox.currentText() )
        dirRoot = dxm.dirReMap( dirRoot )
        subModel = QtGui.QStandardItemModel()
        i = 0
        for d in glob.glob( '%s/*/*' % dirRoot ):
            if os.path.isdir( d ):
                dir  = dxm.dirReMap( d )
                item = QtGui.QStandardItem( dir.split(dirRoot+'/')[-1] )
                font = QtGui.QFont()
                font.setPointSize( 11 )
                item.setFont( font )
                subModel.setItem( i, item )
                i += 1
        self.subdirFilterModel.setFilterCaseSensitivity( QtCore.Qt.CaseInsensitive )
        self.subdirFilterModel.setSourceModel( subModel )

    def changeComboBox_proc( self ):
        sender = self.sender()
        if sender.objectName() == 'show':
            self.add_shotItems()
        self.subdir_lineEdit.setText('')
        self.changeTreeView_proc()
        self.subdirCompleter()

    def changeTreeView_proc( self ):
        show    = self.show_comboBox.currentText()
        shot    = self.shot_comboBox.currentText()
        subdir    = self.subdir_lineEdit.text()
        dirRoot = os.path.join( '/show', show, shot, subdir )
        dirRoot = dxm.dirReMap( dirRoot )
        if os.path.exists( dirRoot ):
            self.fileModel.setRootPath( dirRoot )
            self.dir_treeView.setRootIndex( self.fileModel.index(dirRoot) )

    def setProject_proc( self ):
        importFormat = ['abc', 'obj']
        current = self.fileModel.filePath( self.dir_treeView.currentIndex() )
        source = current.split('/')
        # Name
        if 'asset' in source:
            asset_index = source.index('asset')
            asset_name  = source[asset_index+2]
            self.projName_lineEdit.setText( asset_name )
            # Root
            asset_path = string.join( source[:asset_index+3] + ['texture'], '/' )
            asset_path = dxm.dirReMap( asset_path )
            self.projRoot_lineEdit.setText( asset_path )
        # geo file
        self.projGeo_lineEdit.setText( current )

    def setProject_mari( self ):
        projname = self.projName_lineEdit.text()
        geofile  = self.projGeo_lineEdit.text()
        mari.projects.close()
        mari.projects.create( projname, geofile,
                [
                    mari.ChannelInfo( 'diffC', 8192, 8192, mari.Image.DEPTH_BYTE, False, mari.Color(.5, .5, .5, 0) )
                ]
            )

        # resource path
        mari.resources.setPath( 'MARI_DEFAULT_ARCHIVE_PATH', self.projRoot_lineEdit.text() )
        mari.resources.setPath(
                'MARI_DEFAULT_CAMERA_PATH',
                os.path.dirname( self.projGeo_lineEdit.text() )
            )
        mari.resources.setPath( 'MARI_DEFAULT_EXPORT_PATH', self.projRoot_lineEdit.text() )
        mari.resources.setPath(
                'MARI_DEFAULT_GEOMETRY_PATH',
                os.path.dirname( self.projGeo_lineEdit.text() )
            )
        mari.resources.setPath( 'MARI_DEFAULT_IMAGE_PATH', self.projRoot_lineEdit.text() )
        mari.resources.setPath( 'MARI_DEFAULT_IMPORT_PATH', self.projRoot_lineEdit.text() )

        # texture info json
        source = os.path.splitext( geofile )
        infofn = geofile.replace( source[-1], '.json' )
        if os.path.exists( infofn ):
            txinfo = dxm.readTxLayer_byFile( infofn )
            if txinfo:
                dxm.geoMetadata_txname_update( txinfo )

        # OutPath metadata
        dxm.geoMetadata_outpath_update( os.path.dirname(self.projRoot_lineEdit.text()) )

        self.close()

    def setProjectGeo_file( self ):
        current  = self.projGeo_lineEdit.text()
        startDir = None
        if current:
            startDir = os.path.dirname( current )
        fn = QtWidgets.QFileDialog.getOpenFileName( None, 'Select Object File', startDir, '*.obj *.abc' )
        if fn:
            self.projGeo_lineEdit.setText( dxm.dirReMap(fn[0]) )

    def setProjectRoot_path( self ):
        current  = self.projRoot_lineEdit.text()
        startDir = None
        if current:
            startDir = current
        path = QtWidgets.QFileDialog.getExistingDirectory( None, 'Project Root', startDir )
        path = path.replace( '\\', '/' ) # to linux path
        if path:
            self.projRoot_lineEdit.setText( dxm.dirReMap(path) )

def show_ui():
    global ProjectDialogWin
    ProjectDialogWin = ProjectDialog()