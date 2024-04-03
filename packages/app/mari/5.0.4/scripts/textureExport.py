#encoding=utf-8
#--------------------------------------------------------------------------------
#
#    Dexter RnD
#
#        daeseok.chae, cds7031@gmail.com
#
#    daeseok.chae 2018.12.31
#
#-------------------------------------------------------------------------------

import os, sys
import string
import json
import shutil
import glob
import mari


try:
    from PIL import Image
except:
    print('# Result : Not support PILLOW module.')

# from pymodule.Qt import QtWidgets, QtCore, QtGui
from PySide2 import QtWidgets, QtCore, QtGui
import dxMari as dxm
import txmake_proc
from textureExportUI import Ui_Form
import textureCopy

REZ_LIST = ['512', '1k', '2k', '4k', '8k', '16k', '32k']
REZ_MAP = {512:'512', 1024:'1k', 2048:'2k', 4096:'4k', 8192:'8k', 16384:'16k', 32768:'32k'}
REZ_INV_MAP = {'512':512, '1k':1024, '2k':2048, '4k':4096, '8k':8192, '16k':16384, '32k':32768}
SIZE_MAP = {
            512: mari.ImageSet.SIZE_512, 1024: mari.ImageSet.SIZE_1024, 2048: mari.ImageSet.SIZE_2048,
            4096: mari.ImageSet.SIZE_4096, 8192: mari.ImageSet.SIZE_8192,
            16384: mari.ImageSet.SIZE_16384, 32768: mari.ImageSet.SIZE_32768
        }
COLOR_MAP = ['(230,23,23)', '(230,92,23)', '(230,126,23)', '(230,161,23)', '(230,195,23)',
             '(230,230,23)', '(195,230,23)', '(161,230,23)', '(92,230,23)', '(23,230,126)',
             '(23,230,195)', '(23,195,230)']

DISMAP_TYPE = ['disI', 'disF']

g_export_cancelled = False

# 2017.04.19 by daeseok.chae
from dxname import rulebook
from dxname import tag_parser
import pymongo
from pymongo import MongoClient
import datetime
import getpass
from dxConfig import dxConfig

import DXUSD.Utils as utl
import DXRulebook.Interface as rb

DBIP = dxConfig.getConf("DB_IP")
DBNAME = "PIPE_PUB"

def getPubVersion(show, task, data_type, asset_name = "", shot = ""):
    client = MongoClient(DBIP)
    db = client[DBNAME]
    coll = db[show]

    print(show, task, data_type, asset_name)
    record = {'show': show,
              'task': task,
              'data_type': data_type}

    if asset_name != "":
        record['asset_name'] = asset_name

    if shot != "":
        record['shot'] = shot

    recentDoc = coll.find_one(record,
                              sort=[('version', pymongo.DESCENDING)])
    if recentDoc:
        return recentDoc['version'] + 1
    else:
        return 1

def readPathRule(showDir):
    showPath = os.path.dirname(showDir)
    print('showPath:',showPath)
    pathRuleFile = '{DIR}/_config/maya/pathRule.json'.format(DIR=showDir)
    print('pathRuleFile:',pathRuleFile)
    if os.path.exists(pathRuleFile):
        try:
            ruleData = json.load(open(pathRuleFile))
            if ruleData.get('showDir') and ruleData['showDir']:
                _showDir = ruleData['showDir']
                if os.path.isabs(_showDir):
                    showDir = _showDir
                else:
                    showDir = os.path.join(showPath, _showDir)
        except:
            pass
    return showDir
#-------------------------------------------------------------------------------
#    Core
class ExportChannel():
    def __init__( self, outpath ):
        self.outpath = outpath
        if not os.path.exists( self.outpath ):
            os.makedirs( self.outpath )
        self.cObject = mari.geo.current()
        self.cShader = self.cObject.currentShader()
        self.cChannel = self.cObject.currentChannel()
        self.exportChannels = list()
        self.rezMap = dict()
        self.TextureNameInfo = None
        self.doTxMake = None
        self.doProxy  = None
        self.doCopy = None

    def doIt( self ):
        ProgressDialog.instance = ProgressDialog()
        self.Progress = ProgressDialog.instance
        self.Progress.show()
        self.Progress.progress_text.setText( 'Baking Shader ...' )

        self.undosize = 0
        self.message = []

        # resize
        if self.rezMap:
            self.resizePatches()

        selected = self.cObject.selectedPatches()
        if selected:
            self.exportSelectedPatches( selected )
        else:
            self.exportAllPatches()

        if self.undosize:
            for i in range(self.undosize):
                mari.history.undo()

        # exported files
        expfiles = list( self.message )

        if self.doProxy:
            self.proxy_process()

        if self.doTxMake:
            self.txmake_process()

        # version check copy
        if self.doCopy:
            textureCopy.versionCheckCopy( expfiles )

        self.Progress.close()
        if self.message:
            mari.utils.message( '\n'.join(self.message) )

    # resize
    def resizePatches( self ):
        for i in self.cObject.patchList():
            uv_index = i.uvIndex()
            if uv_index in self.rezMap.keys():
                for c in self.exportChannels:
                    doresize = dict()
                    if c.width(uv_index) != self.rezMap[uv_index]:
                        if self.rezMap[uv_index] in doresize.keys():
                            doresize[self.rezMap[uv_index]].append( uv_index )
                        else:
                            doresize[self.rezMap[uv_index]] = [uv_index]
                    for r in doresize:
                        c.resize( SIZE_MAP[r], doresize[r] )
                        self.undosize += 1

    # all
    def exportAllPatches( self ):
        self.Progress.progress_text.setText( 'Exporting : all patches ...' )
        self.Progress.pbar.setValue( 20 )
        mari.app.processEvents()

        for c in self.exportChannels:
            self.Progress.progress_text.setText( '"%s" exporting : all patches ...' % c.name() )
            mari.app.processEvents()

            extension = 'jpg'

            if c.name() in DISMAP_TYPE:
                extension = 'tif'

            # print(os.path.join( self.outpath, '$CHANNEL.$UDIM.jpg' ))
            c.exportImagesFlattened(
                    os.path.join( self.outpath, '$CHANNEL.$UDIM.%s' % extension ),
                    options = 0, file_options = None )
            # rename
            self.exportedFileRename( self.cObject.patchList(), c, extension )

    def exportSelectedPatches( self, patches ):
        self.Progress.progress_text.setText( 'Exporting : selected patches ...' )
        self.Progress.pbar.setValue( 20 )
        mari.app.processEvents()
        for c in self.exportChannels:
            self.Progress.progress_text.setText( '"%s" exporting : selected pathces ...' % c.name() )
            mari.app.processEvents()

            extension = 'jpg'

            if c.name() in DISMAP_TYPE:
                extension = 'tif'

            c.exportSelectedPatchesFlattened(
                    os.path.join( self.outpath, '$CHANNEL.$UDIM.%s' % extension ),
                    options = 0, file_options = None )
            # rename
            self.exportedFileRename( patches, c, extension )

    def exportedFileRename( self, patches, channel, extension ):

        for i in patches:
            uvindex = i.uvIndex()
            orig    = os.path.join( self.outpath, '%s.%s.%s' % (channel.name(), i.name(), extension) )
            # print('>>>>>>>>>>>>>>>>>exportedFileRename:orig', orig)
            # print('i.name():',i.name())

            if self.TextureNameInfo[uvindex]:
                filename = '%s_%s.%s' % ( self.TextureNameInfo[uvindex], channel.name(), extension )
                if self.TextureNameInfo.get('multiuv'):
                    if self.TextureNameInfo['multiuv'].get(uvindex):
                        udim = self.TextureNameInfo['multiuv'][uvindex]
                        filename = '%s_%s.%s.%s' % (self.TextureNameInfo[uvindex], channel.name(), udim, extension)
                else:
                    udim = 1001 + uvindex
                    filename = '%s_%s.%s.%s' % (self.TextureNameInfo[uvindex], channel.name(), udim, extension)

                print('>>>>>>>>filename:',filename)
                new = os.path.join( self.outpath, filename )
                #os.rename( orig, new )
                shutil.move( orig, new )
                self.message.append( dxm.dirReMap(new) )
            else:
                self.message.append( dxm.dirReMap(orig) )

    def proxy_process( self ):
        if not globals().get( 'Image' ):
            print("----- not has key 'Image' -----")
            return
        if not self.message:
            print("----- not message -----")
            return

        print("self.message :", self.message)

        # proxy path
        Path = os.path.dirname( self.message[0] )
        proxyPath = os.path.join( os.path.dirname(os.path.dirname(Path)), 'proxy', os.path.basename(Path) )
        if not os.path.exists( proxyPath ):
            os.makedirs( proxyPath )

        proxyFiles = list()
        for i in self.message:
            if i.find('diffC') > -1:
                fileExt   = os.path.splitext( i )
                proxyFile = os.path.join( proxyPath, os.path.basename(i).replace(fileExt[-1], '.jpg') )
                readImage = Image.open( i )
                readImage.thumbnail( (512,512), Image.ANTIALIAS )
                readImage.save( proxyFile, 'JPEG' )
                print('# Debug : texture proxy : %s' % proxyFile)
                proxyFiles.append( proxyFile )

        if proxyFiles:
            textureCopy.versionCheckCopy( proxyFiles )

    def txmake_process( self ):
        self.Progress.progress_text.setText( 'txmake : ...' )
        self.Progress.pbar.setValue( 60 )
        mari.app.processEvents()
        txmake = txmake_proc.TxMake( self.message )
        txmake.Events = self.Progress
        log = txmake.convert()
        self.message += log

class ProgressDialog( QtWidgets.QDialog ):
    instance = None
    def __init__( self ):
        super( ProgressDialog, self ).__init__()
        self.setWindowTitle( 'Export Texture Progress' )
        self.resize( 400, 50 )

        self.v_layout = QtWidgets.QVBoxLayout()
        self.setLayout( self.v_layout )

        self.pbar = QtWidgets.QProgressBar( self )
        self.progress_text = QtWidgets.QLabel( self )
        self.progress_text.setText( 'Preparing to export ...' )
        self.pbar.setValue(0)

        self.v_layout.addWidget( self.pbar )
        self.v_layout.addWidget( self.progress_text )
        #self.v_layout.addWidget( self.cancel_button )

    def cancel( self ):
        global g_export_cancelled
        g_export_cancelled = True


#-------------------------------------------------------------------------------
#    UI
class ExporterDialog( QtWidgets.QDialog ):
    def __init__( self, parent=None ):
        super( ExporterDialog, self ).__init__( parent )

        if mari.projects.current() is None:
            mari.utils.message( 'Please open a project befor "Set LayerName".' )
            return

        self.geo = mari.geo.current()
        self.channel = self.geo.currentChannel()
        self.selected = []
        for i in self.geo.selectedPatches():
            self.selected.append( i.uvIndex() )

        # Texture Name
        self.TextureNameInfo = dxm.getMetadata_textureNameInfo()

        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.setWindowTitle( 'USD Texture Export' )

        showDir = ""
        assetName = ""
        version = "v001"
        if not rb.MatchFlag('ver', version):
            version = version.upper()
            if not rb.MatchFlag('ver', version):
                raise 'check rulebook ver!!!'

        projectPath = mari.resources.path('MARI_DEFAULT_IMAGE_PATH')
        if 'OutPath' in self.geo.metadataNames():
            projectPath = str(self.geo.metadata('OutPath'))
        splitPath = projectPath.split("/")
        if "show" in splitPath:
            orgShowDir = "/".join(splitPath[:splitPath.index("show") + 2])
            print('orgShowDir :', orgShowDir) #/show/slc)
            # showDir = readPathRule(orgShowDir)+'/_3d'
            showDir =   "/".join(splitPath[:splitPath.index("show") + 3])
            print('showDir:',showDir) #/show/slc/_3d)

            if orgShowDir != showDir:  # configSetup and projectSetup equal.

                if not "_pub" in orgShowDir:  # original file convention. /show/$SHOW/_3d/asset/type/$ASSETNAME/
                    assetName = splitPath[splitPath.index("show") + 4]

                    versionDir = os.path.join(showDir, "asset", assetName)
                    print('assetName:', assetName)
                    print('versionDir:', versionDir)

                    if "branch" in splitPath:
                        branchName = splitPath[splitPath.index("branch") + 1]
                        self.ui.branchCheckBox.setChecked(True)
                        self.ui.branchEdit.setText(branchName)
                        versionDir = os.path.join(versionDir, "branch", branchName)
                    versionDir = os.path.join(versionDir, "texture", "images")
                    version = utl.GetNextVersion(versionDir)
                    print('versionDir:', versionDir)

                else:  # maybe _pub? /show/$SHOW_pub/asset/$assetName
                    assetName = splitPath[splitPath.index("asset") + 1]
                    versionDir = os.path.join(showDir, "asset", assetName)

                    if "branch" in splitPath:
                        branchName = splitPath[splitPath.index("branch") + 1]
                        self.ui.branchCheckBox.setChecked(True)
                        self.ui.branchEdit.setText(branchName)
                        versionDir = os.path.join(versionDir, "branch", branchName)
                    versionDir = os.path.join(versionDir, "texture", "images")
                    version = utl.GetNextVersion(versionDir)
            else:
                assetName = splitPath[splitPath.index("asset") + 1]
                versionDir = os.path.join(showDir, "asset", assetName)

                if "branch" in splitPath:
                    branchName = splitPath[splitPath.index("branch") + 1]
                    self.ui.branchCheckBox.setChecked(True)
                    self.ui.branchEdit.setText(branchName)
                    versionDir = os.path.join(versionDir, "branch", branchName)
                versionDir = os.path.join(versionDir, "texture", "images")
                version = utl.GetNextVersion(versionDir)

        self.ui.showLabel.setFixedWidth( 60 )
        self.ui.showLabel.setAlignment( QtCore.Qt.AlignRight|QtCore.Qt.AlignVCenter )
        self.ui.showEdit.setFixedWidth( 120 )
        self.ui.showEdit.setFixedHeight( 32 )
        self.ui.showEdit.textChanged.connect(self.showChange)
        self.ui.showEdit.setText( showDir )
        self.ui.shotLabel.setVisible(self.ui.shotCheckBox.isChecked())
        self.ui.shotEdit.setVisible(self.ui.shotCheckBox.isChecked())
        self.ui.shotCheckBox.stateChanged.connect(self.shotCheckedClick)
        self.ui.shotEdit.textChanged.connect(self.shotChanged)
        self.ui.branchLabel.setVisible(self.ui.branchCheckBox.isChecked())
        self.ui.branchEdit.setVisible(self.ui.branchCheckBox.isChecked())
        self.ui.branchCheckBox.stateChanged.connect(self.branchCheckedClick)
        self.ui.branchEdit.textChanged.connect(self.branchChange)
        self.ui.assetNameLabel.setFixedWidth(80)
        self.ui.assetNameEdit.setFixedWidth(160)
        self.ui.assetNameEdit.setFixedHeight(32)



        if not showDir == "" :
            if not os.path.exists(showDir):
                assetList = []
                for i in os.listdir("%s/asset" % showDir):
                    if not i.startswith(".") and "asset.usd" != i:
                        assetList.append(i)
                assetCompleter = QtWidgets.QCompleter(assetList)
                self.ui.assetNameEdit.setCompleter(assetCompleter)
                self.ui.assetNameEdit.textChanged.connect(self.assetNameChange)
                self.ui.assetNameEdit.setText(assetName)

            else:
                print('assetName:', assetName)
                print('versionDir:', versionDir)
                self.ui.assetNameEdit.textChanged.connect(self.assetNameChange)
                self.ui.assetNameEdit.setText(assetName)


        else:
            self.ui.assetNameEdit.textChanged.connect(self.assetNameChange)
            self.ui.assetNameEdit.setText(assetName)

        self.ui.textureVersionEdit.textChanged.connect(self.versionChange)
        self.ui.textureVersionEdit.setText(version)


        # #    channels name check
        for i in self.geo.channelList():
            if len(i.name().split()) > 1:
                mari.utils.message( '"%s" channel name error.\n' % i.name() )
                return
        #    channels
        self.channelWidget()
        self.ui.uvpatchLayout.setContentsMargins( 5, 5, 5, 5 )
        self.ui.uvpatchLayout.setVerticalSpacing( 10 )
        self.patchWidget()
        self.ui.sizeComboBox.currentIndexChanged.connect(self.allSizeChange)
        self.ui.saverezBtn.clicked.connect( self.texture_resize )
        self.ui.versionCopyCheckBox.setFixedWidth( 120 )
        self.ui.exportBtn.setIcon(QtGui.QIcon(mari.resources.path('ICONS') + os.sep + 'ExportFile.png'))
        self.ui.exportBtn.clicked.connect( self.texture_export )

        self.show()

    def allSizeChange(self):
        print(self.ui.sizeComboBox.currentText())
        self.patchAll()


    # flexible ui setup
    def channelWidget( self ):
        label = QtWidgets.QLabel( ' Channels :' )
        label.setFixedWidth( 70 )
        self.ui.channelLayout.addWidget( label, 0, 0 )
        channels = self.geo.channelList()
        for i in channels:
            index = channels.index(i)
            widgetName = 'channel_%s' % i.name()
            exec( 'self.%s = QtWidgets.QCheckBox("%s")' % (widgetName, i.name()) )
            exec( 'self.ui.channelLayout.addWidget( self.%s, %s, %s )' % (widgetName, index/len(channels), index+1))
            if self.channel.name() == i.name():
                exec( 'self.%s.setChecked( True )' % widgetName )
            exec( 'self.%s.setContextMenuPolicy( QtCore.Qt.CustomContextMenu )' % widgetName )
            exec( 'self.%s.customContextMenuRequested.connect( self.channelSelectPopup )' % widgetName )


    def patchWidget( self ):

        for i in self.geo.patchList():
            # label
            uv_index= i.uvIndex()
            uv_name = i.name()
            uvlabel = QtWidgets.QLabel( 'uv %s' % uv_name )
            uvlabel.setAlignment( QtCore.Qt.AlignRight|QtCore.Qt.AlignVCenter )
            uvlabel.setFixedWidth( 63 )
            if 'multiuv' in self.TextureNameInfo.keys():
            # if len(self.geo.patchList()) > 1:
                if uv_index in self.TextureNameInfo['multiuv'].keys():
                    uvlabel.setStyleSheet( 'QLabel {color: rgb%s}' % COLOR_MAP[uv_index/10] )
            self.ui.uvpatchLayout.addWidget( uvlabel, uv_index, 0 )

            # line edit
            widgetName = 'txname_edit_%s' % uv_index
            exec( 'self.%s = QtWidgets.QLineEdit()' % widgetName )
            exec( 'self.%s.setObjectName( "%s" )' % (widgetName, widgetName) )
            # exec( 'self.%s.setFixedWidth( 450 )' % widgetName )
            if uv_index in self.TextureNameInfo.keys():
                exec( 'self.%s.setText( "%s" )' % (widgetName, self.TextureNameInfo[uv_index]) )
            exec( 'self.%s.editingFinished.connect( self.layername_update )' % widgetName )
            exec( 'self.ui.uvpatchLayout.addWidget( self.%s, %s, 1 )' % (widgetName, uv_index) )
            # disable
            if self.selected:
                if not uv_index in self.selected:
                    exec( 'self.%s.setDisabled( True )' % widgetName )

            # combobox
            widgetName = 'txname_rez_%s' % uv_index
            exec( 'self.%s = QtWidgets.QComboBox()' % widgetName )
            exec( 'self.%s.addItems( REZ_LIST )' % widgetName )
            exec( 'self.ui.uvpatchLayout.addWidget( self.%s, %s, 2 )' % (widgetName, uv_index) )
            try:
                currentSize = self.channel.width( uv_index )
            except:
                currentSize = 4096
            setIndex = REZ_LIST.index( REZ_MAP[currentSize] )
            exec( 'self.%s.setCurrentIndex( %s )' % (widgetName, setIndex) )
            print('}}}}}}}}}}}}}}}}}}}}}}}}}}}widgetName,uv_index,setIndex',widgetName, uv_index,setIndex)
            # disable
            if self.selected:
                if not uv_index in self.selected:
                    exec( 'self.%s.setDisabled( True )' % widgetName )

    def patchAll( self ):
        size = self.ui.sizeComboBox.currentText()
        currentSize = REZ_INV_MAP[size]
        for i in self.geo.patchList():
            uv_index= i.uvIndex()
            widgetName = 'txname_rez_%s' % uv_index
            setIndex = REZ_LIST.index( REZ_MAP[currentSize] )
            exec( 'self.%s.setCurrentIndex( %s )' % (widgetName, setIndex) )

    # slot setup
    def showChange(self):
        print("showChange")
        rootPath = os.path.join(self.ui.showEdit.text(), "asset")
        if not os.path.exists(rootPath):
            return
        assetList = []
        for i in os.listdir(rootPath):
            if not i.startswith(".") and "asset.usd" != i:
                assetList.append(i)
        assetCompleter = QtWidgets.QCompleter(assetList)
        self.ui.assetNameEdit.setCompleter(assetCompleter)
        self.assetNameChange()
        self.branchChange()
        self.versionChange()
        self.calcOutDir()

    def shotChanged(self):
        if not self.ui.shotCheckBox.isChecked():
            return
        print("shotChanged")
        self.assetNameChange()
        self.versionChange()
        self.calcOutDir()

    def shotCheckedClick(self, state):
        self.ui.shotLabel.setVisible(self.ui.shotCheckBox.isChecked())
        self.ui.shotEdit.setVisible(self.ui.shotCheckBox.isChecked())
        self.calcOutDir()

    def branchCheckedClick(self, state):
        self.ui.branchLabel.setVisible(self.ui.branchCheckBox.isChecked())
        self.ui.branchEdit.setVisible(self.ui.branchCheckBox.isChecked())
        self.calcOutDir()

    def assetNameChange(self):
        print("assetNameChange")
        showDir = str(self.ui.showEdit.text())
        assetName = str(self.ui.assetNameEdit.text())
        rootPath = showDir
        if self.ui.shotCheckBox.isChecked():
            seq = self.ui.shotEdit.text().split("_")[0]
            rootPath = os.path.join(showDir, "shot", seq, self.ui.shotEdit.text())
        assetPath = os.path.join(rootPath, "asset", assetName)
        modelVerDir = os.path.join(assetPath, "model")
        textureVerDir =  os.path.join(assetPath, "texture", "images")
        self.ui.dataVersionEdit.setText(utl.GetLastVersion(modelVerDir))
        self.ui.textureVersionEdit.setText(utl.GetNextVersion(textureVerDir))

        self.versionChange()
        self.calcOutDir()



    def branchChange(self):
        if not self.ui.branchCheckBox.isChecked():
            return
        print("branchChange")

        showDir = str(self.ui.showEdit.text())
        rootPath = showDir
        if self.ui.shotCheckBox.isChecked():
            seq = self.ui.shotEdit.text().split("_")[0]
            rootPath = os.path.join(showDir, "shot", seq, self.ui.shotEdit.text())
        assetName = str(self.ui.assetNameEdit.text())
        branchName = str(self.ui.branchEdit.text())
        assetPath = os.path.join(rootPath, "asset", assetName, "branch", branchName)
        modelVerDir = os.path.join(assetPath, "model")
        textureVerDir = os.path.join(assetPath, "texture", "images")
        self.ui.dataVersionEdit.setText(utl.GetLastVersion(modelVerDir))
        self.ui.textureVersionEdit.setText(utl.GetNextVersion(textureVerDir))

        self.versionChange()
        self.calcOutDir()

    def versionChange(self):
        print("versionChange")
        availableColor = QtGui.QColor(QtCore.Qt.green)
        unavailableColor = QtGui.QColor(QtCore.Qt.red)
        showDir = str(self.ui.showEdit.text())
        assetName = str(self.ui.assetNameEdit.text())
        rootPath = showDir
        if self.ui.shotCheckBox.isChecked():
            try:
                seq = self.ui.shotEdit.text().split("_")[0]
            except:
                seq = ""
            rootPath = os.path.join(rootPath, "shot", seq, str(self.ui.shotEdit.text()))
        rootPath = os.path.join(rootPath, "asset", assetName)
        if self.ui.branchCheckBox.isChecked():
            rootPath = os.path.join(rootPath, "branch", str(self.ui.branchEdit.text()))

        version = str(self.ui.textureVersionEdit.text())
        versionDir = os.path.join(rootPath, "texture", "images", version)
        if os.path.exists(versionDir):
            self.labelColorSet(self.ui.textureVersionEdit, unavailableColor)
            self.overwriteVersion = True
        else:
            self.labelColorSet(self.ui.textureVersionEdit, availableColor)
            self.overwriteVersion = False
        self.calcOutDir()

    def labelColorSet(self, label, qcolor):
        palette = label.palette()
        palette.setColor(label.foregroundRole(), qcolor)
        label.setPalette(palette)

    def channelSelectPopup( self, pos ):
        menu = QtWidgets.QMenu()
        menu.addAction( 'Select All Channels', self.channelSelectAll )
        menu.addAction( 'Select Current Channel', self.channelSelectCurrent )
        menu.addAction( 'Clear All Channels', self.channelClearAll )
        menu.exec_( QtGui.QCursor.pos() )

    def channelSelectAll( self ):
        channels = self.geo.channelList()
        for i in channels:
            widgetName = 'channel_%s' % i.name()
            eval( 'self.%s.setChecked( True )' % widgetName )

    def channelSelectCurrent( self ):
        self.channelClearAll()
        widgetName = 'channel_%s' % self.channel.name()
        eval( 'self.%s.setChecked( True )' % widgetName )

    def channelClearAll( self ):
        channels = self.geo.channelList()
        for i in channels:
            widgetName = 'channel_%s' % i.name()
            eval( 'self.%s.setChecked( False )' % widgetName )

    def getCheckedChannels( self ):
        result = list()
        channels = self.geo.channelList()
        for i in channels:
            widgetName = 'channel_%s' % i.name()
            if eval('self.%s.isChecked()' % widgetName):
                result.append( i )
        return result

    def layername_update(self):
        sender = self.sender()
        uiname = sender.objectName()
        uv_index = int(uiname.split('_')[-1])
        self.TextureNameInfo[uv_index] = sender.text()
        dxm.geoMetadata_txname_update(self.TextureNameInfo)
        # multi-tiled
        layerList = dxm.getMetadata_textureLayers()
        if layerList.count(sender.text()) > 1:
            if 'multiuv' in self.TextureNameInfo.keys():
                self.TextureNameInfo['multiuv'][uv_index] = 1001 + uv_index  # - (uv_index/10)*10
        # else:
        #     if self.TextureNameInfo.get('multiuv'):
        #         if uv_index in self.TextureNameInfo['multiuv'].keys():
        #             self.TextureNameInfo['multiuv'].pop(uv_index)
        dxm.geoMetadata_txname_update(self.TextureNameInfo)
        # debug
        print('# result : TextureName Update.')

    def get_layername(self):
        nameMap = dict()
        for i in self.geo.patchList():
            uv_index = i.uvIndex()
            widgetName = 'txname_edit_%s' % uv_index
            nameMap[uv_index] = eval('self.%s.text()' % widgetName)
        nameList = list()
        for i in nameMap:
            nameList.append(nameMap[i])
        for i in nameMap:
            self.TextureNameInfo[i] = nameMap[i]
            if nameList.count(nameMap[i]) > 1:
                if 'multiuv' in self.TextureNameInfo.keys():
                    self.TextureNameInfo['multiuv'][i] = 1001 + i  # - (i/10)*10
            # else:
            #     if self.TextureNameInfo.get('multiuv'):
            #         if i in self.TextureNameInfo['multiuv'].keys():
            #             self.TextureNameInfo['multiuv'].pop(i)
        dxm.geoMetadata_txname_update(self.TextureNameInfo)

    # export path set
    def set_export_dir( self ):
        startpath = self.path_edit.text()
        dir = QtWidgets.QFileDialog.getExistingDirectory( None, 'Export Path', startpath )
        dir = dir.replace( os.sep, '/' ) # to linux path
        if dir:
            self.path_edit.setText( dxm.dirReMap(dir) )

    # resolution set
    def set_resolution( self ):
        for i in self.geo.patchList():
            uv_index = i.uvIndex()
            widgetName = 'txname_rez_%s' % uv_index
            if eval('self.%s.isEnabled()' % widgetName):
                exec( 'self.%s.setCurrentIndex( self.rez_comboBox.currentIndex() )' % widgetName )

    def get_resolution( self ):
        rezDict = dict()
        for i in self.geo.patchList():
            widgetName = 'txname_rez_%s' % i.uvIndex()
            if eval('self.%s.isEnabled()' % widgetName):
                rez = eval('self.%s.currentText()' % widgetName)
                rezDict[i.uvIndex()] = REZ_INV_MAP[rez]
        return rezDict

    def calcOutDir(self):
        showDir = self.ui.showEdit.text()
        assetName = self.ui.assetNameEdit.text()
        branchName = self.ui.branchEdit.text()
        textureVersion = self.ui.textureVersionEdit.text()

        filepath = showDir
        if self.ui.shotCheckBox.isChecked():
            shot = self.ui.shotEdit.text()
            seq = shot.split("_")[0]
            filepath = os.path.join(filepath, "shot", seq, shot)

        filepath = os.path.join(filepath, "asset", assetName)

        if self.ui.branchCheckBox.isChecked():
            filepath = os.path.join(filepath, "branch", branchName)

        filepath = os.path.join(filepath, "texture", "images", textureVersion)
        self.ui.outDirEdit.setText(filepath)

    # texture export
    def texture_export( self ):
        self.get_layername()
        selected_channels = self.getCheckedChannels()
        if not selected_channels:
            mari.utils.message( 'Select export channels' )
            return

        outDir = self.ui.outDirEdit.text()
        export_class = ExportChannel( outDir )

        export_class.exportChannels = selected_channels
        export_class.rezMap = self.get_resolution()
        export_class.TextureNameInfo = self.TextureNameInfo

        export_class.doTxMake = True # self.txmake_checkBox.isChecked()
        export_class.doProxy  = True
        export_class.doCopy = self.ui.versionCopyCheckBox.isChecked()

        export_class.doIt()

        # set outpath metadata
        dxm.geoMetadata_outpath_update( outDir )

        if self.ui.shotCheckBox.isChecked():
            pass

        showDir = str(self.ui.showEdit.text())
        # showDir = os.path.join('show', str(self.ui.showEdit.text()), '_3d')
        assetName = str(self.ui.assetNameEdit.text())
        textureVersion = str(self.ui.textureVersionEdit.text())
        dataVersion = str(self.ui.dataVersionEdit.text())

        shotName = ""
        if self.ui.shotCheckBox.isChecked():
            shotName = str(self.ui.shotEdit.text())

        branchName = ""
        if self.ui.branchCheckBox.isChecked():
            branchName = str(self.ui.branchEdit.text())

        if not shotName:
            # txPath = os.path.join(str(self.ui.showEdit.text()), "asset", assetName)
            # if branchName:
            #     txPath = os.path.join(txPath, "branch", branchName)

            txPath = os.path.join(showDir, "asset", assetName)
            if branchName:
                txPath = os.path.join(txPath, "branch", branchName)

            txPath = os.path.join(txPath, "texture", "tex", textureVersion)
            AttrExport(txPath)

        else:
            dirPath = "{SHOW}/shot/{SEQ}/{SHOT}".format(SHOW = showDir,
                                                        SEQ = shotName.split("_")[0],
                                                        SHOT = shotName)
            txPath = os.path.join(dirPath, "asset", assetName)
            if branchName:
                txPath = os.path.join(txPath, "branch", branchName)
            txPath = os.path.join(txPath, "texture", "tex", textureVersion)

        self.close()

    def texture_resize( self ):
        rezMap = self.get_resolution()
        doresize = dict()
        for i in self.geo.patchList():
            uv_index = i.uvIndex()
            if uv_index in rezMap.keys():
                if self.channel.width(uv_index) != rezMap[uv_index]:
                    if rezMap[uv_index] in doresize.keys():
                        doresize[rezMap[uv_index]].append( uv_index )
                    else:
                        doresize[rezMap[uv_index]] = [uv_index]
        for r in doresize:
            self.channel.resize( r, doresize[r] )

def show_ui():
    global ExporterDialogWin
    ExporterDialogWin = ExporterDialog()


def AttrExport(txPath):
    import DXUSD.Tweakers as twk
    arg = twk.ATexture()
    arg.texAttrDir = txPath
    if arg.Treat():
        TT = twk.Texture(arg)
        TT.DoIt()
