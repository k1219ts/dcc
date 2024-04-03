#!/usr/bin/python2.7
from __future__ import print_function
#
#
#	Dexter CG-Supervisor
#
#		Sanghun Kim, rman.td@gmail.com
#
#			txmake for dexter pipe-line


from PySide2 import QtWidgets, QtGui, QtCore
import os, sys, string, site, time, re
import txmake_process
import rename_process

current  = os.path.abspath( __file__ )
rootpath = os.path.dirname( current )

from txtoolUi import Ui_dxt

class dxToolWin( QtWidgets.QWidget):
    def __init__( self ):
        QtWidgets.QWidget.__init__( self )

        self.ui = Ui_dxt()
        self.ui.setupUi(self)
        self.iconpath = os.path.join( rootpath, 'icons' )
        self.setWindowIcon( QtGui.QIcon(QtGui.QPixmap(os.path.join(self.iconpath, 'render_PxrTexture.png'))) )
        self.setStyleSheet( 'color: rgb(200,200,200); background: rgb(48,48,48); border-width: 2px;' )
        self.setWindowTitle( 'txmaker v2.0' )

        #	color
        self.default_color = 'QtWidgets.QPushButton {background-color: rgb(48,48,48) }'

        self.ui.logo_label.setText('')
        self.ui.logo_label.setPixmap( QtGui.QPixmap(os.path.join(self.iconpath, 'DEXTER_DIGITAL_W.png')) )
        self.ui.txmove_checkBox.setChecked( False )
        self.txui()
        #self.ui.show()

        # drag & drop
        self.dragEnterEvent = self.DragEnterEvent
        self.dragMoveEvent = self.DragEnterEvent
        self.dropEvent = self.Main_Drop
        self.setAcceptDrops(True)

    def txui( self ):
        #	cmd type
        cmds = ['txmake']
        cmds_comboLineEdit = QtWidgets.QLineEdit()
        cmds_font = cmds_comboLineEdit.font()
        cmds_font.setPointSize( 11 )
        cmds_comboLineEdit.setFont( cmds_font )
        cmds_comboLineEdit.setReadOnly( True )
        self.ui.cmd_comboBox.setLineEdit( cmds_comboLineEdit )
        self.ui.cmd_comboBox.addItems( cmds )

        #   Grid Images
        self.ui.mgrid_label.setPixmap( QtGui.QPixmap(os.path.join(self.iconpath, 'txmake_periodic_periodic.jpg')) )

        #   S Images
        smode_black = QtGui.QIcon()
        smode_black.addFile( os.path.join(self.iconpath, 'txmake-smode-black.gif') )
        self.ui.S_black.setStyleSheet( self.default_color )
        self.ui.S_black.setStyleSheet( 'QtWidgets.QPushButton#S_black:checked {background-color: rgb(78,78,78)}' )
        self.ui.S_black.setIcon( smode_black )
        self.ui.S_black.setIconSize( QtCore.QSize(30,30) )
        smode_clamp = QtGui.QIcon()
        smode_clamp.addFile( os.path.join(self.iconpath, 'txmake-smode-clamp.gif') )
        self.ui.S_clamp.setStyleSheet( self.default_color )
        self.ui.S_clamp.setStyleSheet( 'QtWidgets.QPushButton#S_clamp:checked {background-color: rgb(78,78,78)}' )
        self.ui.S_clamp.setIcon( smode_clamp )
        self.ui.S_clamp.setIconSize( QtCore.QSize(30,30) )
        smode_periodic = QtGui.QIcon()
        smode_periodic.addFile( os.path.join(self.iconpath, 'txmake-smode-periodic.gif') )
        self.ui.S_periodic.setStyleSheet( self.default_color )
        self.ui.S_periodic.setStyleSheet( 'QtWidgets.QPushButton#S_periodic:checked {background-color: rgb(78,78,78)}' )
        self.ui.S_periodic.setIcon( smode_periodic )
        self.ui.S_periodic.setIconSize( QtCore.QSize(30,30) )

        #   T Images
        tmode_black = QtGui.QIcon()
        tmode_black.addFile( os.path.join(self.iconpath, 'txmake-tmode-black.gif') )
        self.ui.T_black.setStyleSheet( self.default_color )
        self.ui.T_black.setStyleSheet( 'QtWidgets.QPushButton#T_black:checked {background-color: rgb(78,78,78)}' )
        self.ui.T_black.setIcon(tmode_black)
        self.ui.T_black.setIconSize(QtCore.QSize(30,30))
        tmode_clamp = QtGui.QIcon()
        tmode_clamp.addFile( os.path.join(self.iconpath, 'txmake-tmode-clamp.gif') )
        self.ui.T_clamp.setStyleSheet( self.default_color )
        self.ui.T_clamp.setStyleSheet( 'QtWidgets.QPushButton#T_clamp:checked {background-color: rgb(78,78,78)}' )
        self.ui.T_clamp.setIcon(tmode_clamp)
        self.ui.T_clamp.setIconSize(QtCore.QSize(30,30))
        tmode_periodic = QtGui.QIcon()
        tmode_periodic.addFile( os.path.join(self.iconpath, 'txmake-tmode-periodic.gif') )
        self.ui.T_periodic.setStyleSheet( self.default_color )
        self.ui.T_periodic.setStyleSheet( 'QtWidgets.QPushButton#T_periodic:checked {background-color: rgb(78,78,78)}' )
        self.ui.T_periodic.setIcon(tmode_periodic)
        self.ui.T_periodic.setIconSize(QtCore.QSize(30,30))

        #   ST Link Images
        st_link = QtGui.QIcon()
        st_link.addFile( os.path.join(self.iconpath, 'txmake-link.gif') )
        self.ui.ST_link.setStyleSheet( self.default_color )
        self.ui.ST_link.setStyleSheet( 'QtWidgets.QPushButton#ST_link:checked {background-color: rgb(78,78,78)}' )
        self.ui.ST_link.setIcon(st_link)
        self.ui.ST_link.setIconSize(QtCore.QSize(30,30))

        #   ST mode ToolTip
        self.ui.S_black.setToolTip('smode black')
        self.ui.S_clamp.setToolTip('smode clamp')
        self.ui.S_periodic.setToolTip('smode periodic')
        self.ui.T_black.setToolTip('tmode black')
        self.ui.T_clamp.setToolTip('tmode clamp')
        self.ui.T_periodic.setToolTip('tmode periodic')
        self.ui.ST_link.setToolTip('Link smode and tmode')
        #-----------------------------------------------------------------------
        #   ST mode
        self.ui.S_black.clicked.connect(self.stmode_process)
        self.ui.S_clamp.clicked.connect(self.stmode_process)
        self.ui.S_periodic.clicked.connect(self.stmode_process)
        self.ui.T_black.clicked.connect(self.stmode_process)
        self.ui.T_clamp.clicked.connect(self.stmode_process)
        self.ui.T_periodic.clicked.connect(self.stmode_process)

        #	format
        formats = ['pixar', 'openexr']
        format_comboLineEdit = QtWidgets.QLineEdit()
        format_font = format_comboLineEdit.font()
        format_font.setPointSize( 11 )
        format_comboLineEdit.setFont( format_font )
        format_comboLineEdit.setReadOnly( True )
        self.ui.format_comboBox.setLineEdit( format_comboLineEdit )
        self.ui.format_comboBox.addItems(formats)

        #   type
        types = ['texturemap', 'envlatl']
        types_comboLineEdit = QtWidgets.QLineEdit()
        types_font = types_comboLineEdit.font()
        types_font.setPointSize( 11 )
        types_comboLineEdit.setFont( types_font )
        types_comboLineEdit.setReadOnly( True )
        self.ui.type_comboBox.setLineEdit( types_comboLineEdit )
        self.ui.type_comboBox.addItems(types)

        #   precision
        precisions = ['preserve', 'byte', 'short', 'half', 'float']
        precisions_comboLineEdit = QtWidgets.QLineEdit()
        precisions_font = precisions_comboLineEdit.font()
        precisions_font.setPointSize( 11 )
        precisions_comboLineEdit.setFont( precisions_font )
        precisions_comboLineEdit.setReadOnly( True )
        self.ui.precision_comboBox.setLineEdit( precisions_comboLineEdit )
        self.ui.precision_comboBox.addItems(precisions)

        #   access
        accesses = ['normalized', 'corrected']
        accesses_comboLineEdit = QtWidgets.QLineEdit()
        accesses_font = accesses_comboLineEdit.font()
        accesses_font.setPointSize( 11 )
        accesses_comboLineEdit.setFont( accesses_font )
        accesses_comboLineEdit.setReadOnly( True )
        self.ui.access_comboBox.setLineEdit( accesses_comboLineEdit )
        self.ui.access_comboBox.addItems(accesses)

        #   pattern
        patterns = ['diagonal', 'single', 'all']
        patterns_comboLineEdit = QtWidgets.QLineEdit()
        patterns_font = patterns_comboLineEdit.font()
        patterns_font.setPointSize( 11 )
        patterns_comboLineEdit.setFont( patterns_font )
        patterns_comboLineEdit.setReadOnly( True )
        self.ui.pattern_comboBox.setLineEdit( patterns_comboLineEdit )
        self.ui.pattern_comboBox.addItems(patterns)

    def stmode_process( self ):
        sender = self.sender()
        object = sender.objectName()

        if self.ui.ST_link.isChecked():
            prefix = object.split('_')
            self.stmode_checkedButton( '%s_%s' % ('S', prefix[-1]) )
            self.stmode_checkedButton( '%s_%s' % ('T', prefix[-1]) )
        else:
            self.stmode_checkedButton( object )

        # button state
        (smode, tmode) = self.stmode_buttonState()
        self.ui.mgrid_label.setPixmap( QtGui.QPixmap(os.path.join(self.iconpath, 'txmake_%s_%s.jpg' % (smode, tmode))) )

    def stmode_checkedButton(self, object):
        current = eval('self.ui.%s' % object)
        current.setChecked(True)

        data = ['black', 'clamp', 'periodic']
        name_split = object.split('_')
        data.remove(name_split[-1])
        for i in data:
            obj = eval('self.ui.%s_%s' % (name_split[0], i))
            obj.setChecked(False)
        return None

    def stmode_buttonState(self):
        smode = ''; tmode = ''
        data = ['black', 'clamp', 'periodic']

        for i in data:
            #   S mode
            sobj = eval('self.ui.S_%s' % i)
            if sobj.isChecked():
                smode = i
            #   T mode
            tobj = eval('self.ui.T_%s' % i)
            if tobj.isChecked():
                tmode = i
        return smode, tmode

    # options
    def txmake_options( self ):
        (smode, tmode) = self.stmode_buttonState()
        format = self.ui.format_comboBox.currentText()
        type = self.ui.type_comboBox.currentText()
        precision = self.ui.precision_comboBox.currentText()
        access = self.ui.access_comboBox.currentText()
        pattern = self.ui.pattern_comboBox.currentText()
        newer = self.ui.newer_checkBox.checkState()

    # ----------------------------------------------------------------------------------
    # Drag and Drop
    def DragEnterEvent( self, event ):
        event.accept()

    def Main_Drop( self, event ):
        imgExt = ['tiff', 'tif', 'jpg', 'exr', 'hdr', 'tga', 'dpx', 'bmp', 'png', 'xpm', 'ppm', 'gif']

        files = []
        link = event.mimeData().urls()
        for i in link:
            data = i.toLocalFile()
            if os.path.isfile(data):
                if str(data).split('.')[-1] in imgExt:
                    files.append(str(data))
                elif os.path.isdir(data):
                    for f in os.listdir(data):
                        fn = os.path.join(str(data), f)
                        if os.path.isfile(fn) and f.split('.')[-1] in imgExt:
                            files.append(fn)
            elif os.path.isdir(data):
                # print 'is dir :', data
                for f in os.listdir(data):
                    if f.split('.')[-1] in imgExt:
                        files.append(os.path.join(str(data), f))

        # Get UI Options
        self.opts = TxOptions(self)

        if len(files) > 5:
            msgbox = QtWidgets.QMessageBox()
            msgbox.setWindowIcon(QtGui.QIcon(QtGui.QPixmap(os.path.join(self.iconpath, 'rm22_txmake.png'))))
            msgbox.setIcon(QtWidgets.QMessageBox.Question)
            msgbox.setWindowTitle('Question')
            msgbox.setText('Please select a process location?\t')
            # msgbox.setInformativeText('information')
            msgbox.addButton('Local', QtWidgets.QMessageBox.NoRole)
            msgbox.addButton('Tractor', QtWidgets.QMessageBox.YesRole)
            msgbox.addButton('Cancel', QtWidgets.QMessageBox.RejectRole)
            result = msgbox.exec_()
            if result == 1:
                # print '>> tractor-spool'
                self.tractorProcess(files)
                return
            elif result == 2:
                return

        self.localProcess(files)

    def tractorProcess(self, files):
        import tractor_process
        tractor_process.JobMain(files, self.opts).doIt()

    def localProcess(self, files):
        cmdString = self.ui.cmd_comboBox.currentText()
        if cmdString == 'txmake' and files:
            self.ThreadProcess( 'txmake', files )

    def ThreadProcess( self, cmd, files ):
        self.progress = QtWidgets.QProgressDialog( files[0], 'Cancel', 0, len(files) )
        self.progress.setWindowTitle( '%s Process' % cmd )
        self.progress.resize(400, 100)

        if cmd == 'txmake':
            thread = txmake_process.txmake_thread(self, files, self.opts)
            thread.increment.connect(self.incProgress)
            # thread.connect( thread, QtCore.SIGNAL('increment'), self.incProgress )
            self.progress.connect( self.progress, QtCore.SIGNAL('canceled()'), thread, QtCore.SLOT('cancel()') )
            self.progress.show()
            thread.start()
            return None

    def incProgress( self, label ):
        if label:
            self.progress.setLabelText( label )
        self.progress.setValue( self.progress.value() + 1 )
        QtWidgets.QApplication.processEvents()

class TxOptions:
    def __init__(self, parent):
        # Map Type ( texturemap or environment envlatl )
        self.m_maptype = parent.ui.type_comboBox.currentText()

        # tex move to pipeline convention
        self.m_txmove = False
        if parent.ui.txmove_checkBox.checkState() == 2:
            self.m_txmove = True

        # Wrap Mode
        self.m_smode, self.m_tmode = parent.stmode_buttonState()

        # Format
        self.m_format = str(parent.ui.format_comboBox.currentText())

        # Data Type
        self.m_precision = str(parent.ui.precision_comboBox.currentText())

        # Image Resize
        access = str(parent.ui.access_comboBox.currentText())
        self.m_resize = 'up'
        if access == 'normalized':
            self.m_resize = 'up-'

        # MipMap Pattern
        self.m_pattern = str(parent.ui.pattern_comboBox.currentText())

        # Newer
        self.m_newer = False
        if parent.ui.newer_checkBox.checkState() == 2:
            self.m_newer = True

    def getCommand(self):
        if self.m_maptype == 'envlatl':
            result = ['txenvlatl']
        else:
            result  = ['txmake']
            # Wrap Mode
            result += ['-smode', self.m_smode, '-tmode', self.m_tmode]
            # MipMap Pattern
            result += ['-pattern', self.m_pattern]
            # Image Resize
            result += ['-resize', self.m_resize]
            # Format
            if self.m_format != 'pixar':
                result += ['-format', self.m_format]
        # Newer
        if self.m_newer:
            result += ['-newer']
        # result += ['-verbose']
        return result

    def getEnvCommand(self):
        result = ['txenvlatl']
        if self.m_newer:
            result += ['-newer']
        return result

    def getOutFilename(self, filename):
        dirname   = os.path.dirname(filename)
        basename  = os.path.basename(filename)
        base, ext = os.path.splitext(basename)

        # changing output filename
        changingName = rename_process.baseName_mud2mari(base)
        if changingName:
            base = changingName
        result = os.path.join(dirname, '%s.tex' % base)

        if self.m_maptype == 'envlatl' or filename.split('.')[-1] == 'hdr':
            result = base + '.exr'
        else:
            if self.m_txmove:
                if '/texture/images' in dirname:
                    source = [dirname.split('/images')[0], 'tex']
                    vcheck = re.compile(r'v(\d+)').findall(os.path.basename(dirname))
                    if vcheck:
                        source += ['v%s' % vcheck[0]]
                    newdir = '/'.join(source)
                    if not os.path.exists(newdir):
                        os.makedirs(newdir)
                    result = newdir + '/%s.tex' % base
        return result


if __name__ == '__main__':
    app = QtWidgets.QApplication( sys.argv )
    win = dxToolWin()
    win.show()
    sys.exit( app.exec_() )
