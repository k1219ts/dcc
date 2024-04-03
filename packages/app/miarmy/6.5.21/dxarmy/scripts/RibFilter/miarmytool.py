#encoding=utf-8
#!/usr/bin/env python

#-------------------------------------------------------------------------------
#
#   RenderMan TD
#
#       Sanghun Kim, rman.td@gmail.com
#
#
#
#    2015.03.11 $1
#-------------------------------------------------------------------------------


import os, sys
import time

from PyQt4 import QtGui, QtCore, uic

# import rif_process

currentFile = os.path.abspath( __file__ )
currentPath = os.path.dirname( currentFile )

class MiarmyTool( QtGui.QMainWindow ):
    def __init__( self ):
        QtGui.QMainWindow.__init__( self )
        self.ui = uic.loadUi( os.path.join(currentPath, 'tool.ui') )
        self.ui.setWindowFlags( QtCore.Qt.WindowStaysOnTopHint )
        self.ui.setWindowTitle( 'Miarmy Rif Window' )

        self.ui.label.setText( '' )
        self.ui.label.setPixmap( QtGui.QPixmap(os.path.join(currentPath, 'miarmy.jpg')) )

        self.ui.dragEnterEvent = self.DragEnterEvent
        self.ui.dragMoveEvent  = self.DragEnterEvent
        self.ui.dropEvent = self.Main_Drop
        self.ui.setAcceptDrops( True )

        self.ui.show()

    def DragEnterEvent( self, event ):
        event.accept()

    def Main_Drop( self, event ):
        link = event.mimeData().urls()
        #print link
        files = []
        for i in link:
            path = i.toLocalFile()
            if os.path.isfile( path ):
                if os.path.splitext(str(path))[-1] == '.rib':
                    files.append( str(path) )
            elif os.path.isdir( path ):
                for f in os.listdir( path ):
                    if os.path.splitext(f)[-1] == '.rib':
                        files.append( os.path.join(str(path), f) )
        # convert
        files.sort()
        getRmvList = self.ui.idLE.text()
        self.ThreadProcess( files, getRmvList )

    def ThreadProcess( self, files, getRmvList ):
        self.progress = QtGui.QProgressDialog( files[0], 'Cancel', 0, len(files), self )
        self.progress.setWindowTitle( 'Rif Process' )
        self.progress.resize( 400, 100 )
        thread = miarmy_thread( self, files, getRmvList )
        thread.connect( thread, QtCore.SIGNAL('increment'), self.incProgress )
        self.progress.connect( self.progress, QtCore.SIGNAL('canceled()'), thread, QtCore.SLOT('cancel()') )
        self.progress.show()
        thread.start()

    def incProgress( self, label ):
        if label:
            self.progress.setLabelText( label )
        self.progress.setValue( self.progress.value() + 1 )

#-------------------------------------------------------------------------------
class miarmy_thread( QtCore.QThread ):
    def __init__( self, parent, files, getRmvList ):
        super( miarmy_thread, self ).__init__( parent )
        self.cancel = False
        self.parent = parent
        self.files  = files
        self.getRmvList = getRmvList
        # create output dir
        filepath = os.path.dirname( self.files[0] )
        dirname  = os.path.basename( filepath )
        self.outdir = os.path.join( os.path.dirname(filepath), '%s_rif' % dirname )
        if not os.path.exists(self.outdir):
            os.makedirs( self.outdir )

    def run( self ):
        for i in range(len(self.files) + 1):
            if self.cancel:
                break
            label = ''
            if i != 0:
                label = self.files[i - 1]
                print label
                # rif main process
                outfile = os.path.join(self.outdir, os.path.basename(label))
                if self.getRmvList:
                    cmd = 'python %s/rif_process.py -f %s -o %s -x %s' % \
                          (currentPath, label, outfile, self.getRmvList)
                else:
                    cmd = 'python %s/rif_process.py -f %s -o %s' % \
                          (currentPath, label, outfile)
                # print cmd
                os.system(cmd)
                # time.sleep(.5)
            self.emit(QtCore.SIGNAL('increment'), label)

    @QtCore.pyqtSignature("")
    def cancel( self ):
        self.cancel = True


if __name__ == '__main__':
    app = QtGui.QApplication( sys.argv )
    win = MiarmyTool()
    sys.exit( app.exec_() )
