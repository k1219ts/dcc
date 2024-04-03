from PySide2 import QtCore, QtUiTools, QtWidgets, QtGui
from PySide2.QtGui import *
from PySide2.QtWidgets import *
from PySide2.QtCore import *
import hou,PySide2,dragdrop,os

reload(dragdrop)

class dragDropOverlay(QDialog):
    def __init__(self,window,parm):
        super(dragDropOverlay, self).__init__()
        self.setParent(window, QtCore.Qt.Window)
        self.setWindowTitle('Drop Overlay')
        
        self.geo = window.frameGeometry()
        
        self.setFixedSize(self.geo.size())
        self.setGeometry(self.geo)
        self.setWindowOpacity(0.3)
        self.setAcceptDrops(True)
        self.dropEvent = self._dropEvent

        self.parm = parm
        
    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Escape:
            self.hide()
            
    def _dropEvent(self, event):
        self.hide()
        data = event.mimeData()
        if data.hasUrls():
            file_list = [url.toLocalFile() 
                for url in event.mimeData().urls()]

            if file_list:
                if self.parm:
                    Alt = (QtWidgets.QApplication.keyboardModifiers() & QtCore.Qt.AltModifier) == QtCore.Qt.AltModifier
                    if Alt:
                        file_list = dragdrop.seqCheck(file_list)

                    self.parm.set(file_list[0])
                else:
                    dragdrop.dropAccept(file_list)

    def closeAll(self):
        [entry.hide() 
            for entry in PySide2.QtWidgets.QApplication.allWidgets()
                if type(entry).__name__ == 'dragDropOverlay' and
                entry.isVisible()]
                
                
    def hideEvent(self, event):
        self.closeAll()

class activatePanel(QtWidgets.QFrame):
    def __init__(self,parent=None):
        super(activatePanel, self).__init__(parent)
        
        self.setProperty("houdiniStyle", True)
        self._error_brush = QtGui.QBrush(QtGui.QColor(255, 0, 0))
        self._error_brush.setStyle(QtCore.Qt.SolidPattern)

        self._warn_brush = QtGui.QBrush(QtGui.QColor(255, 255, 0))
        self._warn_brush.setStyle(QtCore.Qt.SolidPattern)

        self._info_brush = QtGui.QBrush(QtGui.QColor(0, 255, 0))
        self._info_brush.setStyle(QtCore.Qt.SolidPattern)

        self._msg_brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        self._msg_brush.setStyle(QtCore.Qt.SolidPattern)
        
        self.setWindowTitle('Activate Drag Drop')
        self.setAcceptDrops(True)
        self.setParent(hou.ui.mainQtWindow(), QtCore.Qt.Window)
        self.setWindowFlags(QtCore.Qt.Window | Qt.WindowStaysOnTopHint)
        self.resize(300, 150)

        layout = QVBoxLayout()    
        layout.setSpacing(5)
        layout.setSizeConstraint(QLayout.SetMinimumSize)

        self.info = QLabel('Drag over window to activate overlay.')
        self.info.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.info)

        self.setLayout(layout)

    def dragEnterEvent(self, event):
        data = event.mimeData()
        if data.hasUrls():
            path = event.mimeData().urls()[0].toLocalFile()
            if os.path.isfile(path):

                exist = [entry 
                    for entry in PySide2.QtWidgets.QApplication.allWidgets() 
                        if type(entry).__name__ == 'dragDropOverlay' and entry.isVisible()]

                if not exist:
                    startup(None)


class startup():
    def __init__(self,parm):
        self.parm = parm

        self.closeEntries()
        self.loadOverlay()

    def closeEntries(self):
        [entry.hide() 
            for entry in PySide2.QtWidgets.QApplication.allWidgets() 
                if type(entry).__name__ == 'dragDropOverlay' and entry.isVisible()]

    def loadOverlay(self):
        windowlist = [hou.qt.mainWindow()] + [hou.qt.floatingPanelWindow(p) 
            for p in hou.ui.curDesktop().floatingPanels()]

        for w in windowlist:
            try:
                dragDropOverlay(w,self.parm).show()
            except:
                pass