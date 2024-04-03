import hou,os,sys,time,ast,re,subprocess
from PySide2 import QtCore, QtUiTools, QtWidgets, QtGui
from PySide2.QtGui import *
from PySide2.QtWidgets import *
from PySide2.QtCore import *

class verDirectory(QtWidgets.QMenu):
    def __init__(self,parent=None):
        QtWidgets.QMenu.__init__(self)

        self.setStyleSheet(hou.ui.qtStyleSheet())

        versions = self.versionGet()
        if not versions:
            action = self.addAction('Version not found')
            action.setEnabled(0)
            return

        self.addAction('{0:15}Ctrl+O'.format('Open Folder'),
            self.openFolder)

        self.addSeparator()
        
        [self.addAction('{0:22}{1}'.format(v,m), 
            lambda item=v: self.open(item)) 
            for v,m in versions]
            
    def versionGet(self):
        hipFile,ext = hou.hipFile.path().replace('\\','/').rsplit('.',1)
        path,hipName = hipFile.rsplit('/',1)
        
        verSplit = None
        verFormat = None
        if len(re.findall('_v(?=\d+)', hipName)) > 0:
            verSplit = re.split('_v(?=\d+)', hipName)
            verFormat = '_v'
        elif len(re.findall('_V(?=\d+)', hipName)) > 0:
            verSplit = re.split('_V(?=\d+)', hipName)
            verFormat = '_V'
        
        if verSplit:
            padding = len(verSplit[1])
            matchlist = []

            for f in os.listdir(path):
                if f.endswith('.%s'%(ext)):
                    joined = os.path.join(path,f)
                    f = f.rsplit('.',1)[0]
                    if len(re.findall('%s(?=\d+)'%(verFormat), f)) > 0:
                        sec = re.split('%s(?=\d+)'%(verFormat), f)
                        if sec[0] == verSplit[0]:#and len(sec[1]) == padding
                            modified = time.strftime('%m/%d/%Y %H:%M:%S', 
                                time.gmtime(os.path.getmtime(joined)))
                            vername = 'v%s'%sec[1]
                            matchlist.append((vername,modified))
                            
            self.start = '%s/%s%s'%(path,verSplit[0],verFormat)
            self.end = '.%s'%ext
            
            return matchlist
            
        else: return
        
    def open(self,v):
        self.close()
        
        file = self.start + v[1:] + self.end
        hou.hipFile.load(file)
        
    def openFolder(self):
        file = os.path.dirname(self.start)
        
        platform = sys.platform
        if platform == "win32":
            os.startfile(file)
        elif platform == "darwin":
            subprocess.Popen(["open", file])
        else:
            subprocess.Popen(["xdg-open", file])
        
    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Escape:
            self.close()
            return

        if (QtWidgets.QApplication.keyboardModifiers() & QtCore.Qt.ControlModifier) == QtCore.Qt.ControlModifier:
            if event.key() == 79:
                self.close()
                self.openFolder()
                return

class initMenu(QtWidgets.QWidget):
    def __init__(self):
        super(initMenu, self).__init__()
        
    def showMenu(self):
        verDirectory().exec_(self.mapToGlobal(QtGui.QCursor.pos()))
        self.close()