import hou,PySide2
from PySide2 import QtCore, QtUiTools, QtWidgets, QtGui
from PySide2.QtGui import *
from PySide2.QtWidgets import *
from PySide2.QtCore import *

class outputOverlay(QDialog):
    def __init__(self,items,window):
        super(outputOverlay, self).__init__()
        self.setParent(window, QtCore.Qt.Window)
        self.setWindowTitle('Output Overlay')
        
        self.geo = window.frameGeometry()
        
        self.setFixedSize(self.geo.size())
        self.setGeometry(self.geo)
        self.setWindowOpacity(0.3)

        self.items = items
        self.inIND = len(items[0].inputs())-1
        self.outIND = len(items[1].outputs())-1
        self.mode = 0

        self.updatemode = hou.updateModeSetting()
        hou.setUpdateMode(hou.updateMode.Manual)

        self.setPrompt(False)
        
    def keyPressEvent(self, event):
        key = event.key()
        if key == QtCore.Qt.Key_Escape:
            self.hide()

        elif key == 16777217:
            self.mode = (self.mode+1)%2
            self.setPrompt(False)
        
    def closeAll(self):
        for entry in PySide2.QtWidgets.QApplication.allWidgets():
            if type(entry).__name__ == 'outputOverlay':
                entry.hide()
                
    def wheelEvent(self,event):

        input,out,pane = self.items

        ## Output

        if self.mode:
            reset = self.outIND
            input = out.outputs()

            for node in input:
                for ind, n in enumerate(node.inputs()):
                    if n == out:
                        node.setInput(ind, None)

                        if event.delta() > -1:
                            self.outIND+=1

                        else:
                            self.outIND-=1

                        try:
                            node.setInput(ind, out, self.outIND)
                        except:
                            self.outIND = reset
                            node.setInput(ind, out, self.outIND)

        ## Input

        else:
            reset = self.inIND
                
            curinput = input.inputs()[self.inIND]
            input.setInput(self.inIND, None)

            if event.delta() > -1:
                self.inIND+=1

            else:
                self.inIND-=1

            try:
                input.insertInput(self.inIND, curinput)
            except:
                self.inIND = reset
                input.insertInput(self.inIND, curinput)

        self.setPrompt(True)

    def setPrompt(self,setting):

        input,out,pane = self.items
        mode = ['input','output']
        ind = self.mode

        if ind:
            node = out.name()
            value = self.outIND

        else:
            node = input.name()
            value = self.inIND

        #message = '%s %s set to %i.\n'%(node,mode[ind],value)
        message = "Setting '%s' %s\n"%(node,mode[ind])
        message += 'Press TAB to change the mode.\n'
        message += 'Press ESC to close.'

        if setting:
            flashstr = '%s: %i'%(mode[ind].title(),value)
            pane.flashMessage(None,flashstr,1)

        pane.setPrompt(message)
        
    def hideEvent(self, event):
        self.closeAll()
        hou.setUpdateMode(self.updatemode)
                
class startup():
    def __init__(self, items):
        self.items = items
        self.closeEntries()
        self.loadOverlay()

    def closeEntries(self):
        for entry in PySide2.QtWidgets.QApplication.allWidgets():
            if type(entry).__name__ == 'outputOverlay':
                entry.hide()

    def loadOverlay(self):
        windowlist = [hou.qt.mainWindow()] + [hou.qt.floatingPanelWindow(p) 
            for p in hou.ui.curDesktop().floatingPanels()]
            
        [outputOverlay(self.items,w).show() for w in windowlist]