import hou
from PySide2 import QtCore, QtWidgets, QtGui
from PySide2.QtGui import *
from PySide2.QtWidgets import *
from PySide2.QtCore import *

class jumpNetwork(QtWidgets.QMenu):
    def __init__(self,tab,nodes):
        QtWidgets.QMenu.__init__(self)
        self.setStyleSheet(hou.ui.qtStyleSheet())

        a = self.addAction('Found in:')
        a.setEnabled(0)

        self.addSeparator()
        
        [self.addAction(n.path(), 
            lambda item=n: gotonode(tab,n)) 
            for n in nodes]
        
    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Escape:
            self.close()
            return

class initMenu(QtWidgets.QWidget):
    def __init__(self,tab,nodes):
        super(initMenu, self).__init__()
        self.tab,self.nodes = tab,nodes
        
    def showMenu(self):
        jumpNetwork(self.tab,self.nodes).exec_(self.mapToGlobal(QtGui.QCursor.pos()))
        self.close()

def startup(tab,node):
    nodes = getobjectmerge(node)

    if nodes:
        if len(nodes)>1:
            initMenu(tab,nodes).showMenu()
        else:
            gotonode(tab,nodes[0])

def gotonode(t,node):
    t.setPwd(node.parent())

    b = t.itemRect(node)
    b.expand((2.33,2.33))
    t.setVisibleBounds(b)

def getobjectmerge(node):
    nodes = [n for n in hou.node('/').allSubChildren(1,0) 
        if n.type().category().name() == 'Sop' and 
        n.type().name() == 'object_merge']

    nodes = [n for n in nodes 
        for i in range(n.evalParm('numobj')) 
            if n.node( n.evalParm('objpath%i'%(i+1)) ) == node]

    return list(set(nodes))