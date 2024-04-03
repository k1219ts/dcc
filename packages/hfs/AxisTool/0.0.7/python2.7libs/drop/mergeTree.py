import hou,PySide2,os,subprocess,sys,shutil
from PySide2 import QtCore, QtUiTools, QtWidgets, QtGui
from PySide2.QtGui import *
from PySide2.QtWidgets import *
from PySide2.QtCore import *

class mergetree(QDialog):
    def __init__(self,nodelist,file):
        super(mergetree, self).__init__()
        self.setWindowTitle('Merge %s'%(
            os.path.basename(file).rsplit('.',1)[0]))
        self.setParent(hou.ui.mainQtWindow(), QtCore.Qt.Window)
       
        ## Window Positioning Start

        cursor_position = QtGui.QCursor.pos().toTuple()
        winsize = ((275,350))
        self.resize(winsize[0],winsize[1])

        self.setGeometry(cursor_position[0] - (winsize[0] / 2), cursor_position[1] - (winsize[1] / 2),winsize[0], winsize[1])
        
        ## Parent to panel under cursor

        panetab = hou.ui.curDesktop().paneTabUnderCursor()
        if panetab:
            panel = panetab.pane().floatingPanel()
            self.setParent(hou.qt.floatingPanelWindow(panel), QtCore.Qt.Window)


        ## Move within monitor bounds

        screen = QApplication.desktop().screenNumber(QApplication.desktop().cursor().pos())
        geo = QApplication.desktop().availableGeometry(screen)

        x,y = [cursor_position[0] - (winsize[0] / 2), cursor_position[1] - (winsize[1] / 2)]
        padx,pady = [22,45]

        if cursor_position[0] + (winsize[0] / 2) > geo.right():
            x = geo.right() - winsize[0] - padx
        if x < geo.x():
            x = geo.x() + (padx/2)

        if cursor_position[1] + (winsize[1] / 2) > geo.bottom():
            y = geo.bottom() - (winsize[1]) - pady
        if y < geo.y():
            y = geo.y() + (pady/3)

        self.move(x,y)

        ## Window Positioning End
       
        sh = hou.ui.qtStyleSheet()
        self.setStyleSheet(sh)

        if sys.platform != 'win32':
            self.setWindowFlags(QtCore.Qt.Window | Qt.WindowStaysOnTopHint)

        self.nodes = nodelist
        self.file = file
       
        layout = QVBoxLayout()    
        layout.setSpacing(0)
        layout.setSizeConstraint(QLayout.SetMinimumSize)
        layout.setContentsMargins(0, 0, 0, 0)

        ## Tree
       
        self.tree = QTreeWidget()
        self.tree.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        self.tree.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.tree.header().setSectionResizeMode(QHeaderView.Stretch)
        self.tree.setHeaderItem(QTreeWidgetItem(['Nodes']))
        self.tree.setAlternatingRowColors(True)
        self.tree.setWordWrap(True)
        self.tree.header().hide()

        self.tree.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self.showMenu)
        layout.addWidget(self.tree)

        self.search = QLineEdit()
        self.search.setPlaceholderText('Search')
        self.search.setMinimumHeight(30)
        self.search.textChanged.connect(self.populate)
        layout.addWidget(self.search)
       
        ## Menu
       
        self.menu = hou.qt.Menu()

        self.menu.addAction('Merge', self.merge,'Ctrl+m')
        QShortcut(QKeySequence('Ctrl+m'), self).activated.connect(self.merge)

        self.menu.addSeparator()

        self.menu.addAction('Deselect All', self.deselect,'Ctrl+d')
        QShortcut(QKeySequence('Ctrl+d'), self).activated.connect(self.deselect)

        self.menu.addAction('Select All', self.selectAll,'Ctrl+a')
        QShortcut(QKeySequence('Ctrl+a'), self).activated.connect(self.selectAll)

        self.menu.addSeparator()

        self.menu.addAction('Close', self.close,'esc')
        QShortcut(QKeySequence('esc'), self).activated.connect(self.close)
       
        self.setLayout(layout)
       
        self.populate()
       
    def populate(self):

        ## Search

        search = self.search.text().lower()
        if not search:
            nodelist = self.nodes
        else:
            nodelist = [path for path in self.nodes if path.lower().find(search) != -1]
       
        ## Populate QTree

        self.tree.clear()
        self.items = []

        for n in nodelist:
            parent = self.tree
            level = len(n.split('/'))

            ## Find parent accounting for dive target nodes

            for s in range( level-2 ):
                s+=1

                par = n.rsplit('/',s)[0]
                parentlist = [i for i in self.items if i.text(0) == par]
                if parentlist: 
                    parent = parentlist[0]
                    break

            item = QtWidgets.QTreeWidgetItem(parent, [n])
            #Icons can't be set as we're just reading plain text# item.setIcon(0, QIcon(hou.qt.Icon(n.type().icon(),BUTTON_ICON_SIZE, BUTTON_ICON_SIZE)))
           
            self.items.append(item)
       
    def getSelected(self):
        return [i.text(0) for i in self.tree.selectedItems()]
       
    def deselect(self):
        [i.setSelected(0) for i in self.tree.selectedItems()]
       
    def selectAll(self):
        [i.setSelected(1) for i in self.items]

    def showMenu(self,event):
        self.menu.exec_(self.mapTo(
            self,QtGui.QCursor.pos()))

    def merge(self):

        ## Get selected

        sel = self.getSelected()
        if not sel: return

        ## Get parent nodes

        parents = []
        for n in sel:
            level = len(n.split('/'))

            for s in range( level-2 ):
                s+=1

                par = n.rsplit('/',s)[0]
                if par not in sel:
                    parents.append(par)

        ## Append child nodes

        sel = [s for ss in sel for s in [ss,'%s/*'%ss]]

        ## Prepend parent list

        sel = parents+sel

        ## List to string

        sel = ','.join(sel)

        ## Node Collision

        col = hou.hipFile.collisionNodesIfMerged(self.file, sel)
        if col:
            col = hou.ui.displayMessage('Overwrite existing nodeswith the same name?', buttons=('Yes','No'), 
                severity=hou.severityType.Message, default_choice=0, close_choice=1, details=', '.join([n.path() for n in col]), title='Collision')

            if col:
                col = False
            else:
                col = True

        else:
            col = False

        ## Merge

        hou.hipFile.merge(self.file, sel, col, False)

def hipNodes(file):
    if file.find(' ') != -1:
        hou.ui.displayMessage('Cannot merge hip files with spaces in the directory.', 
            buttons=('OK',), severity=hou.severityType.Error, default_choice=0, close_choice=0, title='Error')

        return

    platform = sys.platform

    if platform == "win32":
        hcmd = os.path.abspath( hou.expandString('$HB/hcpio.exe') )
        cmd = '%s -i -t -I %s'%(hcmd,file)
        
        process=subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output = process.communicate()

        cmdout = output[0].split('\n')

    else:# linux and mac being different
        hcmd = os.path.abspath( hou.expandString('$HB/hexpand') )
        cmd = '%s %s'%(hcmd,file)
        
        process=subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output = process.communicate()
        
        path = os.getenv("HOME")
        file0 = '%s/%s'%(path,output[0].split('stored in ')[1].split('\n')[0])
        file1 = '%s/%s'%(path,output[0].split('expanded into the directory ')[1].split('\n')[0])
        
        cmdout = open(file0, 'r').readlines() 
        
        ## Remove Temp Files
        
        os.remove(file0)
        shutil.rmtree(file1)

    nodes = []
    dirs = []
    for n in cmdout:
        if '/' in n:
            path = '/%s'%(n.split('.',1)[0])
            dir = '/%s'%path.split('/',2)[1]
            if path not in nodes:
                nodes.append(path)
            if dir not in dirs:
                dirs.append(dir)
                
    nodes = sorted(nodes, key=len)

    return dirs + nodes

def run(file):
    [entry.close() for entry in PySide2.QtWidgets.QApplication.allWidgets() if type(entry).__name__ == 'mergetree']

    nodes = hipNodes(file)
    if not nodes:
        return

    mergetree(nodes,file).show()