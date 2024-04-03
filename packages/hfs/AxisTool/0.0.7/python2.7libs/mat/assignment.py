import hou,PySide2,assign,sys
from PySide2 import QtCore, QtUiTools, QtWidgets, QtGui
from PySide2.QtGui import *
from PySide2.QtWidgets import *
from PySide2.QtCore import *

reload(assign)

BUTTON_ICON_SIZE = hou.ui.scaledSize(50)
houVer = int(hou.expandString('$HOUDINI_VERSION').split('.',1)[0])

class tree(QDialog):
    def __init__(self,node,reflist):
        super(tree, self).__init__()
        self.setWindowTitle(node.path())
        self.setParent(hou.ui.mainQtWindow(), QtCore.Qt.Window)
        self.setAcceptDrops(True)
        self.dropEvent = self._dropEvent
       
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

        self.node = node
        self.list = reflist
       
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
        self.tree.setAcceptDrops(True)
        self.tree.dropEvent = self._dropEvent
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
        self.menu.addAction('Remove', self.remove,'Delete')
        QShortcut(QKeySequence('Delete'), self).activated.connect(self.remove)

        self.menu.addAction('Replace', self.replace)

        self.menu.addAction('Assign New', self.assignMat,'Alt+c')
        QShortcut(QKeySequence('Alt+c'), self).activated.connect(self.assignMat)

        self.menu.addAction('Select Node(s)', self.selectNodes)

        self.menu.addSeparator()

        self.menu.addAction('Deselect All', self.deselect,'Ctrl+d')
        QShortcut(QKeySequence('Ctrl+d'), self).activated.connect(self.deselect)

        self.menu.addAction('Copy Path', self.selectAll,'Ctrl+c')
        QShortcut(QKeySequence('Ctrl+a'), self).activated.connect(self.copy)

        self.menu.addAction('Select All', self.selectAll,'Ctrl+a')
        QShortcut(QKeySequence('Ctrl+a'), self).activated.connect(self.selectAll)

        self.menu.addAction('Refresh', self.refresh,'Ctrl+r')
        QShortcut(QKeySequence('Ctrl+r'), self).activated.connect(self.refresh)

        self.menu.addAction('Close', self.close,'esc')
        QShortcut(QKeySequence('esc'), self).activated.connect(self.close)
       
        self.setLayout(layout)
       
        self.populate()
       
    def populate(self):

        ## Startup

        self.tree.clear()
        self.list.sort(key=lambda t: len(t[0].path()), reverse=False)
        self.items = []

        ## Search

        search = self.search.text().lower()
        if not search:
            nodelist = self.list
        else:
            nodelist = [n for n in self.list if n[0].path().lower().find(search) != -1]
        
        ## Populate

        for n,p,par in nodelist:
            parent = [i for i in self.items if i.text(0) == par]
            if parent:
                parent = parent[0]
                parent.setExpanded(1)

            else:
                parent = self.tree

            item = QtWidgets.QTreeWidgetItem(parent, [n.path()])
            item.setIcon(0, QIcon(hou.qt.Icon(n.type().icon(),
            BUTTON_ICON_SIZE, BUTTON_ICON_SIZE)))
           
            self.items.append(item)
           
    def copy(self):
        hou.ui.copyTextToClipboard(self.getSelected()[0])

    def remove(self):
        sel = self.getSelected()

        with hou.undos.group("Remove Materials"):
            [[parm.set('') for parm in p] for n,p,par in self.list if n.path() in sel]

        self.list = [(n,p,par) for n,p,par in self.list if n.path() not in sel]
       
        self.populate()
   
    def replace(self):
        sel = [n for n in self.list if n[0].path() in self.getSelected()]
        
        mat = hou.ui.selectNode(relative_to_node=None,
        initial_node=None, node_type_filter=None,
        title=None, width=0, height=0, multiple_select=False)

        with hou.undos.group("Replace Materials"):
            [[parm.set(mat) for parm in p] for n,p,par in sel]  
                   
        self.list = [n for n in self.list if n not in sel]

        self.populate()
       
    def getSelected(self):
        return [i.text(0) for i in self.tree.selectedItems()]
       
    def deselect(self):
        [i.setSelected(0) for i in self.tree.selectedItems()]
       
    def selectAll(self):
        [i.setSelected(1) for i in self.items]
    
    def selectNodes(self):
        hou.clearAllSelected()
        [hou.node(n).setSelected(True,clear_all_selected=False)
        for n in self.getSelected()]

    def refresh(self):
        run(self.node)

    def showMenu(self,event):
        self.menu.exec_(self.mapTo(
            self,QtGui.QCursor.pos()))

    def assignMat(self):

        ## Validity checks

        tree_selection = self.getSelected()
        if not tree_selection: return

        set_list = [n for n in self.list if n[0].path() in tree_selection]
        if not set_list: return

        ## Run assign material

        assign.run(set_list)

        ## Check if new mat assigned

        parmcompare = [n[1][0] for n in self.list if n[0].path() in tree_selection][0]

        if parmcompare.eval() != self.node.path():
            self.list = [n for n in self.list if n not in set_list]
            self.populate()

#    def dragMoveEvent(self, event):
#        if self.underMouse():
#            self.pane = self.getPane()

    def getPane(self):
        curdesk = hou.ui.curDesktop()
        activepane = curdesk.paneTabUnderCursor()
        
        if activepane:
            if activepane.type().name() == 'NetworkEditor':
                return [activepane,activepane.pwd()]

        return None

    def _dropEvent(self, event):
        if houVer > 17:

            ## Reset pane under cursor

            self.pane = self.getPane()

            if self.pane:
                self.count = 0
                self.resetPane()

            ## Get drop data

            data = event.mimeData().data(hou.qt.mimeType.nodePath)
            if data.isEmpty(): return

            paths = str(data).split("\t")

            nodes = [hou.node(p) for p in paths if p not in [n[0].path() for n in self.list]]

            if not nodes: return

            ## Add Path to Node(s)

            nodelist = []

            with hou.undos.group("Assign Materials"):
                for n in nodes:
                    cat = n.type().category().name()
                    name = n.type().name()
                    parent = n.parent().path()

                    if cat == 'Object':
                        if 'shop_materialpath' in [p.name() for p in n.parms()]:
                            parm = n.parm('shop_materialpath')
                            parm.set(self.node.path())
                            
                        else:
                            continue

                    elif cat == 'Sop':
                        if n.type().name() == 'material':
                            parm = n.parm('shop_materialpath1')
                            parm.set(self.node.path())
                            n.setName(self.node.name())

                        else:
                            chl = n.outputs()

                            mat = n.parent().createNode('material',self.node.name())
                            mat.setNextInput(n)
                            parm = mat.parm('shop_materialpath1')
                            parm.set(self.node.path())

                            # Add to wire
                            if chl:
                                for child in chl:
                                    for ind, input in enumerate(child.inputs()):
                                        if n.name() in input.name():
                                            index = ind
                                    child.setInput(index, None)
                                    child.insertInput(index, mat)
                                
                            mat.moveToGoodPosition(True, False, True, True)

                            n = mat

                    nodelist.append((n,[parm],parent))

            ## Repopulate tree

            if nodelist:
                self.list += nodelist
                self.populate()

    def resetPane(self):
        if self.count == 0:
            hou.ui.addEventLoopCallback(self.resetPane)
        elif self.count == 2:
            with hou.undos.disabler():
                self.pane[0].setPwd(self.pane[1])
        elif self.count == 3:
            hou.ui.removeEventLoopCallback(self.resetPane)

        self.count+=1

def getReferenced(node):
    geo = hou.node('/').allSubChildren(True, False)
   
    nodelist = []
   
    for n in geo:
        cat = n.type().category().name()
        name = n.type().name()
        parent = n.parent().path()
       
        if cat == 'Object':
            if 'shop_materialpath' in [p.name() for p in n.parms()]:
                try:
                    if n.node(n.evalParm('shop_materialpath')).path() == node.path():
                        nodelist.append((n,[n.parm('shop_materialpath')],parent))
                except: pass
               
        elif cat == 'Sop' and name == 'material':
            matcount = n.evalParm('num_materials')
            if matcount > 0:
                valid = []
                for p in range(matcount):
                    parm = n.parm('shop_materialpath%i'%(p+1))
                    try:
                        if n.node(parm.eval()).path() == node.path():
                            valid.append(parm)
                    except: pass

                if valid:
                    nodelist.append((n,valid,parent))
       
        elif cat == 'Driver':
            valid = []
            for p in n.parms():
                if 'string_type=NodeReference' in repr(p.parmTemplate().type):
                        
                    try:
                        if n.node(p.eval()).path() == node.path():
                            valid.append(p)
                    except: pass

            if valid:
                nodelist.append((n,valid,parent))
               
    return nodelist

def run(node):
    [entry.close() 
        for entry in PySide2.QtWidgets.QApplication.allWidgets() 
            if type(entry).__name__ == 'tree']

    reflist = getReferenced(node)

    tree(node,reflist).show()