import hou,os,sys,ast,re,PySide2
from math import sqrt,ceil
from PySide2 import QtCore, QtUiTools, QtWidgets, QtGui
from PySide2.QtGui import *
from PySide2.QtWidgets import *
from PySide2.QtCore import *

class extractor2(QDialog):
    def __init__(self):
        super(extractor2, self).__init__()
        self.setWindowTitle("Extract")

        ## Parent to panel under cursor or main window

        panetab = hou.ui.curDesktop().paneTabUnderCursor()
        if panetab:
            panel = panetab.pane().floatingPanel()
            self.setParent(hou.qt.floatingPanelWindow(panel), QtCore.Qt.Window)
        else:
            self.setParent(hou.ui.mainQtWindow(), QtCore.Qt.Window)

        ## On top if not windows

        if sys.platform != 'win32':
            self.setWindowFlags(QtCore.Qt.Window | Qt.WindowStaysOnTopHint)

        ## Window Position

        cursor_position = QtGui.QCursor.pos().toTuple()
        winsize = ((400,100))
        self.setFixedSize(winsize[0],winsize[1])

        self.setGeometry(cursor_position[0] - (winsize[0] / 2), cursor_position[1] - (winsize[1] / 2),winsize[0], winsize[1])

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
       
        sh = hou.ui.qtStyleSheet()
        self.setStyleSheet(sh)

        minsize = 75
        BUTTON_ICON_SIZE = hou.ui.scaledSize(24)
       
        layout = QVBoxLayout()    
        layout.setSpacing(5)
        layout.setSizeConstraint(QLayout.SetMinimumSize)

        tabh = QHBoxLayout()
        tabh.setSpacing(5)
        tabh.setSizeConstraint(QLayout.SetMinimumSize)
        layout.addLayout(tabh)
        
        tabh.addItem(QSpacerItem(0, 55, QSizePolicy.Fixed, QSizePolicy.Fixed))
        label = QLabel('Name')
        label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        label.setMinimumWidth(minsize)
        label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        tabh.addWidget(label)
        self.name = QLineEdit()
        self.name.setMinimumHeight(30)
        self.name.setPlaceholderText('Extract Name')
        tabh.addWidget(self.name)

        self.namemode = QPushButton()
        self.namemode.setIcon(hou.qt.Icon("BUTTONS_comment_out", 55, 55))
        self.namemode.setFlat(True)
        self.namemode.setCheckable(1)
        self.namemode.setFixedSize(30, 30)
        tabh.addWidget(self.namemode)

        tabh = QHBoxLayout()
        tabh.setSpacing(5)
        tabh.setSizeConstraint(QLayout.SetMinimumSize)
        layout.addLayout(tabh)
        
        label = QLabel('Null Name')
        label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        label.setMinimumWidth(minsize)
        label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        tabh.addWidget(label)
        self.nullname = hou.qt.ComboBox()
        self.nullname.addItems(['_out','out_','None',"Don't create"])
        tabh.addWidget(self.nullname)

        self.null = QPushButton()
        self.null.setIcon(hou.qt.Icon("BUTTONS_delete", 55, 55))
        self.null.setFlat(True)
        self.null.setCheckable(1)
        self.null.setFixedSize(25, 25)
        tabh.addWidget(self.null)        

        tabh = QHBoxLayout()
        tabh.setSpacing(5)
        tabh.setSizeConstraint(QLayout.SetMinimumSize)
        layout.addLayout(tabh)
        
        label = QLabel('Layout')
        label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        label.setMinimumWidth(minsize)
        label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        tabh.addWidget(label)
        self.layout = hou.qt.ComboBox()
        self.layout.addItems(['Below','Above','Left','Right'])
        tabh.addWidget(self.layout)

        self.xform = QPushButton()
        self.xform.setIcon(hou.qt.Icon("OBJ_null", 55, 55))
        self.xform.setFlat(True)
        self.xform.setCheckable(1)
        self.xform.setFixedSize(25, 25)
        tabh.addWidget(self.xform)

        layout.addItem(QSpacerItem(0, 10, QSizePolicy.Expanding, QSizePolicy.Fixed))

        tabh = QHBoxLayout()
        tabh.setSpacing(5)
        tabh.setSizeConstraint(QLayout.SetMinimumSize)
        layout.addLayout(tabh)

        self.info = QLabel()
        self.info.setEnabled(0)
        tabh.addWidget(self.info)

        tabh.addItem(QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Minimum))


        BUTTON_ICON_SIZENEW = hou.ui.scaledSize(35)
        self.confirm = QPushButton()
        self.confirm.setIcon(hou.qt.Icon("CHANNELS_value_time_handles", 55, 55))

        self.confirm.setFlat(True)
        self.confirm.setFixedSize(25, 25)
        self.confirm.setToolTip('Enter or Ctrl+Enter to assign material & close.')

        tabh.addWidget(self.confirm)

        self.setLayout(layout)

        ## User Shortcuts

        shortcut1 = QShortcut(QKeySequence("Return"), self)
        shortcut1.activated.connect(lambda:self.extract(0))
        shortcut1 = QShortcut(QKeySequence("Ctrl+Return"), self)
        shortcut1.activated.connect(lambda:self.extract(1))

        shortcut2 = QShortcut(QKeySequence("Ctrl+1"), self)
        shortcut2.activated.connect(lambda:self.switchType(self.nullname))
        self.nullname.setToolTip('Ctrl+1')

        shortcut3 = QShortcut(QKeySequence("Ctrl+2"), self)
        shortcut3.activated.connect(lambda:self.switchType(self.layout))
        self.layout.setToolTip('Ctrl+2')

        shortcut4 = QShortcut(QKeySequence("Tab"), self)
        shortcut4.activated.connect(lambda:self.switchType(self.namemode))
        self.namemode.setToolTip('Get name from selected node. Tab')

        shortcut5 = QShortcut(QKeySequence("Shift+Tab"), self)
        shortcut5.activated.connect(lambda:self.switchType(self.null))
        self.null.setToolTip("Don't create null. Shift+Tab")

        shortcut6 = QShortcut(QKeySequence("Ctrl+Tab"), self)
        shortcut6.activated.connect(lambda:self.switchType(self.xform))
        self.xform.setToolTip('Transform into object. Ctrl+Tab')

        ## Callbacks

        self.nullname.currentIndexChanged.connect(self.focus)
        self.layout.currentIndexChanged.connect(self.focus)
        self.xform.clicked.connect(self.focus)
        self.null.clicked.connect(lambda:self.toggle(self.null,self.nullname))
        self.namemode.clicked.connect(lambda:self.toggle(self.namemode,self.name))

        self.confirm.clicked.connect(lambda:self.extract(0))

        ## Load Previous Selections
        
        self.pref = hou.expandString('$HOUDINI_USER_PREF_DIR/extract.pref')

        self.loadDefaults()

        self.shift = (1.0, 2.0)

    def toggle(self,button,combo):
        combo.setEnabled(
            button.isChecked() != True)

        self.focus()

    def switchType(self,widget):
        if widget == self.null:
            widget.toggle()
            self.toggle(widget,self.nullname)
        elif widget == self.namemode:
            widget.toggle()
            self.toggle(widget,self.name)
        elif widget == self.xform:
            widget.toggle()
        else:
            newInd = (widget.currentIndex() + 1) % widget.count()
            widget.setCurrentIndex(newInd)

    def extract(self,close):
        sel = hou.selectedNodes()
        nfo = self.info.setText

        if not sel:
            nfo('Nodes not selected.')
            return

        #level = self.getLevel(sel[0])

        for n in sel:
            outputs = n.outputs()

            name = self.getName(n)
            if not name:
                nfo('Name contains illegal characters.')

            ## Create Geo Node

            geo = hou.node('/obj').createNode('geo', name)
            name = geo.name()

            ## Null

            ind = self.nullname.currentIndex()
            null_txt = self.nullname.currentText()
            if ind != 3:
                if ind == 0:
                    null_name = name+null_txt
                elif ind == 1:
                    null_name = null_txt+name
                else:
                    null_name = name

                null = n.createOutputNode('null',null_name)

                if outputs:
                    for child in outputs:
                        [child.setInput(ind, null) 
                            for c2 in child.inputs() 
                                if c2 == n]

                null.moveToGoodPosition(1,0,1,1)

                n = null

            ## Create Object Merge

            obmerge = geo.createNode('object_merge',None)
            obmerge.parm('objpath1').set(n.path())
            if self.xform.isChecked():
                obmerge.parm('xformtype').set(1)

            self.setPos(n.parent(),geo)

        if close:
            self.hide()

    def getLevel(self,n):
        n = n.path()
        level = len(n.split('/'))

        for s in range( level-2 ):
            s+=1

            par = hou.node(n.rsplit('/',s)[0])

            if par:
                if par.type().category().name() == 'Object':
                    return par


    def getName(self,n):
        if self.namemode.isChecked():
            return n.name()

        name = self.name.currentText()
        if name:
            if not re.search(r"([^a-z0-9_ ])", name.lower()):

                name = name.replace(' ','_')
                return name

        return None

    def takeSecond(self,elem):
        return elem[1]

    def setPos(self,source,geo):
        children = geo.parent().children()

        matchpos = source.position()[0]

        nodes = [(c,c.position()[1]) for c in children 
            if c.position()[0] == matchpos and 
            c.position()[1] < source.position()[1]]

        nodes.sort(key=self.takeSecond)

        movefrom = None
        if len(nodes)>1:
            count = 0
            for node,pos in nodes[1:]:
                if nodes[ind][pos] > pos - 2:
                    movefrom = node

                count+=1

        if not movefrom:
            movefrom = source

        geo.setPosition(movefrom)
        geo.move((0,-self.shift[1]))

    ## Focus on Line Edit

    def mousePressEvent(self, event):
        self.focus()
        self.name.selectAll()

    def focus(self):
        self.name.setFocus()

    ## Load & Save Selection

    def hideEvent(self, event):
        self.saveChanges()
        self.setParent(None)

    def saveChanges(self):
        item = [self.namemode.isChecked(),self.nullname.currentIndex(),self.null.isChecked(),self.layout.currentIndex(),self.xform.isChecked()]
        a = file(self.pref, 'w')
        a.write(repr(item))

    def loadDefaults(self):
        if os.path.isfile(self.pref):
            f = file(self.pref,'r')
            with open(self.pref, 'r') as f:
                recent = ast.literal_eval(f.read())

            try:
                self.namemode.setChecked(recent[0])
                self.toggle(self.namemode,self.name)
            except: pass
            try:
                self.nullname.setCurrentIndex(recent[1])
            except: pass
            try:
                self.null.setChecked(recent[2])
                self.toggle(self.null,self.nullname)
            except: pass
            try:
                self.layout.setCurrentIndex(recent[3])
            except: pass
            try:
                self.xform.setChecked(recent[4])
            except: pass

def run():
    [entry.close() for entry in __import__('PySide2').QtWidgets.QApplication.allWidgets() 
        if type(entry).__name__ == 'extractor2' 
            and entry.isVisible()]

    extractor2().show()
