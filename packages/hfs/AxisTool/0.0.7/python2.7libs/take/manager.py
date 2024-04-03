import hou,re,sys
from PySide2 import QtCore, QtUiTools, QtWidgets, QtGui
from PySide2.QtGui import *
from PySide2.QtWidgets import *
from PySide2.QtCore import *
from functools import partial 

houVer = int(hou.expandString('$HOUDINI_VERSION').split('.',1)[0])

class takeManager(QtWidgets.QFrame):
    def __init__(self,parent=None):
        super(takeManager, self).__init__(parent)
        
        self.setProperty("houdiniStyle", True)
        self._error_brush = QtGui.QBrush(QtGui.QColor(255, 0, 0))
        self._error_brush.setStyle(QtCore.Qt.SolidPattern)

        self._warn_brush = QtGui.QBrush(QtGui.QColor(255, 255, 0))
        self._warn_brush.setStyle(QtCore.Qt.SolidPattern)

        self._info_brush = QtGui.QBrush(QtGui.QColor(0, 255, 0))
        self._info_brush.setStyle(QtCore.Qt.SolidPattern)

        self._msg_brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        self._msg_brush.setStyle(QtCore.Qt.SolidPattern)
        
        self.setWindowTitle('Take Manager')
        self.resize(1000, 700)
        self.setParent(hou.ui.mainQtWindow(), QtCore.Qt.Window)

        if sys.platform != 'win32':
            self.setWindowFlags(QtCore.Qt.Window | Qt.WindowStaysOnTopHint)
       
        sh = hou.ui.qtStyleSheet()
        self.setStyleSheet(sh)
        minsize = 75
       
        takegen = QtWidgets.QWidget()
        layout = QVBoxLayout()    
        layout.setSpacing(5)
        layout.setSizeConstraint(QLayout.SetMinimumSize)
        takegen.setLayout(layout)
        
        col1 = QHBoxLayout()    
        col1.setSpacing(5)
        col1.setSizeConstraint(QLayout.SetMinimumSize)
        layout.addLayout(col1)
        
        # Initialize tab screen
        self.tabs = QTabWidget()
        self.tab1 = QWidget()
        self.tab2 = QWidget()
        self.tab3 = QWidget()
        self.tab4 = QWidget()
        self.tabs.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding)
       
        # Add tabs
        self.tabs.addTab(self.tab1,"Switch")
        self.tabs.addTab(self.tab2,"Menu")
        self.tabs.addTab(self.tab3,"Attribute")
        self.tabs.addTab(self.tab4,"Node Changes")
       
#        groupBox = QGroupBox("Generate Takes")
#        col1.addWidget(groupBox)
        
        #
        #
        # Method
        #
        #
        
        methodlayout = QVBoxLayout()
        methodlayout.setSpacing(5)
        methodlayout.setSizeConstraint(QLayout.SetMinimumSize)
#        groupBox.setLayout(methodlayout)
        col1.addLayout(methodlayout)
        
        groupBox = QGroupBox("Method")
        methodlayout.addWidget(groupBox)
        
        line1 = QVBoxLayout()
        line1.setSpacing(5)
        line1.setSizeConstraint(QLayout.SetMinimumSize)
        groupBox.setLayout(line1)
        line1.addWidget(self.tabs)
        
        #
        # Switch
        #
        
        tablayout = QVBoxLayout()    
        tablayout.setSpacing(5)
        tablayout.setSizeConstraint(QLayout.SetMinimumSize)
        self.tab1.setLayout(tablayout)
        tabh = QHBoxLayout()
        tabh.setSizeConstraint(QLayout.SetMinimumSize)
        tabh.setSpacing(5)
        tablayout.addLayout(tabh)
        
        label = QLabel('Switch Parm')
        label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        label.setMinimumWidth(minsize)
        tabh.addWidget(label)
        self.switchnode = QLineEdit()
        self.switchnode.setPlaceholderText('Switch Parameter')
        self.switchnode.dropEvent = partial(self._dropEvent,self.switchnode,False)
        tabh.addWidget(self.switchnode)
        self.switchbutton = hou.qt.ParmChooserButton()
        
        self.switchbutton.parmSelected.connect(partial( self.node_tree_on_select,self.switchnode,False ))
        
        tabh.addWidget(self.switchbutton)
        #self.switchbutton.nodeSelected.connect(lambda:self.nodeselected(self.switchnode))
        
        tabh = QHBoxLayout()
        tabh.setSpacing(5)
        tabh.setSizeConstraint(QLayout.SetMinimumSize)
        tablayout.addLayout(tabh)
        
        label = QLabel('Name Mode')
        label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        label.setMinimumWidth(minsize)
        label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        tabh.addWidget(label)
        self.switchnamingmode = hou.qt.ComboBox()
        self.switchnamingmode.addItems('Input Name, Input Index'.split(', '))
        self.switchnamingmode.currentIndexChanged.connect(self.switchindexChanged)
        tabh.addWidget(self.switchnamingmode)

        tabh = QHBoxLayout()
        tabh.setSizeConstraint(QLayout.SetMinimumSize)
        tabh.setSpacing(5)
        tablayout.addLayout(tabh)
        
        label = QLabel('Padding')
        label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        label.setMinimumWidth(minsize)
        label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        tabh.addWidget(label)
        self.spadding = QSpinBox()
        self.spadding.setMinimum(1)
        self.spadding.setEnabled(0)
        tabh.addWidget(self.spadding)
        
        label = QLabel('Start From')
        label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        label.setMinimumWidth(minsize)
        label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        tabh.addWidget(label)
        self.sstartnum = QSpinBox()
        self.sstartnum.setMinimum(0)
        self.sstartnum.setEnabled(0)
        tabh.addWidget(self.sstartnum)
        
        tablayout.addItem(QSpacerItem(0, 0, QSizePolicy.Minimum, QSizePolicy.Expanding))
        
        #
        # Menu
        #
        
        tablayout = QVBoxLayout()
        tablayout.setSpacing(5)
        tablayout.setSizeConstraint(QLayout.SetMinimumSize)
        self.tab2.setLayout(tablayout)
        
        tabh = QHBoxLayout()
        tabh.setSizeConstraint(QLayout.SetMinimumSize)
        tabh.setSpacing(5)
        tablayout.addLayout(tabh)

        label = QLabel('Menu Parm')
        label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        label.setMinimumWidth(minsize)
        tabh.addWidget(label)
        self.menunode = QLineEdit()
        self.menunode.setPlaceholderText('Menu Parameter')
        self.menunode.dropEvent = partial(self._dropEvent,self.menunode,False)
        tabh.addWidget(self.menunode)
        self.menubutton = hou.qt.ParmChooserButton()
        
        self.menubutton.parmSelected.connect(partial( self.node_tree_on_select,self.menunode,False ))
        tabh.addWidget(self.menubutton)
        #self.switchbutton.nodeSelected.connect(lambda:self.nodeselected(self.menunode))
        
        
        tabh = QHBoxLayout()
        tabh.setSizeConstraint(QLayout.SetMinimumSize)
        tabh.setSpacing(5)
        tablayout.addLayout(tabh)
        
        label = QLabel('Name Mode')
        label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        label.setMinimumWidth(minsize)
        label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        tabh.addWidget(label)
        self.menunamingmode = hou.qt.ComboBox()
        self.menunamingmode.addItems('Menu Label, Menu Index'.split(', '))
        self.menunamingmode.currentIndexChanged.connect(self.menuindexChanged)
        tabh.addWidget(self.menunamingmode)

        tabh = QHBoxLayout()
        tabh.setSizeConstraint(QLayout.SetMinimumSize)
        tabh.setSpacing(5)
        tablayout.addLayout(tabh)
        
        label = QLabel('Padding')
        label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        label.setMinimumWidth(minsize)
        label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        tabh.addWidget(label)
        self.padding = QSpinBox()
        self.padding.setMinimum(1)
        self.padding.setEnabled(0)
        tabh.addWidget(self.padding)
        
        label = QLabel('Start From')
        label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        label.setMinimumWidth(minsize)
        label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        tabh.addWidget(label)
        self.mstartnum = QSpinBox()
        self.mstartnum.setMinimum(0)
        self.mstartnum.setEnabled(0)
        tabh.addWidget(self.mstartnum)
        
        tablayout.addItem(QSpacerItem(0, 0, QSizePolicy.Minimum, QSizePolicy.Expanding))

        #
        # Attribute
        #
        
        tablayout = QVBoxLayout()
        tablayout.setSpacing(5)
        tablayout.setSizeConstraint(QLayout.SetMinimumSize)
        self.tab3.setLayout(tablayout)
        
        tabh = QHBoxLayout()
        tabh.setSizeConstraint(QLayout.SetMinimumSize)
        tabh.setSpacing(5)
        tablayout.addLayout(tabh)

        label = QLabel('Change Parm')
        label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        label.setMinimumWidth(minsize)
        tabh.addWidget(label)

        self.parm_change = QLineEdit()
        self.parm_change.setPlaceholderText('Parameter to change')
        self.parm_change.dropEvent = partial(self._dropEvent,self.parm_change,False)
        tabh.addWidget(self.parm_change)
        self.parm_change_menu = hou.qt.ParmChooserButton()
        
        self.parm_change_menu.parmSelected.connect(partial( self.node_tree_on_select,self.parm_change,False ))
        tabh.addWidget(self.parm_change_menu)
        
        tabh = QHBoxLayout()
        tabh.setSizeConstraint(QLayout.SetMinimumSize)
        tabh.setSpacing(5)
        tablayout.addLayout(tabh)

        label = QLabel('Entry')
        label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        label.setMinimumWidth(minsize)
        tabh.addWidget(label)

        self.prefix = QLineEdit()
        self.prefix.setPlaceholderText("@rand!=$ATTRIB;")
        self.prefix.setToolTip('$ATTRIB will be replaced with the\nattribute specified below and must be included.')
        tabh.addWidget(self.prefix)

        tablayout.addItem(QSpacerItem(0, 5, QSizePolicy.Minimum, QSizePolicy.Minimum))
        tablayout.addWidget(hou.qt.Separator())
        tablayout.addItem(QSpacerItem(0, 5, QSizePolicy.Minimum, QSizePolicy.Minimum))
        
        tabh = QHBoxLayout()
        tabh.setSizeConstraint(QLayout.SetMinimumSize)
        tabh.setSpacing(5)
        tablayout.addLayout(tabh)
        
        label = QLabel('Attrib Node')
        label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        label.setMinimumWidth(minsize)
        tabh.addWidget(label)

        self.attribute_node = QLineEdit()
        self.attribute_node.setPlaceholderText('Node to get attribute from')
        self.attribute_node.dropEvent = partial(self._dropEvent,self.attribute_node,True)
        self.attribute_node.textChanged.connect(self.attribute_update)
        tabh.addWidget(self.attribute_node)
        self.attribute_node_chooser = hou.qt.NodeChooserButton()
        self.attribute_node_chooser.setNodeChooserFilter(hou.nodeTypeFilter.Sop)
        
        self.attribute_node_chooser.nodeSelected.connect(partial( self.node_tree_on_select,self.attribute_node,True ))
        tabh.addWidget(self.attribute_node_chooser)
        
        tabh = QHBoxLayout()
        tabh.setSizeConstraint(QLayout.SetMinimumSize)
        tabh.setSpacing(5)
        tablayout.addLayout(tabh)

        label = QLabel('Attrib Type')
        label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        label.setMinimumWidth(minsize)
        label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        tabh.addWidget(label)
        self.attribute_type = hou.qt.ComboBox()
        self.attribute_type.addItems(['Point','Primitive'])
        self.attribute_type.currentIndexChanged.connect(self.attribute_update)
        tabh.addWidget(self.attribute_type)

        tabh = QHBoxLayout()
        tabh.setSizeConstraint(QLayout.SetMinimumSize)
        tabh.setSpacing(5)
        tablayout.addLayout(tabh)

        label = QLabel('Attribute')
        label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        label.setMinimumWidth(minsize)
        label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        tabh.addWidget(label)
        self.attribute_name = hou.qt.ComboBox()
        tabh.addWidget(self.attribute_name)
        
        tablayout.addItem(QSpacerItem(0, 0, QSizePolicy.Minimum, QSizePolicy.Expanding))
        
        #
        # Node Record
        #
        
        tablayout = QVBoxLayout()    
        tablayout.setSpacing(5)
        tablayout.setSizeConstraint(QLayout.SetMinimumSize)
        self.tab4.setLayout(tablayout)
        tabh = QHBoxLayout()
        tabh.setSizeConstraint(QLayout.SetMinimumSize)
        tabh.setSpacing(5)
        tablayout.addLayout(tabh)
        
        label = QLabel('Records changes made to node flags and parameter values.')
        label.setEnabled(False)
        label.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        label.setMinimumWidth(minsize)
        tabh.addWidget(label)
        
        tabh = QHBoxLayout()
        tabh.setSizeConstraint(QLayout.SetMinimumSize)
        tabh.setSpacing(5)
        tablayout.addLayout(tabh)
        
        tabh.addItem(QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Minimum))
        self.record = QPushButton('Record')
        self.record.clicked.connect(self.recordFunct)
        self.record.setCheckable(True)
        self.record.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        tabh.addWidget(self.record)

        self.clearreset = QPushButton('Clear && Reset Nodes')
        self.clearreset.clicked.connect(self.resetNodes)
        self.clearreset.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        tabh.addWidget(self.clearreset)
        tabh.addItem(QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Minimum))
        
        tablayout.addItem(QSpacerItem(0, 15, QSizePolicy.Minimum, QSizePolicy.Minimum))
        
        tabh = QHBoxLayout()
        tabh.setSpacing(5)
        tabh.setSizeConstraint(QLayout.SetMinimumSize)
        tablayout.addLayout(tabh)
        
        label = QLabel('History')
        #label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        tabh.addWidget(label)############################
        tabh.addItem(QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Minimum))
        
        tablayout.addItem(QSpacerItem(0, 5, QSizePolicy.Minimum, QSizePolicy.Minimum))
                
        tabh = QHBoxLayout()
        tabh.setSpacing(5)
        tabh.setSizeConstraint(QLayout.SetMinimumSize)
        groupBox.setLayout(tabh)
        
        self.recordList = QTreeWidget()
        self.recordList.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.recordList.setHeaderItem(QTreeWidgetItem(["Path","Type","Value"]))
        self.recordList.header().setSectionResizeMode(QHeaderView.Stretch)
        self.recordList.setAlternatingRowColors(True)
        self.recordList.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.recordList.setWordWrap(True)
        tablayout.addWidget(self.recordList)

       
        #
        #
        # Checkout
        #
        #
                
        groupBox = QGroupBox("Take Creation")
        methodlayout.addWidget(groupBox)
        
        gblayout = QVBoxLayout()
        gblayout.setSpacing(5)
        groupBox.setLayout(gblayout)
        
        line1 = QHBoxLayout()
        line1.setSpacing(5)
        gblayout.addLayout(line1)
        
        label = QLabel('Take Name')
        label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        label.setMinimumWidth(minsize)
        line1.addWidget(label)
        
        self.takeName = QLineEdit()
        description = '*METHOD* / *PARENT* & plain text'
        self.takeName.setPlaceholderText(description)
        self.takeName.setToolTip(description)
        line1.addWidget(self.takeName)
        
        self.addTerm = QPushButton()
        if houVer > 17:
            self.addTerm.setMaximumWidth(27)
        else:
            self.addTerm.setMaximumWidth(24)
        self.termMenu = hou.qt.Menu()
        self.termMenu.addAction('*METHOD*', lambda:self.addTermFunct('*METHOD*'),'Ctrl+1',context=QtCore.Qt.WidgetShortcut)
        self.termMenu.addAction('*PARENT*', lambda:self.addTermFunct('*PARENT*'),'Ctrl+2')
        self.addTerm.setMenu(self.termMenu)
        
        line1.addWidget(self.addTerm)
        
        line1 = QHBoxLayout()
        line1.setSpacing(5)
        gblayout.addLayout(line1)
        
        self.takeInfo = QLabel('')
        self.takeInfo.setEnabled(0)
        line1.addWidget(self.takeInfo)
        
        line1.addItem(QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Minimum))
        
        self.menu = hou.qt.Menu()
        self.menu.addAction('Add to Current Take', lambda:self.createTakes(0),'Shift+Return')
        self.menu.addAction('Create Child of Selected', lambda:self.createTakes(1),'Ctrl+Return')
        
        self.takes = QPushButton('Create Take(s)')
        self.takes.setMenu(self.menu)
        line1.addWidget(self.takes)
        
        
        #
        #
        # Takes & Actions
        #
        #
        
        row1 = QVBoxLayout()    
        row1.setSpacing(5)
        row1.setSizeConstraint(QLayout.SetMinimumSize)
        col1.addLayout(row1)
        
        groupBox = QGroupBox("Takes")
        row1.addWidget(groupBox)
        
        line1 = QHBoxLayout()
        line1.setSpacing(5)
        groupBox.setLayout(line1)

        self.list = QTreeWidget()
        self.list.setMinimumWidth(340)
        self.list.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        self.list.setHeaderItem(QTreeWidgetItem(["Takes"]))
        self.list.header().setSectionResizeMode(QHeaderView.Stretch)
        self.list.setAlternatingRowColors(True)
        self.list.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.list.itemSelectionChanged.connect(self.selectionChange)
        self.list.setWordWrap(True)
        self.list.header().hide()
                        
        line1.addWidget(self.list)
        
        #
        #
        # Take Actions
        #
        #
        
        minsize = 100
        
        groupBox = QGroupBox("Take Actions")
        row1.addWidget(groupBox)
        
        gblayout = QVBoxLayout()
        gblayout.setSpacing(5)
        groupBox.setLayout(gblayout)
        
        line1 = QHBoxLayout()
        line1.setSpacing(5)
        gblayout.addLayout(line1)
        
        label = QLabel('Depth Selection')
        label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        label.setMinimumWidth(minsize)
        line1.addWidget(label)
        
        self.depth = QSpinBox()
        self.depth.valueChanged.connect(self.selectDepth)
        self.depth.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        line1.addWidget(self.depth)        
        
        line1.addItem(QSpacerItem(minsize, 0, QSizePolicy.Expanding, QSizePolicy.Minimum))
        
        self.menussev = hou.qt.Menu()
        self.menussev.addAction('Select All', lambda:self.selectAll(0),'Ctrl+a')
        self.menussev.addAction('Invert Selection', lambda:self.selectAll(1), 'Ctrl+i')
        self.menussev.addAction('Select Children', self.selectChildren)
        self.menussev.addAction('Select Latest Created', lambda:self.selectTakes(self.recent))
        self.menussev.addAction('Edit Selection', self.editSelection)
        self.menussev.addSeparator()
        
        self.menussev.addAction('Node Copies per Take', self.copyNodes)
        self.menussev.addSeparator()
        
        self.menussev.addAction('Delete Selected', self.takeDelete)
        self.menussev.addSeparator()
        
        self.menussev.addAction('Refresh Takes', self.populate,'Ctrl+r')
        
        self.sev = QPushButton('Actions')
        self.sev.setMenu(self.menussev)
        self.sev.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        line1.addWidget(self.sev)
        
        #gblayout.addItem(QSpacerItem(0, 15, QSizePolicy.Expanding, QSizePolicy.Minimum))
        self.actionInfo = QLabel('')
        self.actionInfo.setEnabled(0)
        self.actionInfo.setMinimumHeight(25)
        self.actionInfo.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        gblayout.addWidget(self.actionInfo)
        
#        self.detailsplitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
#        self.detailsplitter.addWidget(takegen)
#        self.detailsplitter.addWidget(takeactions)

        # Top level layout just contains the main horizontal splitter.
        #layout = QtWidgets.QVBoxLayout()
        #layout.setContentsMargins(MARGIN, MARGIN, MARGIN, MARGIN)
        #layout.addWidget(self.detailsplitter)

        self.setLayout(layout)
        
        self.populate()
        
        shortcut1 = QShortcut(QKeySequence("Delete"), self.list)
        shortcut1.setContext(QtCore.Qt.WidgetWithChildrenShortcut)
        shortcut1.activated.connect(self.takeDelete)
        
        # Attributes        
        self.recorddata = []
        self.recent = []
        self.selectedNodes = []
        self.accepted_flags = ['Render','Bypass','Display']
        self.resetList = []
    
    ## POPULATE TAKES START
        
    def populate(self):
        self.list.clear()
        
        self.itemlist = []
        for t,p,d in self.getTakes():
            if not p:
                row = QtWidgets.QTreeWidgetItem(self.list, [t.name()])
            else:
                for row,text,bla in self.itemlist:
                    if p.name() == text:
                        parent = row
                        break
                        
                row = QtWidgets.QTreeWidgetItem(parent, [t.name()])
                parent.setExpanded(1)
                
            self.itemlist.append(( row, t.name(),d ))
        
    def getTakes(self):
        take = hou.takes.rootTake()
        hou.takes.setCurrentTake(take)
        childlist = []
        finallist = [(take,None,0)]
        templist = []
        
        run = 1
        if take.children(): 
            for c in take.children():
                childlist.append(c)
                finallist.append((c,c.parent(),1 ))
        else: run = 0
        
        count = 2
        while run:
            run = 0
            for ind,t in enumerate(childlist):
                for c in t.children():
                    finallist.append((c,c.parent(),count ))
                    if t.children(): 
                        templist.append(c)
                        run = 1
        
            childlist = templist
            templist = []
            count+=1
            
        return finallist
        
    def getSelected(self):
        return [i for i in self.list.selectedItems()]
        
    def selectionChange(self):
        count = len(self.getSelected())
        if count == 0: 
            self.actionInfo.clear()
            return
        if count == 1: plural = ''
        else: plural = 's'
        self.actionInfo.setText('%i Take%s Selected.'%(count,plural))

    def selectAll(self,mode):
        s = True
        if self.itemlist[0][0].isSelected(): s = False
        for r,n,d in self.itemlist:
            if mode:
                if r.isSelected(): r.setSelected(0)
                else: r.setSelected(1)
            else: r.setSelected(s)

    def selectChildren(self):
        sel = self.getSelected()
        desel = hou.ui.displayMessage('Deselect initial takes?', buttons=('Yes','No'),default_choice=0, close_choice=1, title='Info')
        wait = True
        for r,n,d in self.itemlist:
            if r.isSelected():
                depth = d
                wait = False
            elif not wait:
                if d > depth:
                    r.setSelected(1)
                else:
                    wait = True

        if not desel:
            [i.setSelected(0) for i in sel]

            
    def editSelection(self):
        selection = self.getSelected()
        def restoree():
            [i.setSelected(1) for i in selection]

        def funct():
            if every.value() == 1:
                restoree()
            else:
                [i.setSelected((ind+offset.value())%every.value() != 0) for ind, i in enumerate(selection)]

        d = QDialog()
        d.setModal(1)
        d.setWindowTitle('Edit Selection')
        d.setFixedSize(250,0)
        d.setParent(hou.ui.mainQtWindow(), QtCore.Qt.Window)

        minSize = 100

        layout = QVBoxLayout()
        layout.setSpacing(10)
        layout.setSizeConstraint(QLayout.SetMinimumSize)

        line = QHBoxLayout()    
        line.setSpacing(5)
        line.setSizeConstraint(QLayout.SetMinimumSize)
        layout.addLayout(line)

        label = QLabel('Select Every')
        label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignTop)
        label.setMinimumWidth(minSize)
        line.addWidget(label)

        every = QSpinBox()
        every.setMinimum(1)
        every.valueChanged.connect(funct)
        line.addWidget(every)

        line = QHBoxLayout()    
        line.setSpacing(5)
        line.setSizeConstraint(QLayout.SetMinimumSize)
        layout.addLayout(line)

        label = QLabel('Offset')
        label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignTop)
        label.setMinimumWidth(minSize)
        line.addWidget(label)

        offset = QSpinBox()
        offset.valueChanged.connect(funct)
        line.addWidget(offset)

        layout.addWidget(hou.qt.Separator())

        line = QHBoxLayout()    
        line.setSpacing(5)
        line.setSizeConstraint(QLayout.SetMinimumSize)
        layout.addLayout(line)

        line.addItem(QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Minimum))
        restore = QPushButton('Restore Selection')
        restore.clicked.connect(restoree)
        line.addWidget(restore)

        d.setLayout(layout)
        d.show()

    def selectDepth(self):
        depth = self.depth.value()
        for r,n,d in self.itemlist:
            if d == depth:
                r.setSelected(1)
            else:
                r.setSelected(0)
                
    def selectTakes(self,namelist):
        if namelist:
            [r.setSelected(r.text(0) in namelist) for r,n,d in self.itemlist]
        else:
            self.actionInfo.setText('Empty list.')
            
    def nodeselected(self,entrybox):
        entrybox.setText(node_path)
        
    def takeDelete(self):
        selected = [hou.takes.findTake(t.text(0)).destroy() for t in self.getSelected()]
        
        self.populate()
        
    ## RECORDING ACTIONS START
        
    def recordFunct(self):
        if self.record.isChecked():
            [self.callbackList(n) for n in hou.node('/').allSubChildren()]
            hou.clearAllSelected()
        else:
            self.setCallbacks()
            #self.resetNodes()
            
    def populateRecord(self):#Populate Tree
        self.recordList.clear()

        for p,t,v in self.recorddata:
            item = QtWidgets.QTreeWidgetItem(self.recordList, [p,t,v])
            item.setToolTip(0, p)
            item.setToolTip(1, t)
            item.setToolTip(2, v)
            
    def addData(self,input):
        if not self.recorddata: 
            self.recorddata = [input]
        else:
            self.recorddata = [input if e[0] == input[0] and e[2].rsplit(': ',1)[0] == input[2].rsplit(': ',1)[0] else e for e in self.recorddata]
            
        if input not in self.recorddata:
            self.recorddata.append(input)
            
        # Render/Display Flag Fix
        oglist = self.recorddata
        if input[1] == 'Flag':
            if input[2].startswith(('Render: ','Display: ')):
                count = -1
                for p,t,v in oglist:
                    count+=1
                    node = hou.node(p)
                    if p!=input[0]:
                        if node.type().category().name() == 'Sop':
                            if node.parent() == hou.node(input[0]).parent():
                                if v.split(': ',1)[0] == input[2].split(': ',1)[0]:
                                    v = '%s: False'%v.split(': ',1)[0]
                                    self.recorddata[count] = (p,t,v)
                        
        self.populateRecord()
    
    def setCallbacks(self):
        [n.removeAllEventCallbacks() for n in hou.node('/').allSubChildren()]
            
    def callbackList(self,n):
        n.addEventCallback((hou.nodeEventType.ParmTupleChanged,), self.parmChange)
        n.addEventCallback((hou.nodeEventType.FlagChanged,), self.flagChange)
        n.addEventCallback((hou.nodeEventType.ChildCreated,), self.childChange)
        n.addEventCallback((hou.nodeEventType.SelectionChanged,hou.nodeEventType.ChildSelectionChanged), self.selectChange)
                
    def parmChange(self,**kwargs):#Parm Change
        p = kwargs["parm_tuple"]
        node = kwargs["node"]
        if not [item for item in self.resetList if item[0].path() == node.path()]:
            self.confirmation('Select node before making changes!')
            return
        ptype = p.parmTemplate().type().name()
        if ptype == 'FolderSet': return
        parm = (( node.path(),'Parm','{0}: {1}'.format(p.name(),p.eval()) ))
        self.addData(parm)
        
    # Select Change
    def selectChange(self,**kwargs):
        self.selectedNodes = []
        sel = hou.selectedNodes()
        if not sel: return

        for ind,node in enumerate(hou.selectedNodes()):
            if ind == 0:
                appendList = [nd for nd in node.parent().children() if nd.isGenericFlagSet(hou.nodeFlag.Display) or nd.isGenericFlagSet(hou.nodeFlag.Render)]
                appendList.append(node)
            else: appendlist = [node]

            for n in appendList:
                flagDict,valid = self.dictBuild(n)
                if valid: self.selectedNodes.append(( n.path(), flagDict))

                # Add to reset list
                if self.resetList:
                    if [item for item in self.resetList if item[0] == n]: continue
                self.resetList.append(( n, flagDict, self.parmdictBuild(n) ))
                
            
    def dictBuild(self,n):
        valid = 0
        flagDict = {}
        for f in self.accepted_flags:
            try:
                result = n.isGenericFlagSet(eval('hou.nodeFlag.'+f))
                flagDict[f] = result
                valid = 1
            except: pass
        return [flagDict,valid]
        
    def parmdictBuild(self,n):
        return {p.name():p.eval() for p in n.parmTuples()}
        
    def flagChange(self,**kwargs):#Flag Change
        node = kwargs["node"]
        if not [item for item in self.resetList if item[0].path() == node.path()]:
            self.confirmation('Select node before making changes!')
            return

        compare = [item for item in self.selectedNodes if item[0] == node.path()]
        if not compare: return
        path,dict = compare[0]
        
        flagDict = self.dictBuild(node)[0]
        
        for key, value in flagDict.items():
            if value != dict.get(key):
                self.addData((( node.path(),'Flag','{0}: {1}'.format(key,value) )))
        
        update = ((node.path(),flagDict))
        self.selectedNodes = [update if x[0]==update[0] else x for x in self.selectedNodes]
                
    def childChange(self,**kwargs):#Node added  
        self.callbackList(kwargs["child_node"])
        
    def getActions(self):
        return [i for i in self.recordList.selectedItems()]
        
    def recordDelete(self):
        for s in self.getActions():
            self.recorddata.remove((s.text(0),s.text(1),s.text(2)))
        self.populateRecord()
        
    def resetNodes(self):
        if self.resetList:
            if not hou.ui.displayMessage('Clear record data & reset parameters?', buttons=("Yes", "No"), close_choice=1, severity=hou.severityType.Message, title="Info"):
                if self.record.isChecked():
                    self.record.setChecked(0)
                    self.recordFunct()
                hou.takes.setCurrentTake(hou.takes.rootTake())
                for x,y,z in self.resetList:
                    [x.setGenericFlag(eval('hou.nodeFlag.'+k),v) for k,v in y.items()]
                    [x.parmTuple(k).set(v) for k,v in z.items()]
                        
                    self.takeInfo.setText('Reset %i node(s).'%len(self.resetList))
                        
                self.recordList.clear()
                self.resetList = []
                self.recorddata = []
            
    ## RECORDING ACTIONS END
        
    def closeEvent(self, event):
        self.setCallbacks()
        
    ## CREATE TAKES START
    
    def addTermFunct(self,term):
       b = self.takeName
       b.setText(b.text()+term)
    
    def createTakes(self,append):
        nfo = self.takeInfo
        self.recent = []
        tab = self.tabs.currentIndex()
        
        # Get Selected Takes
        selected = [hou.takes.findTake(t.text(0)) for t in self.getSelected()]
        if not selected:
            nfo.setText('No takes selected.')
            return
            
        # Naming
        name = self.takeName.text()
        if append:
            if not name:
                nfo.setText('No take Name.')
                return
            if re.search(r"([^a-z0-9_* ])", name.lower()):
                nfo.setText('Take name contains illegal characters.')
                return
            name = name.replace(' ','_')
        
        # Various Take Saves
        originalTake = hou.takes.currentTake()
        root = hou.takes.rootTake()
                
        # Method
        
        if tab == 0:#switch
            if not append: 
                nfo.setText('Mode only available for \'append takes\'.')
                return
            parm = hou.parmTuple(self.switchnode.text())
            try:
                node = parm.node()
            except:
                nfo.setText('Invalid switch parameter.')
                return
            inputs = node.inputs()
            ogval = parm.eval()
            if parm.parmTemplate().type().name() != 'Int':
                nfo.setText('Invalid parameter type - must be an integer.')
                return
            if not inputs:
                nfo.setText('Switch has no inputs.')
                return

            # Create Takes
            for take in selected:
                for ind, input in enumerate(inputs):
                    hou.takes.setCurrentTake(root)
                    parm.set((ind,))
                    takename = name.replace('*PARENT*',take.name())
                    if self.switchnamingmode.currentIndex(): methodType = str(ind+self.sstartnum.value()).zfill(self.spadding.value())
                    else: methodType = input.name()
                    takename = takename.replace('*METHOD*',methodType).replace(' ','_')
                    newtake = take.addChildTake(takename)
                    self.recent.append(newtake.name())
                    hou.takes.setCurrentTake(newtake)
                    newtake.addParmTuple(parm)
                        
            hou.takes.setCurrentTake(root)
            parm.set(ogval)
            nfo.setText('Created Takes.')
                        
        elif tab == 1:#menu
            if not append: 
                nfo.setText('Mode only available for \'append takes\'.')
                return
            parm = hou.parm(self.menunode.text())
            parmtuple = hou.parmTuple(self.menunode.text())
            try:
                node = parm.node()
            except:
                nfo.setText('Invalid menu parameter.')
                return
            ogval = parm.eval()
            try: items = parm.menuLabels()
            except:
                nfo.setText('Invalid parameter type - must be a menu.')
                return

            # Create Takes
            for take in selected:
                for ind, item in enumerate(items):
                    hou.takes.setCurrentTake(root)
                    parm.set(parm.menuItems()[ind])
                    takename = name.replace('*PARENT*',take.name())
                    if not self.menunamingmode.currentIndex(): methodType = item
                    else: methodType = str(ind+self.mstartnum.value()).zfill(self.padding.value())
                    takename = takename.replace('*METHOD*',methodType).replace(' ','_')
                    newtake = take.addChildTake(takename)
                    self.recent.append(newtake.name())
                    hou.takes.setCurrentTake(newtake)
                    newtake.addParmTuple(parmtuple)
                        
            hou.takes.setCurrentTake(root)
            parm.set(ogval)
            nfo.setText('Created Takes.')

        elif tab == 2:#attribute
            if not append: 
                nfo.setText('Mode only available for \'append takes\'.')
                return

            # Get UI Parms

            change = self.parm_change.text()
            prefix = self.prefix.text()
            at_node = self.attribute_node.text()
            at_type = self.attribute_type.currentText()
            at_name = self.attribute_name.currentText()

            # Validity Checks

            parmtuple = hou.parmTuple(change)
            change = hou.parm(change)
            ogval = change.eval()
            if not change:
                nfo.setText('Invalid change parm.')
                return

            if prefix.find('$ATTRIB') < 0:
                nfo.setText("Entry must contain '$ATTRIB'.")
                return

            at_node = hou.node(at_node)
            if not at_node:
                nfo.setText("Attribute node doesn't exist.")
                return

            if not at_name:
                nfo.setText("Select an attribute.")
                return

            # Get attribute information

            if self.attribute_name.currentIndex() == 0:
                if at_type == 'Point':
                    if at_node.geometry().points():
                        unique_vals = range(len(at_node.geometry().points()))

                    else:
                        nfo.setText("Node '%s' has no point geometry."%at_node.name())
                        return

                elif at_type == 'Primitive':
                    if at_node.geometry().prims():
                        unique_vals = range(len(at_node.geometry().prims()))

                    else:
                        nfo.setText("Node '%s' has no prim geometry."%at_node.name())
                        return

            else:
                unique_vals = [p.attribValue(at_name) for p in at_node.geometry().points()]
                unique_vals = set(unique_vals)
                unique_vals = list(unique_vals)

            # Create Takes

            for take in selected:
                for item in unique_vals:
                    hou.takes.setCurrentTake(root)

                    item = str(item)

                    change.set(prefix.replace('$ATTRIB',item))

                    takename = name.replace('*PARENT*',take.name())
                    takename = takename.replace('*METHOD*',item).replace(' ','_')

                    newtake = take.addChildTake(takename)

                    self.recent.append(newtake.name())
                
                    hou.takes.setCurrentTake(newtake)
                    newtake.addParmTuple(parmtuple)

            # Cleanup

            hou.takes.setCurrentTake(root)
            change.set(ogval)
            nfo.setText('Created Takes.')

        elif tab == 3:#record
            if not self.recorddata:
                nfo.setText('Missing recording data.')
                return
            
            if self.record.isChecked():
                self.record.setChecked(0)
                self.recordFunct()

            # Create Takes
            for take in selected:
                if append: 
                    takename = name.replace('*PARENT*',take.name())
                    takename = takename.replace('*METHOD*','')
                    take = take.addChildTake(takename)

                self.recent.append(take.name())
                hou.takes.setCurrentTake(take)

                for p,t,v in self.recorddata:
                
                    node = hou.node(p)
                    vname,value = v.split(': ',1)
                    ptuple = node.parmTuple(vname)

                    if t == 'Flag':
                        if vname == 'Bypass':
                            take.removeNodeBypassFlag(node)
                            take.addNodeBypassFlag(node)
                        elif vname == 'Display':
                            take.removeNodeDisplayFlag(node)
                            take.addNodeDisplayFlag(node)
                        elif vname == 'Render':
                            take.removeNodeRenderFlag(node)
                            take.addNodeRenderFlag(node)
                            
                    elif t == 'Parm':
                        if take.hasParmTuple(ptuple): take.removeParmTuple(ptuple)
                        take.addParmTuple(ptuple)

            self.resetNodes()
                
        hou.takes.setCurrentTake(originalTake)
        self.populate()
        self.selectTakes(self.recent)
    
    ## CREATE TAKES END
        
    ## ACTIONS START
    
    def copyNodes(self):
        # Get Selected Takes
        stakes = self.getSelected()
        if not stakes:
            self.actionInfo.setText('Select Takes.')
            return
            
        # Get selected node
        sel = hou.selectedNodes()
        hou.copyNodesToClipboard(sel)
        try: sel = sel[0]
        except:
            self.actionInfo.setText('Select a node to copy.')
            return
        location = sel.parent()
        pos = sel.position()
        
        # Get Take Parm Name
        cancel,input = self.txtInput('Take Parameter',('Ok','Cancel'),None,'take')
        if cancel: return
        if not input:
            self.actionInfo.setText('Parameter name cannot be empty.')
            return
        if re.search(r"([^a-z0-9_ ])", input.lower()):
            self.actionInfo.setText('Parameter name contains illegal characters.')
            return
        name = input.replace(' ','_')
        
        # Make Copies
        stakes.reverse()#Reverse so you can layout in the correct order
        for take in stakes:
            hou.pasteNodesFromClipboard(location)
            
            name = take.text(0)
            sel = hou.selectedNodes()[0]
            sel.parm(input).set(name)
            sel.setPosition(pos)
            name=name.replace(' ','_')
            name=name.replace('-','_')
            sel.setName(name,True)
            sel.setColor(hou.Color((.2,.2,.2)))
            sel.move((0,-2))
        
        self.actionInfo.setText('')
            
    ## ACTIONS END
            
    def menuindexChanged(self):
        self.padding.setEnabled(self.menunamingmode.currentIndex())
        self.mstartnum.setEnabled(self.menunamingmode.currentIndex())

    def switchindexChanged(self):
        self.spadding.setEnabled(self.switchnamingmode.currentIndex())
        self.sstartnum.setEnabled(self.switchnamingmode.currentIndex())

    def attribute_update(self):
        self.attribute_name.clear()
        node = self.attribute_node.text()

        node = hou.node(node)
        if not node:
            self.attribute_name.addItem("Invalid Node!")
            return

        if node.type().category().name() != 'Sop':
            self.attribute_name.addItem("SOP Required!")
            return

        geo = node.geometry()
        if not geo:
            return

        type = self.attribute_type.currentIndex()
        if type == 0:
            attriblist = geo.pointAttribs()
            add = ['ptnum']
        elif type == 1:
            attriblist = geo.primAttribs()
            add = ['primnum']
        elif type == 2:
            attriblist = geo.vertexAttribs()
        elif type == 3:
            attriblist = geo.globalAttribs()

        attriblist = add+[p.name() for p in attriblist]
        self.attribute_name.addItems(attriblist)


    def node_tree_on_select(self, widget, node, *args):
        input = args
        if input:
            if node:
                input = args[0]
            elif args[0]:
                input = args[0][0]
            else:
                return
            
            try:
                widget.setText(input.path())
            except:
                return

    ## HOUDINI UI POPUPS
    
    def txtInput(self,msg,Buttons,hlp,default):
        return hou.ui.readInput(msg, buttons=Buttons, default_choice=0, close_choice=1, help=hlp, title='Input',initial_contents=default)

    def confirmation(self,msg):
        hou.ui.displayMessage(msg, buttons=('OK',),default_choice=0, close_choice=0, title='Info')
        
    def _dropEvent(self, parm, node, event):
        if houVer > 17:
            if node:
                type = hou.qt.mimeType.nodePath
            else:
                type = hou.qt.mimeType.parmPath

            data = event.mimeData().data(type)
            if not data.isEmpty():
                text = str(data).split("\t")[0]
            else: return

        else:
            if event.mimeData().hasText():
                text = event.mimeData().text()
            else: return

        parm.clear()
        parm.setText(text)

    # ON CLOSE

    def onDeactivate(self):
        self.setCallbacks()