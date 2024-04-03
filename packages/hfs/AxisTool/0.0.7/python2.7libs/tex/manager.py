import hou,os,sys,time,ast,re,subprocess
from PySide2 import QtCore, QtUiTools, QtWidgets, QtGui
from PySide2.QtGui import *
from PySide2.QtWidgets import *
from PySide2.QtCore import *

class axisFileReplace(QDialog):
    def __init__(self, *args, **kwargs):

        count = 0
        dellist = []
        for p in hou.ui.paneTabs():
            if p.type().name() == 'PythonPanel':
                if p.activeInterface().name() == 'texManager':
                    count+=1
                    dellist.append(p)

        if count >1:
            dellist[0].close()

        super(axisFileReplace, self).__init__()
        
        # widget config
        self.setProperty("houdiniStyle", True)
        self._error_brush = QtGui.QBrush(QtGui.QColor(255, 0, 0))
        self._error_brush.setStyle(QtCore.Qt.SolidPattern)

        self._warn_brush = QtGui.QBrush(QtGui.QColor(255, 255, 0))
        self._warn_brush.setStyle(QtCore.Qt.SolidPattern)

        self._info_brush = QtGui.QBrush(QtGui.QColor(0, 255, 0))
        self._info_brush.setStyle(QtCore.Qt.SolidPattern)

        self._msg_brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        self._msg_brush.setStyle(QtCore.Qt.SolidPattern)
        
        self.setWindowTitle("File Replace")
        self.resize(1200, 700)
        self.setParent(hou.ui.mainQtWindow(), QtCore.Qt.Window)
       
        sh = hou.ui.qtStyleSheet()
        self.setStyleSheet(sh)

        if sys.platform != 'win32':
            self.setWindowFlags(QtCore.Qt.Window | Qt.WindowStaysOnTopHint)
       
        layout = QVBoxLayout()    
        layout.setSpacing(5)
        layout.setSizeConstraint(QLayout.SetMinimumSize)
               
        groupBox = QGroupBox("Networks")
        layout.addWidget(groupBox)
       
        layout2 = QVBoxLayout()    
        layout2.setSpacing(5)
        layout2.setSizeConstraint(QLayout.SetMinimumSize)
        groupBox.setLayout(layout2)
        
        self.network = QPushButton('')
        self.network.setStyleSheet("QPushButton { text-align: left; }")
        self.netmenu = hou.qt.Menu()
        self.netmenu.addAction('All', lambda: self.menuclicked(0))
        self.netmenu.addSeparator()
        self.netmenu.addSeparator()
        self.menulist = []
        for n in hou.node('/').children():
            self.oneRow = self.netmenu.addAction(n.path(), lambda: self.menuclicked(1))
            self.oneRow.setCheckable(True)

            self.menulist.append(self.oneRow)

        self.network.setMenu(self.netmenu)

        #self.network.addItems([n.path() for n in hou.node('/').children()])
        #self.network.insertSeparator(1)
        #self.network.setCurrentIndex(-1)
        #self.network.currentIndexChanged.connect(lambda:self.populate(0))
        layout2.addWidget(self.network)
        
        groupBox = QGroupBox("Files")
        layout.addWidget(groupBox)
       
        layout2 = QVBoxLayout()    
        layout2.setSpacing(5)
        layout2.setSizeConstraint(QLayout.SetMinimumSize)
        groupBox.setLayout(layout2)
       
        line1 = QHBoxLayout()
        line1.setSpacing(5)
        layout2.addLayout(line1)

        self.table = QTableWidget()
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)

        minSize = 65
       
        #Set Column Names
        headerLabels = ["Found", "Path", "Nodes"]
        self.table.setColumnCount(len(headerLabels))
        self.table.setHorizontalHeaderLabels(headerLabels)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.table.setAlternatingRowColors(True)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().hide()
        self.table.setShowGrid(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSortingEnabled(1)
        self.table.doubleClicked.connect(self.sel_nodes)
       
        # Set Column Sizes
        self.table.setColumnWidth(1,700)
        self.table.resizeRowsToContents()
        line1.addWidget(self.table)
       
        line2 = QHBoxLayout()
        line2.setSpacing(5)
        layout.addLayout(line2)
       
        self.spacer = QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Minimum)
        line2.addItem(self.spacer)
       
        self.info = QLabel()
        self.info.setMinimumHeight(30)
        line2.addWidget(self.info)        
       
        ## Replace
       
        # Initialize tab screen
        self.tabs = QTabWidget()
        self.tab1 = QWidget()
        self.tab2 = QWidget()
        self.tab3 = QWidget()
        self.tabs.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
       
        # Add tabs
        self.tabs.addTab(self.tab1,"Replace")
        self.tabs.addTab(self.tab2,"Filter")
        self.tabs.addTab(self.tab3,"Replace Term")
       
        layout.addWidget(self.tabs)
       
        ##
       
       
        # Layout
        layout3 = QVBoxLayout()
        layout3.setSpacing(5)
        layout3.setSizeConstraint(QLayout.SetMinimumSize)
        self.tab1.setLayout(layout3)
       
        line2 = QHBoxLayout()
        line2.setSpacing(5)
        layout3.addLayout(line2)
        groupBox.setLayout(layout3)
       
        # path
        self.label = QLabel('File Path:')
        self.label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.label.setMinimumWidth(minSize)
        line2.addWidget(self.label)
        self.path = QLineEdit('')
        line2.addWidget(self.path)
                       
        # path
        def onFileSelected(file_path):
            self.pathr = file_path
           
            if self.pathr:
                self.path.setText(self.pathr)

        self.icon = hou.qt.FileChooserButton()
        self.icon.setFileChooserTitle("Select Textures to Import")
        self.icon.setFileChooserMode(hou.fileChooserMode.Read)
        self.icon.setFileChooserIsImageChooser(True)
        self.icon.setFileChooserMultipleSelect(False)
        self.icon.fileSelected.connect(onFileSelected)
        line2.addWidget(self.icon)
       
        layout3.addItem(QSpacerItem(0, 0, QSizePolicy.Minimum, QSizePolicy.Expanding))

        line3 = QHBoxLayout()
        line3.setSpacing(5)
        layout3.addLayout(line3)

        line3.addItem(QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Minimum))

        # replace button
        self.replace = QPushButton('Options')
        self.replace.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        line3.addWidget(self.replace)
               
        # Menu Button
        self.menu = hou.qt.Menu()
        self.menu.addAction('Replace Selected                 ', self.replace_links,'Ctrl+Return')
        self.menu.addSeparator()
        self.menu.addAction('Open Folder', self.open_folder, 'Ctrl+o')
        self.menu.addAction('Copy Path', self.copy_path, 'Ctrl+c')
        self.menu.addAction('Select Nodes', self.sel_nodes, 'Ctrl+a')
        self.menu.addAction('Clear Selection', self.clear_sel, 'Ctrl+d')
        self.menu.addSeparator()
        self.menu.addAction('Full Refresh', lambda: self.populate(0), 'Ctrl+r')
        #self.menu.addAction('Quick Refresh', lambda: self.populate(1))
        self.menu.addSeparator()
        self.oneRow = self.menu.addAction('One Row per Node', lambda: self.populate(1), 'Ctrl+e')
        self.oneRow.setCheckable(True)
       
        self.replace.setMenu(self.menu)
       
        ## Filter

       
        # Layout
        layout3 = QVBoxLayout()
        layout3.setSpacing(5)
        layout3.setSizeConstraint(QLayout.SetMinimumSize)
        self.tab2.setLayout(layout3)
       
        line2 = QHBoxLayout()
        line2.setSpacing(5)
        layout3.addLayout(line2)
        groupBox.setLayout(layout3)
       
        # starts with
        self.label = QLabel('Starts with:')
        self.label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignTop)
        self.label.setMinimumWidth(minSize)
        line2.addWidget(self.label)
        self.filter = QLineEdit('')
        self.filter.textChanged.connect(self.filterFunc)
        line2.addWidget(self.filter)
        
        
        line2 = QHBoxLayout()
        line2.setSpacing(5)
        layout3.addLayout(line2)
        groupBox.setLayout(layout3)
        
        # find term
        self.label = QLabel('Find term:')
        self.label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignTop)
        self.label.setMinimumWidth(minSize)
        line2.addWidget(self.label)
        self.term = QLineEdit('')
        self.term.textChanged.connect(self.filterFunc)
        line2.addWidget(self.term)
       
        layout3.addItem(QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Expanding))
       
        ## Term Replace
       
        # Layout
        layout3 = QVBoxLayout()
        layout3.setSpacing(5)
        layout3.setSizeConstraint(QLayout.SetMinimumSize)
        self.tab3.setLayout(layout3)
       
        line2 = QHBoxLayout()
        line2.setSpacing(5)
        layout3.addLayout(line2)
        groupBox.setLayout(layout3)
       
        # path
        self.label = QLabel('Find:')
        self.label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.label.setMinimumWidth(minSize)
        line2.addWidget(self.label)
       
        self.find = QLineEdit('')
        line2.addWidget(self.find)
       
        line2 = QHBoxLayout()
        line2.setSpacing(5)
        layout3.addLayout(line2)
        groupBox.setLayout(layout3)
       
        self.label = QLabel('Replace:')
        self.label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.label.setMinimumWidth(minSize)
        line2.addWidget(self.label)
       
        self.replace = QLineEdit('')
        line2.addWidget(self.replace)
       
        line3 = QHBoxLayout()
        line3.setSpacing(5)
        layout3.addLayout(line3)

        self.replaceInfo = QLabel()
        line3.addWidget(self.replaceInfo)
       
        line3.addItem(self.spacer)

        # replace button
        self.replacebut = QPushButton('Replace')
        self.replacebut.clicked.connect(self.replaceFunc)
        self.replacebut.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        line3.addWidget(self.replacebut)
       
        self.setLayout(layout)
       
        ## First Load
        #self.populate(0)
        self.info.setText('Select a network.')
       
    def load(self,quick):
        list1 = []
        nodelist = []
        if self.netlist == 'All':
            searchnetwork = hou.node('/').allSubChildren(recurse_in_locked_nodes=False)
        elif self.netlist:
            searchnetwork = []
            for n in self.netlist:
                searchnetwork += hou.node(n).allSubChildren(recurse_in_locked_nodes=False)
        if not quick:
            for n in searchnetwork:
                for p in n.parms():
                    if 'string_type=FileReference' in repr(p.parmTemplate().type) and p.eval() and not p.isDisabled() and '/' in p.eval():
                        try:
                            list1.append(p.unexpandedString())
                            nodelist.append(n.path())
                        except:
                            pass

        else:
            for cRow in range(self.table.rowCount()):
                nodes = self.table.item(cRow,2).text().split(', ')
                files = self.table.item(cRow,1).text()
               
                for n in nodes:
                    list1.append(files)
                    nodelist.append(n)

        return list1,nodelist

    def menuclicked(self,mode):
        if mode:
            self.netlist = []

            for i in self.netmenu.children():
                if i.text().startswith('/'):
                    if i.isChecked():
                        self.netlist.append(i.text())

            if self.netlist:
                self.network.setText(', '.join(self.netlist))
            else:
                self.network.setText('')
        else:
            for i in self.netmenu.children():
                if i.text().startswith('/'):
                    i.setChecked(0)
            self.netlist = 'All'
            self.network.setText('All')

        self.populate(0)
       
    def populate(self,quick):
        if not self.netlist: return
        
        lists = self.load(quick)
        file_list = lists[0]
        node_list = lists[1]

        self.items_clear()

        if file_list:
            row = -1
            filecount = 0

            for ind,f in enumerate(file_list):
                found = 0
                filecount += 1
                if self.oneRow.isChecked():
                    found = 0
                else:
                    for i in range(self.table.rowCount()):
                        if self.table.item(i,1).text() == f:
                            found = 1
                            filecount -= 1
                            fRow = i
                            break

                if not found:  
                    row += 1
                    fRow = row
                    self.table.insertRow(row)
               
                    # File Path
                    item = QTableWidgetItem(f)
                    item.setToolTip(f)
                    self.table.setItem(row, 1, item)
                   
                    # Node Info
                    path = node_list[ind]

                    # Found Check Box
                    exist = self.existCheck(f)
                    self.chk = QCheckBox()
                    self.chk.setEnabled(False)
                    if exist:
                        self.chk.toggle()
                    self.chk.toggled.connect(self.prevent_toggle)
                    cell_widget = QWidget()
                    lay_out = QHBoxLayout(cell_widget)
                    lay_out.addWidget(self.chk)
                    lay_out.setAlignment(Qt.AlignCenter)
                    lay_out.setContentsMargins(5,0,0,0)
                    cell_widget.setLayout(lay_out)
                    self.chk = QCheckBox()
                    self.table.setCellWidget(row, 0, cell_widget)
                   
                else:
                    # Node Info
                    path = '%s, %s' % (self.table.item(fRow,2).text(),node_list[ind])
                             
                # Node Name
                item = QTableWidgetItem(path)
               
                tipfix = path.split(', ')
                if len(tipfix)>1:
                    plist = []
                    for i,n in enumerate(tipfix):
                        if i+1 == len(tipfix):
                            end = ''
                        else:
                            end = ', '
                        if i%3 == 2:
                            plist.append('%s%s\n'%(n,end))
                        else:
                            plist.append('%s%s'%(n,end))
                else:
                    plist = [tipfix[0]]
                item.setToolTip(''.join(plist))
                self.table.setItem(fRow, 2, item)
                
            self.dict = [(self.table.cellWidget(cRow,0).children()[1].isChecked(),self.table.item(cRow,1).text(),self.table.item(cRow,2).text()) for cRow in range(self.table.rowCount())]
            self.filterFunc()
            details = self.info_details()
            self.info.setText('%s files found in %s nodes.'%( details[1], details[0] ))
        
        else:
            self.info.setText('0 files found in 0 nodes.')
               
    def prevent_toggle(self):
        b=1
   
    def existCheck(self,file):
        checkfile = hou.expandString(file)
        if os.path.isfile(checkfile):
            exist = 1
        else:
            exist = 0
       
        return exist
       
    def items_clear(self):
        rowcount = self.table.rowCount()
        for row in range(rowcount):
            self.table.removeRow(0)
   
    def replace_links(self):
        rows = []
        for idx in self.table.selectedIndexes():
            rows.append(idx.row())
           
        for cRow in rows:
            if cRow != -1:
                ogfile = self.table.item(cRow,1).text()
                node_list = self.table.item(cRow,2).text().split(', ')
                newfile = self.path.text()
                exist = self.existCheck(newfile)
               
                # Change the Node Parms
                for n in node_list:
                    n = hou.node(n)
                    for p in n.parms():
                        if 'string_type=FileReference' in repr(p.parmTemplate().type) and p.eval():
                            if p.unexpandedString() == ogfile:
                                p.set(newfile)
               
                # Edit Table
                self.table.cellWidget(cRow,0).children()[1].setChecked(exist)
                self.table.item(cRow,1).setText(newfile)
                self.table.item(cRow,1).setToolTip(newfile)
               
        details = self.info_details()
        self.info.setText('%s files found in %s nodes.'%( details[1], details[0] ))
        self.populate(1)
       
    def open_folder(self):
        rows = []
        for idx in self.table.selectedIndexes():
            rows.append(idx.row())
           
        for cRow in rows:
            if cRow != -1:
                file = hou.expandString(self.table.item(cRow,1).text())
                exist = self.existCheck(file)
                file = file.rsplit('/', 1)[0]
               
                platform = sys.platform
                if platform == "win32":
                    os.startfile(file)
                elif platform == "darwin":
                    subprocess.Popen(["open", file])
                else:
                    subprocess.Popen(["xdg-open", file])
               
    def sel_nodes(self):
        rows = []
        for idx in self.table.selectedIndexes():
            rows.append(idx.row())
           
        for i, cRow in enumerate(rows):
            if cRow != -1:
                node_list = self.table.item(cRow,2).text().split(', ')
                for ind,n in enumerate(node_list):
                    n = hou.node(n)
                    if i == 0 and ind == 0:    n.setSelected(True, True, False)
                    else:   n.setSelected(True, False, False)
                   
    def clear_sel(self):
        rows = []
        for idx in self.table.selectedIndexes():
            rows.append(idx.row())
           
        for i, cRow in enumerate(rows):
            self.table.selectRow(cRow)
           
    def info_details(self):
        node_list = []
        file_list = []
        rowcount = self.table.rowCount()
        for cRow in range(rowcount):
            file_list.append(self.table.item(cRow,1).text())
            for n in self.table.item(cRow,2).text().split(', '):
                node_list.append(n)
       
        node_list = list( dict.fromkeys(node_list) )
        file_list = list( dict.fromkeys(file_list) )
        return str(len(node_list)),str(len(file_list))
       
    def filterFunc(self):
        self.items_clear()
       
        row = -1
        for x,y,z in self.dict:
            if y.startswith(self.filter.text()) or not self.filter.text():
                if y.find(self.term.text())!=-1 or not self.term.text():
                    row += 1
                   
                    self.table.insertRow(row)
               
                    # File Path
                    item = QTableWidgetItem(y)
                    item.setToolTip(y)
                    self.table.setItem(row, 1, item)
                   
                    # Node Path(s)                
                    item = QTableWidgetItem(z)
                   
                    tipfix = z.split(', ')
                    if len(tipfix)>1:
                        plist = []
                        for i,n in enumerate(tipfix):
                            if i+1 == len(tipfix):
                                end = ''
                            else:
                                end = ', '
                            if i%3 == 2:
                                plist.append('%s%s\n'%(n,end))
                            else:
                                plist.append('%s%s'%(n,end))
                    else:
                        plist = [tipfix[0]]
                    item.setToolTip(''.join(plist))
                    self.table.setItem(row, 2, item)
           
                    # Found Check Box
                    self.chk = QCheckBox()
                    self.chk.setEnabled(False)
                    if x:   self.chk.toggle()
                    self.chk.toggled.connect(self.prevent_toggle)
                    cell_widget = QWidget()
                    lay_out = QHBoxLayout(cell_widget)
                    lay_out.addWidget(self.chk)
                    lay_out.setAlignment(Qt.AlignCenter)
                    lay_out.setContentsMargins(5,0,0,0)
                    cell_widget.setLayout(lay_out)
                    self.chk = QCheckBox()
                    self.table.setCellWidget(row, 0, cell_widget)
               
        details = self.info_details()
        self.info.setText('%s files found in %s nodes.'%( details[1], details[0] ))
                              
    def replaceFunc(self):
        if self.replace.text() and self.find.text():
            rows = []
            for idx in self.table.selectedIndexes():
                rows.append(idx.row())

            if not rows:
                self.replaceInfo.setText('No path(s) selected.')
                return
               
            find = self.find.text()
            replace = self.replace.text()
            replaced = 0
           
            for cRow in rows:
                if cRow != -1:
                    ogfile = self.table.item(cRow,1).text()
                    node_list = self.table.item(cRow,2).text().split(', ')
                    newfile = ogfile.replace(find,replace)
                    exist = self.existCheck(newfile)
                   
                    # Change the Node Parms
                    with hou.undos.group('Replace Terms'):
                        for n in node_list:
                            n = hou.node(n)
                            for p in n.parms():
                                if 'string_type=FileReference' in repr(p.parmTemplate().type) and p.eval():
                                    if p.unexpandedString() == ogfile and find in p.unexpandedString():
                                        p.set(newfile)
                                        replaced += 1
                                   
                    # Edit Table
                    self.table.cellWidget(cRow,0).children()[1].setChecked(exist)
                    self.table.item(cRow,1).setText(newfile)
                    self.table.item(cRow,1).setToolTip(newfile)
                    self.replaceInfo.setText('Replaced %i path(s).'%(replaced))

            self.populate(1)
        else:
            self.replaceInfo.setText('Input a term to replace.')

    def copy_path(self):
        rows = []
        for idx in self.table.selectedIndexes():
            rows.append(idx.row())

        try: 
            if rows[0] == -1: return
        except: return

        hou.ui.copyTextToClipboard(self.table.item(rows[0],1).text())