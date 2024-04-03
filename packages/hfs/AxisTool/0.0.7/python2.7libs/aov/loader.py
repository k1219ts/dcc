import os,getpass,glob,ast,re,time,hou,sys

from PySide2 import QtCore, QtUiTools, QtWidgets, QtGui
from PySide2.QtGui import *
from PySide2.QtWidgets import *
from PySide2.QtCore import *

user = getpass.getuser()
BUTTON_ICON_SIZE = hou.ui.scaledSize(24)

class loadAOVs(QDialog):
    def __init__(self, *args, **kwargs):
        super(loadAOVs, self).__init__()
        self.setWindowTitle("Load AOVs")
        self.resize(600, 550)
        self.setParent(hou.ui.mainQtWindow(), QtCore.Qt.Window)
        self.pref = '%s/aov.pref'%hou.expandString('$HOUDINI_USER_PREF_DIR')
        
        if os.path.isfile(self.pref):
            f = file(self.pref,'r')
            with open(self.pref, 'r') as f:
                self.defaults = ast.literal_eval(f.read())
       
        sh = hou.ui.qtStyleSheet()
        self.setStyleSheet(sh)

        if sys.platform != 'win32':
            self.setWindowFlags(QtCore.Qt.Window | Qt.WindowStaysOnTopHint)
       
        layout = QVBoxLayout()    
        layout.setSpacing(5)
        layout.setSizeConstraint(QLayout.SetMinimumSize)
                
        # Initialize tab screen
        self.tabs = QTabWidget()
        self.tab1 = QWidget()
        self.tab2 = QWidget()
        self.tab3 = QWidget()
        self.tabs.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        
        #self.tabs.currentChanged.connect(self.populate)
        
        # Add tabs
        self.tabs.addTab(self.tab1, "Load")
        self.tabs.addTab(self.tab2, "Save")
        self.tabs.addTab(self.tab3, "Pref")
        
        layout.addWidget(self.tabs)
        
        layout2 = QVBoxLayout()
        layout2.setSpacing(15)
        layout2.setSizeConstraint(QLayout.SetMinimumSize)
        self.tab1.setLayout(layout2)

        ## Render Engine
        
        linebox = QHBoxLayout()
        linebox.setSpacing(5)
        layout2.addLayout(linebox)
        
        groupBox = QGroupBox("Engine")
        linebox.addWidget(groupBox)
        
        # Layout
        layout3 = QVBoxLayout()
        layout3.setSpacing(5)
        layout3.setSizeConstraint(QLayout.SetMinimumSize)
        
        line2 = QHBoxLayout()
        line2.setSpacing(5)
        layout3.addLayout(line2)
        groupBox.setLayout(layout3)
        
        self.engine = QComboBox()
        
        self.engine.currentIndexChanged.connect(self.engineFunct)
        line2.addWidget(self.engine)
        
        ## User
        
        groupBox = QGroupBox("User")
        linebox.addWidget(groupBox)
        
        # Layout
        layout3 = QVBoxLayout()
        layout3.setSpacing(5)
        layout3.setSizeConstraint(QLayout.SetMinimumSize)
        
        line2 = QHBoxLayout()
        line2.setSpacing(5)
        layout3.addLayout(line2)
        groupBox.setLayout(layout3)
        
        self.user = QComboBox()
        self.user.currentIndexChanged.connect(self.populate)
        line2.addWidget(self.user)
        
        ### AOV Select
        
        groupBox = QGroupBox("AOVs")
        layout2.addWidget(groupBox)
        
        # Layout
        layout3 = QVBoxLayout()
        layout3.setSpacing(5)
        layout3.setSizeConstraint(QLayout.SetMinimumSize)
        
        line2 = QHBoxLayout()
        line2.setSpacing(5)
        layout3.addLayout(line2)
        groupBox.setLayout(layout3)
                
        line1 = QHBoxLayout()
        line1.setSpacing(5)
        groupBox.setLayout(line1)

        self.table = QTableWidget()
        self.table.itemClicked.connect(self.selected)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)

        minSize = 65
        
        #Set Column Names
        headerLabels = ["AOV Name","Date"]
        self.table.setColumnCount(len(headerLabels))
        self.table.setHorizontalHeaderLabels(headerLabels)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.table.setAlternatingRowColors(True)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().hide()
        self.table.horizontalHeader().hide()
        self.table.setShowGrid(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        
        # Set Column Sizes
        for i in range(0):
            self.table.horizontalHeader().setSectionResizeMode(i, QHeaderView.Stretch)
        line2.addWidget(self.table)
        
        layout2.addWidget(hou.qt.Separator())
        
        line1 = QHBoxLayout()
        line1.setSpacing(5)
        layout2.addLayout(line1)
        
        self.info = QLabel('')
        line1.addWidget(self.info)
        
        self.spacer = QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Minimum)
        line1.addItem(self.spacer)
        
        self.delete = QPushButton('Delete')
        self.delete.clicked.connect(self.deleteFunct)
        line1.addWidget(self.delete)
        
        self.load = QPushButton('Load')
        self.menu = hou.qt.Menu()
        self.menu.addAction('Replace AOVs                 ', lambda: self.loadFunct(0))
        self.menu.addAction('Add to Existing AOVs', lambda: self.loadFunct(1))
        self.load.setMenu(self.menu)
        line1.addWidget(self.load)
                
        
        ### SAVE
        
        layout2 = QVBoxLayout()
        layout2.setSpacing(15)
        layout2.setSizeConstraint(QLayout.SetMinimumSize)
        self.tab2.setLayout(layout2)
                
        groupBox = QGroupBox("Name")
        layout2.addWidget(groupBox)
        
        line1 = QHBoxLayout()
        line1.setSpacing(5)
        groupBox.setLayout(line1)
        
        layout2.addItem(QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Expanding))
                
        self.aovname = QLineEdit()
        line1.addWidget(self.aovname)
        
        layout2.addWidget(hou.qt.Separator())
        
        line1 = QHBoxLayout()
        line1.setSpacing(5)
        layout2.addLayout(line1)
                
        self.info2 = QLabel('')
        line1.addWidget(self.info2)
        
        self.spacer = QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Minimum)
        line1.addItem(self.spacer)
        
        self.save = QPushButton('Save')
        self.save.clicked.connect(self.saveFunct)
        line1.addWidget(self.save)
                
        self.setLayout(layout)
        
        self.tabs.blockSignals(False)
        
        ### PREF
        
        layout2 = QVBoxLayout()
        layout2.setSpacing(15)
        layout2.setSizeConstraint(QLayout.SetMinimumSize)
        self.tab3.setLayout(layout2)
                
        groupBox = QGroupBox("AOVs Location")
        layout2.addWidget(groupBox)
        
        line1 = QHBoxLayout()
        line1.setSpacing(5)
        groupBox.setLayout(line1)
        
        layout2.addItem(QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Expanding))
                
        self.aovloc = QLineEdit()
        line1.addWidget(self.aovloc)

        # path
        def onFileSelected(file_path):
            file_path = file_path.replace('\\','/')

            if file_path.endswith('/'):
                file_path = file_path[:-1]

            file_path = hou.expandString(file_path)

            if os.path.isdir(file_path):
                self.aovloc.setText(file_path)
            else:
                self.info3.setText("Folder doesn't exist.")
                
        self.explorer = hou.qt.FileChooserButton()
        self.explorer.setFileChooserTitle("Set AOV Location")
        self.explorer.setFileChooserMode(hou.fileChooserMode.Read)
        self.explorer.setFileChooserIsImageChooser(True)
        self.explorer.setFileChooserMultipleSelect(False)
        self.explorer.fileSelected.connect(onFileSelected)
        self.explorer.setFileChooserFilter(hou.fileType.Directory)
        self.explorer
        line1.addWidget(self.explorer)
        
        layout2.addWidget(hou.qt.Separator())
        
        line1 = QHBoxLayout()
        line1.setSpacing(5)
        layout2.addLayout(line1)
                
        self.info3 = QLabel('')
        line1.addWidget(self.info3)
        
        self.spacer = QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Minimum)
        line1.addItem(self.spacer)
        
        self.save = QPushButton('Use new location')
        self.save.clicked.connect(self.startup)
        line1.addWidget(self.save)
                
        self.setLayout(layout)
        
        ### Startup Calls
        
        try:
            self.aovloc.setText(self.defaults[2])
        except: pass
        
        self.startup()
        
        self.loadDefaults()
        
    def startup(self):
        
        # Find AOV Folder
        if os.path.isdir(self.aovloc.text()):
            self.path = self.aovloc.text()
        else: 
            self.path = '%s/AOVs'%hou.expandString('$HOUDINI_USER_PREF_DIR')
            if not os.path.isdir(self.path): 
                print 'AOV folder "%s" wasn\'t found - making AOV folder in default Houdini directory folder.'%self.path
        
        self.path = self.path.replace('\\','/')

        if not os.path.isdir(self.path): 
            try:
                os.makedirs(self.path)
            except:
                return
        
        self.aovloc.setText(self.path)
                
        engines = os.listdir(self.path)
        
        self.engine.setCurrentIndex(-1)
        self.engine.clear()
        self.user.setCurrentIndex(-1)
        self.user.clear()
        
        ## Populate Engine Combo Box
        if 'Redshift' in engines:
            try:
                self.engine.addItem(QIcon(hou.qt.Icon("ROP_Redshift_ROP", BUTTON_ICON_SIZE, BUTTON_ICON_SIZE)),'Redshift')
            except: pass
        if 'Arnold' in engines:
                try:
                    self.engine.addItem(QIcon(hou.qt.Icon("ROP_arnold", BUTTON_ICON_SIZE, BUTTON_ICON_SIZE)),'Arnold')
                except: pass
        if 'Mantra' in engines:
            try:
                self.engine.addItem(QIcon(hou.qt.Icon("ROP_ifd", BUTTON_ICON_SIZE, BUTTON_ICON_SIZE)),'Mantra')
            except: pass
        
    def populate(self):
        self.items_clear()
        engine = self.engine.currentText()
        user = self.user.currentText()
        self.location = '%s/%s/%s'%(self.path,engine,user)
        ## Populate Table
        
        os.chdir(self.location)
        for i,file in enumerate(glob.glob("*.aov")):
        
            self.table.insertRow(i)
            
            item = QTableWidgetItem(file.rsplit('.',1)[0])
            self.table.setItem(i, 0, item)
            
            item = QTableWidgetItem(time.strftime('%m/%d/%Y %H:%M:%S %p', time.gmtime(os.path.getmtime(self.location + '/' + file))))
            self.table.setItem(i, 1, item)
            
        self.info.setText('')
    
    def items_clear(self):
        rowcount = self.table.rowCount()
        [self.table.removeRow(0) for row in range(rowcount)]
            
    def engineFunct(self):
        self.items_clear()

        self.user.clear()
                
        users = '%s/%s'%(self.path,self.engine.currentText())
        
        for f in os.listdir(users):
            if os.path.isdir('%s/%s'%(users,f)):
                self.user.addItem(f)
                
        self.user.setCurrentIndex(-1)
        
        ## Set Info
        
        if self.user.count() == 0:
            self.info.setText('No user folders found.')
        else:
            self.info.setText('')
        
    def deleteFunct(self):
        cRow = self.table.currentRow()
        if cRow != -1:
            if self.info.text() == 'Click delete again to confirm.':
                aovfile = self.getSelected()
                user = self.user.currentText()
                engine = self.engine.currentText()
                path = '%s/%s/%s/%s'%(self.path, engine, user, aovfile)
                
                if os.path.isfile(path):
                    os.remove(path)
                    self.info.setText('%s Deleted.'%aovfile)
                else:
                    self.info.setText('File doesn\'t exist.')
                
                self.populate()
            else:
                self.info.setText('Click delete again to confirm.')
                
        else:
            self.info.setText('AOV not selected.')
            
    def selected(self):
        cRow = self.table.currentRow()
        if cRow != -1:
            aovfile = self.getSelected()
            user = self.user.currentText()
            engine = self.engine.currentText()
            path = '%s/%s/%s/%s'%(self.path, engine, user, aovfile)
            
            with open(path) as f:
                list = ast.literal_eval(f.read())
                
            self.info.setText('%s contains %s definitions.'%(aovfile,list[0][1]))
            
    def getSelected(self):
        return '%s.aov'%self.table.item(self.table.currentRow(), 0).text()
            
    def loadFunct(self,mode):
        sel = hou.selectedNodes()
        cRow = self.table.currentRow()
        if cRow != -1 and sel:
            # Get AOV File and Read
            aovfile = self.getSelected()
            user = self.user.currentText()
            engine = self.engine.currentText()
            path = '%s/%s/%s/%s'%(self.path,engine,user,aovfile)
            count = 0
            ropcount = 0
            with open(path) as f:
                list = ast.literal_eval(f.read())
            
            with hou.undos.group("Load AOV's"):
                for n in sel:
                    if self.engine.currentIndex() > -1:
                        
                        ## Check for invalid selection
                        
                        if n.type().category().name() != 'Driver':
                            continue
                            
                        seltype = n.type().name()
                        if engine == 'Redshift':
                            if seltype not in ['Redshift_AOVs','Redshift_ROP']:
                                continue
                        elif engine == 'Arnold':
                            if seltype not in ['arnold']:
                                continue
                        elif engine == 'Mantra':
                            if seltype not in ['ifd']:
                                continue
                                
                        ## Add to current AOV count or reset

                        if mode:#Add to AOVs
                            add = int( list[0][1] )
                            aovcount = int( n.evalParm(list[0][0]) )
                            n.parm(list[0][0]).set(aovcount+add)
                        else:
                            n.parm(list[0][0]).set(list[0][1])
                        
                        ropcount+=1
                        
                        ## Set Parameters
                        
                        for p,v in list[1:]:
                            if mode:#Add
                                parm,num = re.split(r'(\d+)', p)[:-1]
                                num = str( int(num) + aovcount )
                                
                                p = parm + num

                            n.parm(p).set(v)
                            count+=1
                                
                self.info.setText("AOV(s) set in %i ROP(s) set."%(ropcount))
                return

        else:
            self.info.setText('Select ROP & AOV')
            return
        
        self.info.setText('Invalid node selected.')
            
    def saveFunct(self):

        ## Start Checks
        
        name = self.aovname.text()
        
        if name:
            if re.search(r"([^a-z0-9_ ])", name.lower()):
                self.info2.setText('Name contains illegal character.')
                return
        else:
            self.info2.setText('No Name Set.')
            return
            
        name = name.replace(' ','_')
            
        try:
            sel = hou.selectedNodes()[0]
        except:
            self.info2.setText('ROP not selected.')
            return
        
        if sel.type().category().name() != 'Driver':
            self.info2.setText('Only RS, Arnold & Mantra ROPs supported.')
            return
            
        ropname = sel.type().name()
        if ropname in ['Redshift_ROP','Redshift_AOVs']:                
            aovparm = sel.parm('RS_aov')
            engine = 'Redshift'
        elif ropname == 'arnold':                
            aovparm = sel.parm('ar_aovs')
            engine = 'Arnold'
        elif ropname == 'ifd':  
            aovparm = sel.parm('vm_numaux')
            engine = 'Mantra'
        #elif ropname == 'Octane_ROP':#Octane AOV support is dire
        #    aovparm = sel.parm('vm_numaux')
        #    engine = 'Octane'
            
        else:
            self.info2.setText('Only RS, Arnold & Mantra ROPs supported.')
            return
            
        if int( aovparm.eval() ) == 0:
            self.info2.setText('No AOVs found.')
            return
            
        ## End Checks
            
        
        ## Get AOVs
        aovlist = [((p.name(),p.eval())) for p in aovparm.multiParmInstances()]
        aovlist = [((aovparm.name(), aovparm.eval()))] + aovlist    
        
        # Exist Check
        loc = '%s/%s/%s'%(self.path,engine,user)
        
        if os.path.isdir(loc):
            if '%s.aov'%name in os.listdir(loc):
                self.info2.setText('AOV name already exists.')
                return
                
        # Write
        if not os.path.isdir(loc):
            os.makedirs(loc)
        
        a = file('%s/%s.aov'%(loc,name), 'w')
        a.write(repr(aovlist))

        self.info2.setText("AOV's written.")

        self.startup()
        
    def hideEvent(self, event):
        self.setParent(None)
        self.saveChanges()
        
    def closeEvent(self, event):
        self.setParent(None)
        self.saveChanges()
        
    def saveChanges(self):
        item = [self.engine.currentText(),self.user.currentText(),self.aovloc.text()]
        a = file(self.pref, 'w')
        a.write(repr(item))

    def loadDefaults(self):
        if self.defaults:
        
            try:
                self.engine.setCurrentText(self.defaults[0])
            except: pass
            try:
                self.user.setCurrentText(self.defaults[1])
            except: pass