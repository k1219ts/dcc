import hou,os,sys,ast,re,PySide2,assignment,time
from math import sqrt,ceil
from PySide2 import QtCore, QtUiTools, QtWidgets, QtGui
from PySide2.QtGui import *
from PySide2.QtWidgets import *
from PySide2.QtCore import *

reload(assignment)

class matAssign(QDialog):
    def __init__(self, set_list, geo_selection):
        super(matAssign, self).__init__()
        self.setWindowTitle("Assign Material")

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
        winsize = ((400,300))
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
        self.name.setPlaceholderText('Leave Blank for Defaults')
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
        
        label = QLabel('Engine')
        label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        label.setMinimumWidth(minsize)
        label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        tabh.addWidget(label)
        self.engine = hou.qt.ComboBox()
        try:
            self.engine.addItem(QIcon(hou.qt.Icon("ROP_Redshift_ROP", BUTTON_ICON_SIZE, BUTTON_ICON_SIZE)),'Redshift')
        except: pass
        try:
            self.engine.addItem(QIcon(hou.qt.Icon("ROP_arnold", BUTTON_ICON_SIZE, BUTTON_ICON_SIZE)),'Arnold')
        except: pass
        try:
            self.engine.addItem(QIcon(hou.qt.Icon("ROP_ifd", BUTTON_ICON_SIZE, BUTTON_ICON_SIZE)),'Mantra')
        except: pass
        try:
            self.engine.addItem(QIcon(hou.qt.Icon("ROP_Octane_ROP", BUTTON_ICON_SIZE, BUTTON_ICON_SIZE)),'Octane')
        except: pass

        tabh.addWidget(self.engine)

        tabh = QHBoxLayout()
        tabh.setSpacing(5)
        tabh.setSizeConstraint(QLayout.SetMinimumSize)
        layout.addLayout(tabh)
        
        label = QLabel('Network')
        label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        label.setMinimumWidth(minsize)
        label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        tabh.addWidget(label)
        self.network = hou.qt.ComboBox()
        self.network.addItem(QIcon(hou.qt.Icon("NETWORKS_mat", BUTTON_ICON_SIZE, BUTTON_ICON_SIZE)),'/mat')
        self.network.addItem(QIcon(hou.qt.Icon("NETWORKS_shop", BUTTON_ICON_SIZE, BUTTON_ICON_SIZE)),'/shop')
        [self.network.addItem(QIcon(hou.qt.Icon("NETWORKS_mat", BUTTON_ICON_SIZE, BUTTON_ICON_SIZE)),n.path()) for n in hou.node('/').allSubChildren(True, False) if n.type().name() == 'matnet']
        [self.network.addItem(QIcon(hou.qt.Icon("NETWORKS_shop", BUTTON_ICON_SIZE, BUTTON_ICON_SIZE)),n.path()) for n in hou.node('/').allSubChildren(True, False) if n.type().name() == 'shopnet']

        tabh.addWidget(self.network)

        tabh = QHBoxLayout()
        tabh.setSpacing(5)
        tabh.setSizeConstraint(QLayout.SetMinimumSize)
        layout.addLayout(tabh)
        
        label = QLabel('Type')
        label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        label.setMinimumWidth(minsize)
        label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        tabh.addWidget(label)
        self.type = hou.qt.ComboBox()
        self.type.addItem(QIcon(hou.qt.Icon("SHOP_surface", BUTTON_ICON_SIZE, BUTTON_ICON_SIZE)),'Standard')
        self.type.addItem(QIcon(hou.qt.Icon("SOP_volume", BUTTON_ICON_SIZE, BUTTON_ICON_SIZE)),'Volume')
        tabh.addWidget(self.type)

        tabh = QHBoxLayout()
        tabh.setSpacing(5)
        tabh.setSizeConstraint(QLayout.SetMinimumSize)
        layout.addLayout(tabh)
        
        label = QLabel('Preset')
        label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        label.setMinimumWidth(minsize)
        label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        tabh.addWidget(label)
        self.presets = QComboBox()
        tabh.addWidget(self.presets)

        tabh = QHBoxLayout()
        tabh.setSpacing(5)
        tabh.setSizeConstraint(QLayout.SetMinimumSize)
        layout.addLayout(tabh)
        
        label = QLabel('Group')
        label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        label.setMinimumWidth(minsize)
        label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        tabh.addWidget(label)
        self.group = hou.qt.ComboBox()
        self.group.setCurrentIndex(-1)
        #self.group.insertSeparator(2)
        tabh.addWidget(self.group)

        layout.addItem(QSpacerItem(0, 10, QSizePolicy.Expanding, QSizePolicy.Fixed))

        tabh = QHBoxLayout()
        tabh.setSpacing(5)
        tabh.setSizeConstraint(QLayout.SetMinimumSize)
        layout.addLayout(tabh)

        self.info = QLabel()
        self.info.setEnabled(0)
        tabh.addWidget(self.info)

        tabh.addItem(QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Minimum))


        BUTTON_ICON_SIZENEW = hou.ui.scaledSize(45)
        self.confirm = QPushButton()
        self.confirm.setIcon(hou.qt.Icon("LOP_materiallibrary", BUTTON_ICON_SIZENEW, BUTTON_ICON_SIZENEW))

        self.confirm.setFlat(True)
        self.confirm.setFixedSize(BUTTON_ICON_SIZENEW, BUTTON_ICON_SIZENEW)
        self.confirm.setToolTip('Enter or Ctrl+Enter to assign material & close.')

        tabh.addWidget(self.confirm)

        self.setLayout(layout)

        ## User Shortcuts

        shortcut1 = QShortcut(QKeySequence("Return"), self)
        shortcut1.activated.connect(lambda:self.assignMaterial(0))
        shortcut1 = QShortcut(QKeySequence("Ctrl+Return"), self)
        shortcut1.activated.connect(lambda:self.assignMaterial(1))

        shortcut2 = QShortcut(QKeySequence("Ctrl+1"), self)
        shortcut2.activated.connect(lambda:self.switchType(self.engine))
        self.engine.setToolTip('Ctrl+1')

        shortcut2 = QShortcut(QKeySequence("Ctrl+2"), self)
        shortcut2.activated.connect(lambda:self.switchType(self.network))
        self.network.setToolTip('Ctrl+2')

        shortcut2 = QShortcut(QKeySequence("Ctrl+3"), self)
        shortcut2.activated.connect(lambda:self.switchType(self.type))
        self.type.setToolTip('Ctrl+3')

        shortcut2 = QShortcut(QKeySequence("Ctrl+4"), self)
        shortcut2.activated.connect(lambda:self.switchType(self.presets))
        self.presets.setToolTip('Ctrl+4')

        shortcut2 = QShortcut(QKeySequence("Ctrl+5"), self)
        shortcut2.activated.connect(lambda:self.switchType(self.group))
        self.group.setToolTip('Ctrl+5')

        shortcut2 = QShortcut(QKeySequence("Tab"), self)
        shortcut2.activated.connect(lambda:self.switchType(self.namemode))
        self.namemode.setToolTip('Get name from selected node. Tab')

        ## Callbacks

        self.engine.currentIndexChanged.connect(self.loadPresets)
        self.type.currentIndexChanged.connect(self.loadPresets)
        self.namemode.clicked.connect(self.nameconnect)
        self.network.currentIndexChanged.connect(self.updateNetboxes)
        self.presets.currentIndexChanged.connect(self.focus)
        self.group.currentIndexChanged.connect(self.groupChange)
        self.confirm.clicked.connect(lambda:self.assignMaterial(0))

        ## Load Previous Selections
        
        self.pref = hou.expandString('$HOUDINI_USER_PREF_DIR/mat.pref')

        self.loadPresets()

        self.loadDefaults()

        ## Decide the mode to run

        self.set_list = set_list
        self.geo_selection = geo_selection
        if self.set_list:

            if self.geo_selection:
                self.info.setText('Scene Viewer Mode')

            else:
                self.info.setText('Setting for %i node(s).'%len(self.set_list))

    ## CREATE & ASSIGN MATERIAL

    def assignMaterial(self,close):
        name = self.name.text()
        engine = self.engine.currentText()
        network = self.network.currentText()
        category = self.type.currentText()
        namemode = self.namemode.isChecked()
        nfo = self.info

        if self.set_list:

            ## Scene Viewer Mode

            if self.geo_selection:
                sceneViewer = self.set_list[0][0]

                self.geo = str(sceneViewer.
                    currentGeometrySelection())

                if not self.geo:
                    nfo.setText('Select geometry.')
                    return

                sel = [sceneViewer.currentNode()]

            ## Material Assign Mode
                
            else:
                sel = [n[0] for n in self.set_list]

        ## Regular Mode

        else:
            sel = hou.selectedNodes()

        self.count = 0

        with hou.undos.group("Assign Material"):
            if not namemode:

                ## USING USER DEFINED NAME

                if name:
                    if re.search(r"([^a-z0-9_ ])", name.lower()):
                        nfo.setText('Name contains illegal characters.')
                        return

                    name = name.replace(' ','_')
                else: name = None

                # Setting Paths

                if sel:
                    shape = sel[0].userData('nodeshape')
                    cd = sel[0].color()

                else: shape,cd = [None,None]

                path = self.buildMaterial(network,engine,name,category,shape,cd)

                if sel:
                    if self.set_list and not self.geo_selection:
                        [[p.set(path) for p in n[1]] for n in self.set_list]
                        self.count = len(self.set_list)
                    else:
                        [self.setPaths(n,path,name) for n in sel]

                nfo.setText('%s material(s) created | %i path(s) set.'%(engine,self.count))

            elif sel:

                ## USING NODE NAME

                for ind, n in enumerate(sel):
                    name = n.name()
                    shape = n.userData('nodeshape')
                    cd = n.color()
                    path = self.buildMaterial(network,engine,name,category,shape,cd)

                    if self.set_list and not self.geo_selection:
                        [p.set(path) for p in self.set_list[ind][1]]
                        self.count = len(self.set_list)
                    else:
                        self.setPaths(n,path,name)

                nfo.setText('%i %s %s created.'%(self.count,engine,category.lower()))

            else:

                nfo.setText('No nodes selected.')
                return

        if close or self.set_list and not self.geo_selection:
            self.hide()


    def buildMaterial(self,network,engine,name,category,shape,cd):

        mattype,subtype,inpt = self.getTypeName(engine,network,category)

        mat = hou.node(network).createNode(mattype,name)

        self.groupNode(mat,network)

        if cd:
            mat.setColor(cd)
        if shape:
            mat.setUserData('nodeshape', shape)

        submat = mat.createNode(subtype)

        preset = self.presets.currentText()
        if preset != 'None':
            hou.hscript('oppresetload %s "%s"'%(submat.path(), preset))

        if engine == 'Mantra':
            [c.destroy() for c in mat.children() if c != submat]
            surface = mat.createNode('output')
            surface = surface.createInputNode(0, 'surfaceexports')
        else:
            [c.destroy() for c in mat.children() if c.type().name().startswith(('redshift::')) and c != submat]
            surface = mat.children()[0]

        surface.insertInput(inpt, submat)

        mat.layoutChildren()

        return mat.path()

    def groupNode(self,mat,network):
        ind = self.group.currentIndex()
        network = hou.node(network)

        if ind == 0:
            network.layoutChildren()

        else:
            netbox = self.netboxes[ind-1]
            nodeshift = (0.7, -1.2)

            ## Create netbox
            
            if not netbox[0]:
                netbox = self.createNetbox(network,netbox[1])

            else:
                netbox = netbox[0]

            netbox.setAutoFit(False)

            ## Add mat to netbox

            mat.setPosition(( netbox.position()[0], netbox.position()[1] + netbox.size()[0] ))
            mat.move(nodeshift)
            netbox.addItem(mat)

            ## Get netboxes and bounding boxes

            netboxes = self.getBBX(network)

            cur = [b[1] for b in netboxes if b[0] == netbox][0]

            ## Remove and layout nodes

            nodes = netbox.nodes(True)

            netbox.removeAllNodes()

            network.layoutChildren(nodes)

            ## Move all nodes into the top right

            first = self.getFirstNode(nodes)

            shift = ( cur.min()[0] - first.position().x(), cur.max()[1] - first.position().y() )

            for n in nodes:
                n.move(shift)
                n.move(nodeshift)
                netbox.addNode(n)

        return

        ## TODO Resolve intersections

        netboxes = self.getBBX(network)## Returns netbox(0) boundingbox(1)
        ungrouped = self.ungrouped(network)## Returns none(0) boundingbox(1)
        if ungrouped[0][1]:
            netboxes += ungrouped
        
        if netboxes:
            overlaps = netboxes
            i=-1
            for n in overlaps:
                i+=1
                if n[0]:
                    for others in overlaps:
                        if n[0] != others[0]:
                            #if n[1].intersects(others[1]):
                            #    print 'b'
                            x = others[1].min()[0]
                            avoid = n[1].getOffsetToAvoid(others[1], hou.Vector2(0.0, -1.0))
                            if avoid[1] != 0:
                                print avoid
                                bounds = n[1]
                                bounds.translate((0.0, avoid[1] - 2))

                                n[0].setBounds(bounds)

                                overlaps[i] = (overlaps[i][0], bounds)

    def createNetbox(self,network,comment):
        netbox = network.createNetworkBox(None)
        netbox.setComment(comment)

        self.netboxes = self.netboxes[:-1]
        self.netboxes.append((netbox,comment))

        return netbox

    def getBBX(self,network):
        netboxes = network.networkBoxes()
        if not netboxes:
            return [(None,None)]

        netboxes = [(b,b.position(),b.size()) for b in netboxes]

        netboxes = [( b[0],
        hou.BoundingRect(b[1][0], b[1][1], b[1][0] + b[2][0], b[1][1] + b[2][0]) ) 
        for b in netboxes]

        return netboxes

    def getnodeBBX(self,node):
        b = ((node.position(),node.size()))

        bound = hou.BoundingRect(b[0][0], b[0][1], b[0][0] + b[1][0], b[0][1] + b[1][0])

        return bound

    def takeSecond(self,elem):
        return elem[1]

    def getFirstNode(self,nodes):
        nodes = [(n,n.position().x()) for n in nodes]
        nodes.sort(key=self.takeSecond)

        nodes = [(n[0],n[0].position().y()) for n in nodes if n[1] == nodes[0][1]]
        nodes.sort(key=self.takeSecond)

        return nodes[-1][0]

    def ungrouped(self,network):
        ungrouped = [(n,n.position().x()) for n in network.children() if not n.parentNetworkBox()]
        if not ungrouped:
            return [( None, None )]

        ungrouped.sort(key=self.takeSecond)
        x = ungrouped

        ungrouped = [(n[0],n[0].position().y()) for n in ungrouped]
        ungrouped.sort(key=self.takeSecond)
        ungrouped.reverse()
        y = ungrouped

        p1 = hou.Vector2((x[0][1], y[0][1]))
        p2 = hou.Vector2((x[-1][1], y[-1][1]))

        return [( None, hou.BoundingRect(p1,p2) )]

    def getTypeName(self,engine,network,category):
        e = engine.lower()
        enginedict = {
          "redshift_/mat": (('redshift_vopnet','redshift::Volume','redshift::Material',4,0)),
          "redshift_/shop": (('redshift_vopnet','redshift::Volume','redshift::Material',4,0)),
          "arnold_/mat": (('arnold_materialbuilder','arnold::standard_volume','arnold::standard_surface',2,0)),
          "arnold_/shop": (('arnold_vopnet','arnold::standard_volume','arnold::standard_surface',2,0)),
          "mantra_/mat": (('materialbuilder','volumeshadercore','principledshadercore',0,0)),
          "mantra_/shop": (('vopmaterial','volumeshadercore','principledshadercore',0,0)),
          "octane_/mat": (('octane_vopnet','octane::NT_MED_VOLUME','octane::NT_MAT_UNIVERSAL',1,0)),
          "octane_/shop": (('octane_vopnet','octane::NT_MED_VOLUME','octane::NT_MAT_UNIVERSAL',1,0))
        }

        network = hou.node(network).type().name()
        if network in ['matnet','mat']:
            network = '/mat'
        else:
            network = '/shop'

        for e,t in enginedict.items():
            if '%s_%s'%(engine.lower(),network.lower()) == e:

                mattype = t[0]
                if category.lower() == 'volume':
                    subtype = t[1]
                    inpt = t[3]
                else:
                    subtype = t[2]
                    inpt = t[4]

                break

        return [mattype,subtype,inpt]


    def setPaths(self,s,path,name):
        geodict = {
          "geo": "shop_materialpath",
          "instance": "shop_materialpath",
          "arnold_procedural": "shop_materialpath",
          "arnold_volume": "shop_materialpath",
        }

        ntype = s.type().category().name().lower()

        if ntype == 'object':
            setlist = [s.parm(p).set(path) 
                for t,p in geodict.items() 
                    if s.type().name() == t]
            if setlist:
                self.count+=1

        elif ntype == 'sop':
            chl = s.outputs()

            if s.type().name() == 'material' and not hou.node(s.evalParm('shop_materialpath1')):
                mat = s
                move = False
            else:
                mat = s.parent().createNode('material', '%s_mat1'%name)
                mat.setNextInput(s)
                move = True

                for flag in [hou.nodeFlag.Render, hou.nodeFlag.Display]:
                    if s.isGenericFlagSet(flag):
                        mat.setGenericFlag(flag,1)

            mat.parm('shop_materialpath1').set(path)

            ## Set scene view selection if applicable

            if self.geo_selection:
                mat.parm('group1').set(self.geo)

            ## Add to wire

            if move:
                if chl:
                    for child in chl:
                        for ind, input in enumerate(child.inputs()):
                            if s.name() in input.name():
                                index = ind
                        child.setInput(index, None)
                        child.insertInput(index, mat)
            
                mat.moveToGoodPosition(True, False, True, True)

            self.count+=1

    ## Load Presets on Node change

    def loadPresets(self):
        self.presets.clear()
        self.presets.addItem('None')

        type = self.getTypeName(self.engine.currentText(),'/mat',self.type.currentText())[1]
        presets = self.getPresets(type)
        if presets[0]:
            self.presets.addItems(presets[1])

        ## Update group combo box

        self.updateNetboxes()

    def getPresets(self,type):
        if '::' in type:
            splt = type.split("::")
            string = '%s::Vop/%s'%(splt[0],splt[1])
        else:
            string = 'Vop/%s'%type

        presets = hou.hscript('oppresetls -t %s'%string)
        if presets[1].startswith('Invalid operator type:'):
            return [0,None]
        else:
            presets = presets[0].split('\n')[:-1]
            try: presets.remove('Permanent Defaults')
            except: pass
            return [1,presets]

    ## Switch Name Mode

    def nameconnect(self):
        self.name.setEnabled(
            self.namemode.isChecked() == 0)

        self.focus()

    ## Focus on Line Edit

    def mousePressEvent(self, event):
        self.focus()
        self.name.selectAll()

    def focus(self):
        self.name.setFocus()

    ## Switch Type

    def switchType(self,widget):
        if widget == self.namemode:
            self.namemode.toggle()
            self.nameconnect()
        else:
            newInd = (widget.currentIndex() + 1) % widget.count()
            widget.setCurrentIndex(newInd)

    ## Group functions

    def updateNetboxes(self):
        self.group.clear()
        self.group.addItem('None')

        ## Get network boxes

        network = hou.node(self.network.currentText())

        self.netboxes = []

        if network:
            self.netboxes = [(n,n.comment()) for n in network.networkBoxes()]
            if self.netboxes:
                self.group.addItems([n[1] for n in self.netboxes])

        ## Add new group option

        self.group.addItem('-- New Group --')

        ## Focus on name

        self.focus()

    def groupChange(self):
        index = self.group.currentIndex()
        text = self.group.currentText()

        if index == self.group.count()-1 and text == '-- New Group --':
            grp = QInputDialog.getText(self, 'Input', 'New Group:', 
                flags = QtCore.Qt.WindowSystemMenuHint | QtCore.Qt.WindowTitleHint)

            self.group.setCurrentIndex(0)

            if grp[0] and grp[1]:
                name = grp[0]
                ind = self.group.count()-1

                ## Add new item and adjust group

                if self.netboxes:
                    if not self.netboxes[-1][0]:
                        self.netboxes = self.netboxes[:-1]
                        self.group.removeItem(ind-1)

                        ind = self.group.count()-1

                self.group.insertItem(ind, name)

                self.netboxes.append((None,name))

                self.group.setCurrentIndex(ind)

        self.focus()

    ## Load & Save Selection

    def hideEvent(self, event):
        self.saveChanges()
        self.setParent(None)

    def saveChanges(self):
        item = [self.engine.currentText(),self.network.currentText(),self.type.currentText(),self.presets.currentText(),self.namemode.isChecked(),self.group.currentText()]
        a = file(self.pref, 'w')
        a.write(repr(item))

    def loadDefaults(self):
        if os.path.isfile(self.pref):
            f = file(self.pref,'r')
            with open(self.pref, 'r') as f:
                recent = ast.literal_eval(f.read())

            try:
                self.engine.setCurrentText(recent[0])
            except: pass
            try:
                self.network.setCurrentText(recent[1])
            except: pass
            try:
                self.type.setCurrentText(recent[2])
            except: pass
            try:
                self.presets.setCurrentText(recent[3])
            except: pass
            try:
                self.namemode.setChecked(recent[4])
                self.nameconnect()
            except: pass
            try:
                self.group.setCurrentText(recent[5])
            except: pass

def sceneViewCheck():
    sceneViewer = hou.ui.curDesktop().paneTabUnderCursor()

    if sceneViewer:
        if sceneViewer.type().name() == 'SceneViewer':
            return [(sceneViewer, True)]
                
    return None

def run(set_list):
    [entry.close() for entry in __import__('PySide2').QtWidgets.QApplication.allWidgets() if type(entry).__name__ == 'matAssign']

    if set_list:
        matAssign(set_list,False).exec_()

    else:
        set_list = sceneViewCheck()

        if set_list:
            matAssign(set_list,True).show()

        else:
            matAssign(None,False).show()