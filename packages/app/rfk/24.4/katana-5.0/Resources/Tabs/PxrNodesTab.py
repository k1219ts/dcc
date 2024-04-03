from Katana import UI4, NodegraphAPI
from PyQt5 import QtGui, QtCore, QtWidgets
import os, sys
import xml.etree.ElementTree as ET

scriptroot = os.path.dirname(os.path.abspath(__file__))


class CreatePxrNodeTab(UI4.Tabs.BaseTab):
    def __init__(self, parent):
        UI4.Tabs.BaseTab.__init__(self, parent)

        areawidget = QtWidgets.QWidget()
        self.layout = QtWidgets.QVBoxLayout(areawidget)
        scrollarea = QtWidgets.QScrollArea()
        scrollarea.setWidget(areawidget)
        scrollarea.setWidgetResizable(True)

        mainlayout = QtWidgets.QVBoxLayout()
        mainlayout.addWidget(scrollarea)
        self.setLayout(mainlayout)

        self.pxrdata = dict()
        self.getPxrNodes()
        self.buildUI()


    def getShaderType(self, argsfile):
        tree = ET.parse(argsfile)
        root = tree.getroot()
        shadertypes = root.findall('shaderType')
        if shadertypes:
            return shadertypes[0].getchildren()[0].attrib['value']

    def getPxrNodes(self):
        for dirpath in os.getenv('RMAN_RIXPLUGINPATH').split(':'):
            if dirpath:
                if os.path.exists(dirpath):
                    for f in os.listdir(dirpath):
                        fn = os.path.join(dirpath, f)
                        if os.path.isfile(fn) and os.path.splitext(fn)[-1] == '.so':
                            argsfile = os.path.join(dirpath, 'Args', f.replace('.so', '.args'))
                            if os.path.exists(argsfile):
                                shadertype = self.getShaderType(argsfile)
                                if shadertype:
                                    if not self.pxrdata.get(shadertype):
                                        self.pxrdata[shadertype] = list()
                                    self.pxrdata[shadertype].append(f.replace('.so', ''))


    def buildUI(self):
        labelMap = {
            'bxdf': 'Materials', 'displacement': 'Displacements',
            'pattern': 'PxrPatterns'
        }

        if self.pxrdata.get('bxdf'):
            except_shaders = [
                'PxrDiffuse', 'PxrDisney', 'PxrHair', 'PxrLMDiffuse', 'PxrLMGlass',
                'PxrLMMetal', 'PxrLMPlastic', 'PxrLMSubsurface', 'PxrSkin',
                'PxrGlass', 'PxrLightEmission'
            ]
            shaders = list(set(self.pxrdata['bxdf']) - set(except_shaders))
            shaders.sort()
            self.GroupBuild(labelMap['bxdf'], shaders)

        if self.pxrdata.get('displacement'):
            shaders = self.pxrdata['displacement']
            self.GroupBuild(labelMap['displacement'], shaders)

        if self.pxrdata.get('pattern'):
            except_shaders = [
                'DxTexture', 'PxrLMLayer', 'PxrLMMixer'
            ]
            append_shaders = [
                'PxrLayerMixer', 'PxrLayer', 'PxrShadedSide'
            ]
            shaders = list(set(self.pxrdata['pattern']) - set(except_shaders))
            shaders+= append_shaders
            shaders.sort()
            self.GroupBuild(labelMap['pattern'], shaders)


        spacer = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.layout.addItem(spacer)

    def GroupBuild(self, title, shaders):
        titleStyle = 'font-weight: bold; font-size: 18pt;'
        label = QtWidgets.QLabel(title)
        label.setStyleSheet(titleStyle)
        self.layout.addWidget(label)

        grid = QtWidgets.QGridLayout()
        grid.setSpacing(0)
        column_size = 3
        for i in range(len(shaders)):
            row = int(i / column_size)
            col = i % column_size
            grid.addWidget(self.buttonWidget(shaders[i]), row, col, 1, 1)
        self.layout.addLayout(grid)

        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.layout.addWidget(line)

    def buttonWidget(self, shader):
        button = QtWidgets.QPushButton()
        button.setText(shader)
        button.setFixedHeight(34)
        button.setMinimumWidth(100)
        button.setStyleSheet('text-align: left;')
        if shader:
            button.setObjectName(shader)
            icfile = self.getIconPath(shader)
            if icfile:
                button.setIcon(QtGui.QIcon(icfile))
                button.setIconSize(QtCore.QSize(30, 30))
            button.clicked.connect(self.createPrmanShadingNode)
        return button

    def getIconPath(self, shader):
        iconfile = None
        rfmtree  = '/opt/pixar/RenderManForMaya-{VER}/icons'.format(VER=os.getenv('RMAN_VER'))
        iconfile = os.path.join(rfmtree, 'render_%s.png' % shader)
        if os.path.exists(iconfile):
            return iconfile
        extendpath = '{DIR}/resources/icons'.format(DIR=os.getenv('EXTEND_RMAN_PATH'))
        iconfile = os.path.join(extendpath, 'render_%s.png' % shader)
        if os.path.exists(iconfile):
            return iconfile


    def createPrmanShadingNode(self):
        sender  = self.sender()
        objname = sender.objectName()

        x_offset = 0
        y_offset = 60

        NodegraphTab = UI4.App.Tabs.FindTopTab('Node Graph')
        if not NodegraphAPI:
            return
        parentNode = NodegraphTab.getEnteredGroupNode()
        if parentNode.getType() == 'NetworkMaterialCreate':
            x_offset = -300
            y_offset = 0

        selectedNodes = NodegraphAPI.GetAllSelectedNodes()
        if selectedNodes:
            if selectedNodes[-1] == parentNode:
                pos = NodegraphAPI.GetViewPortPosition(parentNode)[0]
            else:
                pos = NodegraphAPI.GetNodePosition(selectedNodes[-1])
        else:
            pos = NodegraphAPI.GetViewPortPosition(parentNode)[0]

        node = NodegraphAPI.CreateNode('PrmanShadingNode', parentNode)
        name = node.setName(str(objname))
        node.getParameter('name').setValue(name, 0)
        node.getParameter('nodeType').setValue(str(objname), 0)
        node.checkDynamicParameters()
        NodegraphAPI.SetNodePosition(node, (pos[0]+x_offset, pos[1]+y_offset))


PluginRegistry = [
    ('KatanaPanel', 2.0, 'CreatePrmanShadingNode', CreatePxrNodeTab),
]
