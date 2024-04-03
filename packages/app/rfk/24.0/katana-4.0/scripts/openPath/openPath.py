from Katana import NodegraphAPI, Nodes3DAPI
from Katana import UI4
import os


#Open Texture Path
class OpenTexturePath:
    def __init__(self):
        getSceneGraph = Nodes3DAPI.ScenegraphManager.getActiveScenegraph()
        self.getselectedLocation = getSceneGraph.getSelectedLocations()

        # ERROR Message
        if not self.getselectedLocation:
            UI4.Widgets.MessageBox.Critical('Selection Error', 'Please Select Nodes!')

    def openDirectory(self, node):
        # Search Project
        filepath = node.getGlobalAttribute('info.usdOpArgs.fileName').getValue()
        show = filepath.split('/')[2]
        if filepath.startswith('/mach'):
            show = filepath.split('/')[3]

        # Find Attribute
        baseName = node.getAttribute('geometry.arbitrary.txBasePath.value').getValue()
        txVersion = node.getAttribute('prmanStatements.attributes.user.txVersion').getValue()
        path = '/show/' + show + '/_3d/' + baseName + '/tex/' + txVersion + '/'

        # Open Directory
        if path:
            os.system('nautilus %s &' % os.path.dirname(path))

    def textureDoIt(self):
        for location in self.getselectedLocation:
            rootLocation = Nodes3DAPI.GetGeometryProducer()
            node = rootLocation.getProducerByPath(location)
            # ERROR Message
            if node.getType() != 'subdmesh' and node.getType() != 'polymesh' and \
                    node.getType() != 'curves':
                UI4.Widgets.MessageBox.Critical('Selection Error', 'Please Select Geomoetry Nodes!')
            elif node.getAttribute('geometry.arbitrary.txBasePath.value') == None or \
                node.getAttribute('prmanStatements.attributes.user.txVersion') == None:
                UI4.Widgets.MessageBox.Critical('Attribute Error', 'Please Check Attributes!')
            else:
                #Run
                self.openDirectory(node)


#Open Material Path
class OpenMaterialPath(OpenTexturePath):
    def __init__(self):
        OpenTexturePath.__init__(self)

    def materialDoIt(self):
        for location in self.getselectedLocation:
            rootLocation = Nodes3DAPI.GetGeometryProducer()
            node = rootLocation.getProducerByPath(location)
            # ERROR Message
            if node.getType() != 'material':
                UI4.Widgets.MessageBox.Critical('Selection Error', 'Please Select Material Nodes!')
            elif node.getAttribute('usd.layerPath') == None:
                UI4.Widgets.MessageBox.Critical('Attribute Error', 'Please Check LayerPath!')
            else:
                # Run
                path = node.getAttribute('usd.layerPath').getValue()
                if path:
                    os.system('nautilus %s &' % os.path.dirname(path))


#Open Shot Path
class OpenShotPath:
    def __init__(self):
        # Find '/root/world Find'
        worlds = ['/root/world', '/root/world/geo']
        self.world = list()

        for fileName in worlds:
            rootLocation = Nodes3DAPI.GetGeometryProducer()
            node = rootLocation.getProducerByPath(fileName)
            if node:
                usdOpArgs = node.getAttribute('info.usdOpArgs.location')
                if usdOpArgs:
                    usdOpArgsValue = usdOpArgs.getValue()
                    self.world.append(usdOpArgsValue)

    def shotDoIt(self):
        # Open Directory
        rootAttribute = Nodes3DAPI.GetGeometryProducer().getProducerByPath(self.world[0])
        fileName = rootAttribute.getAttribute('info.usdOpArgs.fileName').getValue()

        if fileName.find('asset') != -1:
            # Asset
            if fileName:
                os.system('nautilus %s &' % os.path.dirname(fileName))
        else:
            # Shot
            fileNameSplit = fileName.split('shot.usd')
            shot = rootAttribute.getAttribute('info.usdOpArgs.system.variables.shot').getValue()
            seq = shot.split('_')[0]
            shotPath = fileNameSplit[0] + seq + '/' + shot + '/'
            if shotPath:
                os.system('nautilus %s &' % os.path.dirname(shotPath))
