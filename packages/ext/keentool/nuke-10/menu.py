import nuke

nodesMenu = nuke.menu('Nodes')

n = nodesMenu.addMenu('KeenTools', icon='KeenTools.png')
n.addCommand('GeoTracker', lambda: nuke.createNode('GeoTracker'), 'GeoTracker.png')
n.addCommand('PinTool', lambda: nuke.createNode('PinTool'), 'PinTool.png')
n.addCommand('ReadRiggedGeo', lambda: nuke.createNode('ReadRiggedGeo'), 'ReadRiggedGeo.png')