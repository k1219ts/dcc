"""
NAME: Reload All LiveGroup
ICON: /backstage/share/icons/katana/LiveGroupEditable16.png
DROP_TYPES:
SCOPE:

Reload All LiveGroups in the current scene
"""

from Katana import NodegraphAPI

nodes = NodegraphAPI.GetAllNodesByType('LiveGroup')
for node in nodes:
    print('# reload livegroup : %s' % node.getName())
    node.reloadFromSource()
