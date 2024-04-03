# if will you use mayapy? from pymel.all import *
# because load environment setting
from pymel.all import *

import sys
import os

for p in os.getenv('PYTHONPATH').split(':'):
    if p and not p in sys.path:
        sys.path.append( p )

import rfmShading
import rfm.rmanAssetsMaya as ram

import maya.cmds as cmds
import maya.mel as mel

# if __name__ == '__main__':
for argv in sys.argv:
    print argv

print "import json file path :", sys.argv[1]
print "export ma path :", sys.argv[2]

# import template json shader
ram.importAsset( sys.argv[1] )

# rfm shader and binding
rfmClass = rfmShading.RfMShaders()
rfmClass.exportProcess( File = sys.argv[2], Mode='Entire', Binding=False )

os._exit(0)