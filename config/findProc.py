#!/bin/python

import os, sys

if __name__ == '__main__':
    opts = sys.argv
    if 'maya' in opts or 'mayapy' in opts:
        sys.exit('maya')
    if 'xgenMaya' in opts:
        sys.exit('maya')
    if 'houdini' in opts or 'hython' in opts:
        sys.exit('houdini')
    if 'katana' in opts:
        sys.exit('katana')
    if 'rfk' in opts:
        sys.exit('rfk')
    if 'nuke' in opts or 'nukeX' in opts or 'nukeS' in opts or 'nukeP' in opts:
        sys.exit('nuke')
    if 'motionbuilder' in opts:
        sys.exit('motionbuilder')
    if 'mari' in opts:
        sys.exit('mari')
    if '3de' in opts:
        sys.exit('3dequalizer')
    if 'clarisse' in opts:
        sys.exit('clarisse')
    if 'rv' in opts:
        sys.exit('rv')
    if 'golaem' in opts:
        sys.exit('golaem')
