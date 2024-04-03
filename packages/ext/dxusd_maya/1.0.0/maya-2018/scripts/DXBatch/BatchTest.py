import os, sys


def doIt():
    import maya.api.OpenMaya as OpenMaya
    OpenMaya.MGlobal.displayError('batch script test.')
    os._exit(1)

if __name__ == '__main__':
    from pymel.all import *

    doIt()
