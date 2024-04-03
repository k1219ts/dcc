import os
import math
from maya import cmds, mel, OpenMayaUI
from maya.api import OpenMaya

# import pyside, do qt version check for maya 2017 >
qtVersion = cmds.about(qtVersion=True)
if qtVersion.startswith("4") or type(qtVersion) not in [str, unicode]:
    from PySide.QtGui import *
    from PySide.QtCore import *
    import shiboken
else:
    from PySide2.QtGui import *
    from PySide2.QtCore import *
    from PySide2.QtWidgets import *
    import shiboken2 as shiboken
    
# ----------------------------------------------------------------------------

class UndoChunkContext(object):

    def __enter__(self):
        cmds.undoInfo(openChunk=True)
        
    def __exit__(self, *exc_info):
        cmds.undoInfo(closeChunk=True)
    
# ----------------------------------------------------------------------------

FONT = QFont()
FONT.setFamily("Consolas")

BOLT_FONT = QFont()
BOLT_FONT.setFamily("Consolas")
BOLT_FONT.setWeight(100)  

# ----------------------------------------------------------------------------

def mayaWindow():

    window = OpenMayaUI.MQtUtil.mainWindow()
    window = shiboken.wrapInstance(long(window), QMainWindow)
    
    return window  
    
def getMayaTimeline():

    return mel.eval("$tmpVar=$gPlayBackSlider")
    
# ----------------------------------------------------------------------------
    
def findIcon(icon):

    paths = []

    # get maya icon paths
    if os.environ.get("XBMLANGPATH"):     
        paths = os.environ.get("XBMLANGPATH").split(os.pathsep)                                 

    # append tool icon path
    paths.insert(
        0,
        os.path.join(
            os.path.split(__file__)[0], 
            "icons" 
        ) 
    )

    # loop all potential paths
    for path in paths:
        filepath = os.path.join(path, icon)
        if os.path.exists(filepath):
            return filepath
            
# ----------------------------------------------------------------------------

ATTRIBUTES = ["translate", "rotate"]
CHANNELS = ["X", "Y", "Z"]

# ----------------------------------------------------------------------------

def getMatrix(transform, time=None, matrixType="worldMatrix"):

    if not transform:
        return OpenMaya.MMatrix()
        
    if not time:
        time = cmds.currentTime(query=True)
    
    rotatePivot = cmds.getAttr("{0}.rotatePivot".format(transform))[0]
    
    matrix = cmds.getAttr("{0}.{1}".format(transform, matrixType), time=time)
    return OpenMaya.MMatrix(matrix)
    
def decomposeMatrix(matrix, rotOrder, rotPivot):

    matrixTransform = OpenMaya.MTransformationMatrix(matrix)
    
    # set pivots
    matrixTransform.setRotatePivot(
        OpenMaya.MPoint(rotPivot), 
        OpenMaya.MSpace.kTransform, 
        True
    )
    
    # get rotation pivot translation
    posOffset =  matrixTransform.rotatePivotTranslation(
        OpenMaya.MSpace.kTransform
    )
    
    # get pos values
    pos = matrixTransform.translation(OpenMaya.MSpace.kTransform)
    pos += posOffset
    pos = [pos.x, pos.y, pos.z]
    
    # get rot values
    euler = matrixTransform.rotation()
    euler.reorderIt(rotOrder)
    rot = [math.degrees(angle) for angle in [euler.x, euler.y, euler.z]]
    
    # get scale values
    scale = matrixTransform.scale(OpenMaya.MSpace.kTransform)
    
    return [pos, rot, scale]
    
# ----------------------------------------------------------------------------

def getInTangent(animCurve, time):
    times = cmds.keyframe(animCurve, query=True, timeChange=True) or []
    for t in times:
        if t <= time:
            continue
        
        tangent = cmds.keyTangent(
            animCurve, 
            time=(t,t), 
            query=True, 
            inTangentType=True
        )
        
        return tangent[0]

    return "auto"
    
def getOutTangent(animCurve, time):
    times = cmds.keyframe(animCurve, query=True, timeChange=True) or []
    for t in times:
        if t >= time:
            continue
        
        tangent = cmds.keyTangent(
            animCurve, 
            time=(t,t), 
            query=True, 
            outTangentType=True
        )
        
        return tangent[0]

    return "auto"
