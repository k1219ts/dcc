import os

def createFolder(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def execMaya():
    import maya.cmds as cmds
    import maya.mel as mel
    import sys

    cmds.loadPlugin('DXUSD_Maya')
    try:
        if sys.argv[-1].endswith('.mb'):
            workspace = os.path.dirname(os.path.dirname(sys.argv[-1]))
            mel.eval('setProject \"' + workspace + '\"')
            #for fileRule in cmds.workspace(query=True, fileRuleList=True):
#                fileRuleDir = cmds.workspace(fileRuleEntry=fileRule)
#                mayaFileRuleDir = os.path.join(workspace, fileRuleDir)
#                createFolder(mayaFileRuleDir)
            cmds.file(sys.argv[-1], o=True, f=True)
    except:
        pass
        #cmds.file(rename=sys.argv[-1])
        #cmds.file(save=True)

def execFull():
    import maya.cmds as cmds
    import maya.mel as mel
    import sys

    cmds.loadPlugin('DXUSD_Maya')
    # cmds.loadPlugin('TaneForMaya')
    cmds.loadPlugin('xgenToolkit')
    cmds.loadPlugin('ZENNForMaya')
    try:
        if sys.argv[-1].endswith('.mb'):
            workspace = os.path.dirname(os.path.dirname(sys.argv[-1]))
            mel.eval('setProject \"' + workspace + '\"')
            # for fileRule in cmds.workspace(query=True, fileRuleList=True):
            #                fileRuleDir = cmds.workspace(fileRuleEntry=fileRule)
            #                mayaFileRuleDir = os.path.join(workspace, fileRuleDir)
            #                createFolder(mayaFileRuleDir)
            cmds.file(sys.argv[-1], o=True, f=True)
    except:
        pass
        #cmds.file(rename=sys.argv[-1])
        #cmds.file(save=True)


def execGolaem():
    import maya.cmds as cmds
    import maya.mel as mel
    import sys

    cmds.loadPlugin('glmCrowd')
    cmds.loadPlugin('DXUSD_Maya')
    try:
        if sys.argv[-1].endswith('.mb'):
            workspace = os.path.dirname(os.path.dirname(sys.argv[-1]))
            mel.eval('setProject \"' + workspace + '\"')
            #for fileRule in cmds.workspace(query=True, fileRuleList=True):
#                fileRuleDir = cmds.workspace(fileRuleEntry=fileRule)
#                mayaFileRuleDir = os.path.join(workspace, fileRuleDir)
#                createFolder(mayaFileRuleDir)
            cmds.file(sys.argv[-1], o=True, f=True)
    except:
        pass
        #cmds.file(rename=sys.argv[-1])
        #cmds.file(save=True)


def execMiarmy():
    import maya.cmds as cmds
    import maya.mel as mel
    import sys

    cmds.loadPlugin('DXUSD_Maya')
    cmds.loadPlugin('MiarmyProForMaya2018')
    cmds.loadPlugin('MiarmyForDexter')
    try:
        if sys.argv[-1].endswith('.mb'):
            workspace = os.path.dirname(os.path.dirname(sys.argv[-1]))
            mel.eval('setProject \"' + workspace + '\"')
            #for fileRule in cmds.workspace(query=True, fileRuleList=True):
#                fileRuleDir = cmds.workspace(fileRuleEntry=fileRule)
#                mayaFileRuleDir = os.path.join(workspace, fileRuleDir)
#                createFolder(mayaFileRuleDir)
            cmds.file(sys.argv[-1], o=True, f=True)
    except:
        pass
        #cmds.file(rename=sys.argv[-1])
        #cmds.file(save=True)


def execTane():
    import maya.cmds as cmds
    import maya.mel as mel
    import sys

    cmds.loadPlugin('DXUSD_Maya')
    cmds.loadPlugin('TaneForMaya')
    try:
        if sys.argv[-1].endswith('.mb'):
            workspace = os.path.dirname(os.path.dirname(sys.argv[-1]))
            mel.eval('setProject \"' + workspace + '\"')
            #for fileRule in cmds.workspace(query=True, fileRuleList=True):
#                fileRuleDir = cmds.workspace(fileRuleEntry=fileRule)
#                mayaFileRuleDir = os.path.join(workspace, fileRuleDir)
#                createFolder(mayaFileRuleDir)
            cmds.file(sys.argv[-1], o=True, f=True)
    except:
        pass
        #cmds.file(rename=sys.argv[-1])
        #cmds.file(save=True)


def execZENN():
    import maya.cmds as cmds
    import maya.mel as mel
    import sys

    cmds.loadPlugin('DXUSD_Maya')
    cmds.loadPlugin('xgenToolkit')
    cmds.loadPlugin('ZENNForMaya')
    try:
        if sys.argv[-1].endswith('.mb'):
            workspace = os.path.dirname(os.path.dirname(sys.argv[-1]))
            mel.eval('setProject \"' + workspace + '\"')
            #for fileRule in cmds.workspace(query=True, fileRuleList=True):
#                fileRuleDir = cmds.workspace(fileRuleEntry=fileRule)
#                mayaFileRuleDir = os.path.join(workspace, fileRuleDir)
#                createFolder(mayaFileRuleDir)
            cmds.file(sys.argv[-1], o=True, f=True)
    except:
        pass
        #cmds.file(rename=sys.argv[-1])
        #cmds.file(save=True)


def execZiva():
    import maya.cmds as cmds
    import maya.mel as mel
    import sys

    cmds.loadPlugin('DXUSD_Maya')
    cmds.loadPlugin('ziva')
    try:
        if sys.argv[-1].endswith('.mb'):
            workspace = os.path.dirname(os.path.dirname(sys.argv[-1]))
            mel.eval('setProject \"' + workspace + '\"')
            #for fileRule in cmds.workspace(query=True, fileRuleList=True):
#                fileRuleDir = cmds.workspace(fileRuleEntry=fileRule)
#                mayaFileRuleDir = os.path.join(workspace, fileRuleDir)
#                createFolder(mayaFileRuleDir)
            cmds.file(sys.argv[-1], o=True, f=True)
    except:
        pass
        #cmds.file(rename=sys.argv[-1])
        #cmds.file(save=True)
