
import maya.cmds as cmds

class XformBlock:
    def __init__(self, node):
        self.m_translate = cmds.getAttr('%s.translate' % node)[0]
        self.m_rotate = cmds.getAttr('%s.rotate' % node)[0]
        self.m_scale = cmds.getAttr('%s.scale' % node)[0]
        self.m_rotatePivot = cmds.getAttr('%s.rotatePivot' % node)[0]
        self.m_scalePivot = cmds.getAttr('%s.scalePivot' % node)[0]
        self.m_rotatePivotTranslate = cmds.getAttr('%s.rotatePivotTranslate' % node)[0]
        self.m_scalePivotTranslate = cmds.getAttr('%s.scalePivotTranslate' % node)[0]
    def Set(self, node):
        cmds.setAttr('%s.rotatePivot' % node, *self.m_rotatePivot)
        cmds.setAttr('%s.rotatePivotTranslate' % node, *self.m_rotatePivotTranslate)
        cmds.setAttr('%s.scalePivot' % node, *self.m_scalePivot)
        cmds.setAttr('%s.scalePivotTranslate' % node, *self.m_scalePivotTranslate)
        cmds.setAttr('%s.scale' % node, *self.m_scale)
        cmds.setAttr('%s.rotate' % node, *self.m_rotate)
        cmds.setAttr('%s.translate' % node, *self.m_translate)


def CopyNode(target, source):
    parentNode = cmds.listRelatives(target,p=True,f=True)[0]
    xb = XformBlock(target)
    copynode = cmds.instance(source)[0]
    copynode = cmds.ls(copynode, l=1)[0]
    copynode = cmds.parent(copynode, parentNode)[0]
    xb.Set(copynode)
    cmds.delete(target)

def SourceInit():
    # Get raw source
    sclist = []
    for sc in cmds.ls('|reference_sc_tmp', dag=1,type = 'transform'):
        if '_loc' in sc:
            sc = sc.split('|')[-1].split('_loc')[0]
            sc = '|reference_sc_tmp'+'|'+sc
            if cmds.objExists(sc):
                if not sc in sclist:
                    sclist.append(sc)
    if not sclist:
        return

    # Source
    for sc in sclist:
        scname = sc.split('|')[-1]
        scname += '_loc'
        for target in cmds.ls('|reference_sc_tmp', dag=1, type = 'transform', l=1):
            if scname in target:
                CopyNode(target, sc)

def ReplaceAll():
    for sc in cmds.listRelatives('|reference_sc_tmp', c=True, f=True):
        scname = sc.split('|')[-1]
        scname += '_loc'
        for target in cmds.ls('*|%s*' %scname, type='transform'):
            parentNode = cmds.listRelatives(target,p=True,f=True)[0]
            xb = XformBlock(target)
            copynode = cmds.instance(sc)[0]
            copynode = cmds.ls(copynode, l=1)[0]
            copynode = cmds.parent(copynode, parentNode)[0]
            xb.Set(copynode)
            cmds.delete(target)

def ReplaceSelected(nodes=''):
    if not nodes:
        nodes = cmds.ls(sl=1,type= 'transform', l=True)
    else:
        nodes = cmds.ls(nodes, l =True)

    for locator in nodes:
        if '_loc' in locator:
            sc = locator.split('_loc')[0].split('|')[-1]
            scpath = '|reference_sc_tmp' + '|' + sc
            if cmds.objExists(scpath):
                CopyNode(locator, sc)


SourceInit()
ReplaceAll()
#ReplaceSelected()

