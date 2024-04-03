import maya.cmds as cmds


#-------------------------------------------------------------------------------
#
#   dAgentGroup
#
#-------------------------------------------------------------------------------
def addSelectedAgentsForAgentGroup(ln):
    result = list()
    for i in cmds.ls(sl=True, dag=True, type='McdAgent'):
        result.append(cmds.getAttr('%s.agentId' % i))
    ids = cmds.getAttr(ln)
    if ids:
        result += ids
    result = list(set(result))
    result.sort()
    cmds.setAttr(ln, result, type='Int32Array')

def selectByAgentGroup(ln):
    ids = cmds.getAttr(ln)
    result = list()
    for i in ids:
        result.append('McdAgent%d' % i)
    cmds.select(result)

def selectItemsByAgentGroup(ln, strIds):
    result = list()
    for i in strIds.split(','):
        result.append('McdAgent%s' % i)
    cmds.select(result)

def removeItemsByAgentGroup(ln, strIds):
    ids = cmds.getAttr(ln)
    ui_ids = list()
    for i in strIds.split(','):
        ui_ids.append(int(i))
    result = list(set(ids) - set(ui_ids))
    result.sort()
    cmds.setAttr(ln, result, type='Int32Array')
