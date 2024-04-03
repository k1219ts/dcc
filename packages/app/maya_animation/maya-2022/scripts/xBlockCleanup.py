import xBlockUtils
import maya.cmds as cmds

def cleanupXBlock():
    for node in cmds.ls(sl = True, type = 'dxRig'):
        renderMeshses = cmds.getAttr('%s.renderMeshes' % node)

        expandedNodes = cmds.ls(node, dag=True, type='pxrUsdReferenceAssembly')

        for expNode in expandedNodes:
            if expNode in renderMeshses:
                xBlockUtils.Expanded(expNode).doIt()

        expandedNodes = cmds.ls(node, dag=True, type='pxrUsdReferenceAssembly')
        for expNode in expandedNodes:
            if cmds.objExists(expNode):
                try:
                    cmds.delete(expNode)
                except:
                    print "WARNING", "don't delete node : %s" % expNode
            else:
                print "WARNING", "not found node : %s" % expNode