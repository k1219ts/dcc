#encoding=utf-8
#!/usr/bin/env python
#---------------------------------------------------#
#   author		 : Dohyeon Yang @Dexter Studios		#
#	last updates : 2017.01.31						#
#---------------------------------------------------#

import os

import maya.cmds as cmds
import maya.mel as mel

def ZNCacheImportDialog():
    cur_workspace = cmds.workspace(query=True, rootDirectory=True)
    path = cmds.fileDialog2(fileMode = 3, 
                            caption = 'Select ZNCache Directory', 
                            okCaption = 'import',
                            startingDirectory=cur_workspace )
    if not path:
        return
    mode = cmds.optionVar(q='dxAbcImportMode')
    print mode
    path = path[0]
    ZNGroup = cmds.createNode('transform', name='%s_znGrp' % os.path.basename(path))

    for cache_folder in os.listdir(path):
        # print 'cache path : %s/%s' % (path, cache_folder)
        cache_path = '%s/%s' % (path, cache_folder)
        
        node = cmds.createNode('ZN_Archive')   # return shapenode
        
        cmds.setAttr('%s.cachePath' % node, cache_path, type='string')
        
        #cmds.addAttr(longName = 'rman__torattr___preShapeScript', dataType = 'string');
        #cmds.setAttr('%s.rman__torattr___preShapeScript' % node, 'dxarc', type='string',);
		
		#mel.eval('rmanAddAttr %s rman__torattr___preShapeScript dxarc' % node)
        
        parent_node = cmds.listRelatives(node, p=True)[0]
        parent_node = cmds.rename(parent_node, cache_folder)
        cmds.parent(parent_node, ZNGroup)
