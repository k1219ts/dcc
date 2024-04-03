# -*- coding: utf-8 -*-
import maya.cmds as cmds
import maya.mel as mel
import dxCameraUI
import sgComponent
import sgAssembly
import sgZenn
import sgUI
import os
import getpass
from dxstats import inc_tool_by_user
try:
    if not cmds.pluginInfo( 'RenderMan_for_Maya', q=True, l=True):
        cmds.loadPlugin('RenderMan_for_Maya')
    import lgtUI
except :
    print 'no renderman. pass'
    pass

MODEMAP = { 'mesh': 0, 'GPU': 1 }
WORLDMAP = { 'none': 0, 'baked': 1, 'seperate': 2 }

def getWorkSpace():
    show = None;
    seq = None;
    shot = None
    current = cmds.file(q=True, sn=True)
    if not current:
        current = cmds.workspace(q=True, rd=True)
    src = current.split('/')
    if 'show' in src:
        show = src[src.index('show') + 1]
    if 'shot' in src:
        seq = src[src.index('shot') + 1]
        shot = src[src.index('shot') + 2]
    return show, seq, shot

def importCamera(cameraPath, panzoomPath=None):
    # tool stats
    inc_tool_by_user.run('action.sceneSetupManager.import_camera', getpass.getuser())

    dxcam = cmds.createNode('dxCamera')
    dxCameraUI.import_cameraFile('%s.fileName' % dxcam, cameraPath, panzoomPath)
    cmds.setAttr( '%s.action' %dxcam, 1 )
    return dxcam

def importAssem(assemPath):
    # tool stats
    inc_tool_by_user.run('action.sceneSetupManager.import_assembly', getpass.getuser())

    #
    # set display modecreate asb node, set file path
    asbNode = cmds.createNode('ZAssemblyArchive')
    cmds.setAttr('%s.asbFilePath' % asbNode, assemPath, type="string")
    cmds.setAttr('%s.dispMode' % asbNode, 0)
    return asbNode
    # sgAssembly.importAssemblyFile(assemPath)

def importGeo( assetPath, worldOpt, alembicOpt):
    # tool stats
    inc_tool_by_user.run('action.sceneSetupManager.import_geoCache', getpass.getuser())

    mesh = MODEMAP[alembicOpt]
    world = WORLDMAP[worldOpt]
    myImport = sgUI.ComponentImport(
        Files=[assetPath],
        World=world )
    myImport.m_mode = mesh  # 0:mesh, 1:gpu
    myImport.m_display = 1  # 0:bbox, 1:render, 2:mid, 3:low, 4:sim
    myImport.doIt()

def importZenn(zennPath):
    # tool stats
    inc_tool_by_user.run('action.sceneSetupManager.import_zenn_simulation', getpass.getuser())

    zennGroup = cmds.createNode( 'zennArchive')
    cmds.setAttr('%s.cachePath' %zennGroup, zennPath, type='string')
    lgtUI.zennArchive_setCache('%s.cachePath' %zennGroup, zennPath)
    cmds.addAttr(zennGroup, ln='rman__torattr___preShapeScript', dt='string')
    cmds.setAttr('%s.rman__torattr___preShapeScript' % zennGroup, 'dxarc', type='string')
    geoGroup = os.path.basename(zennPath).split(':')[0]
    #geoGroup += '_rig_GRP'
    if cmds.objExists(geoGroup):
        conGroup = geoGroup
    elif cmds.objExists("%s:move_CON" %geoGroup):
        conGroup = "%s:move_CON" %geoGroup
    elif cmds.objExists("%s:world_CON" %geoGroup):
        conGroup = "%s:world_CON" %geoGroup
    else:
        return

    zennGroup = cmds.listRelatives(zennGroup, p=True)[0]
    if conGroup:
        cmds.parent(zennGroup, conGroup)
        zennGroup = cmds.rename(zennGroup, '%s_zennGrp'%os.path.basename(zennPath) )
        cmds.setAttr('%s.t' % zennGroup, 0, 0, 0)
        cmds.setAttr('%s.r' % zennGroup, 0, 0, 0)
        cmds.setAttr('%s.s' % zennGroup, 1, 1, 1)
        # connect initScale
        if cmds.attributeQuery('initScale', n=conGroup, ex=True):
            for i in ['sx', 'sy', 'sz']:
                cmds.connectAttr('%s.initScale' % conGroup, '%s.%s' % (zennGroup, i))

    cmds.select(cl=True)

def importZennStatic(namespace, abcPath, zennPath):
    # tool stats
    inc_tool_by_user.run('action.sceneSetupManager.import_zenn_static', getpass.getuser())

    rmanOutputProceduralName = 'rmanOutputZN_StrandsArchiveRigidBindingProcedrual'
    root = cmds.createNode('transform', name='%s_zennGrp' %namespace )
    for zenncachename in os.listdir(zennPath):
        node = cmds.createNode('ZN_StrandsArchive')
        cmds.connectAttr('time1.outTime', '%s.inTime' % node)
        cmds.addAttr(node, dt='string', ln='rman__torattr___preShapeScript')
        cmds.setAttr('%s.rman__torattr___preShapeScript' % node, rmanOutputProceduralName, type='string')
        cmds.setAttr('%s.inAbcCachePath' % node, abcPath, type='string')
        cmds.setAttr('%s.inZennCachePath' % node, zennPath, type='string')
        cmds.setAttr('%s.inZennCacheName' % node, zenncachename, type='string')
        cmds.setAttr('%s.drawStrands' % node, lock=1 )

        node = cmds.rename(node, '%s_%sShape' %( namespace, zenncachename ))
        parent = cmds.listRelatives(node, p=True, f=True)[0]
        parent = cmds.rename(parent, '%s_%s' %( namespace, zenncachename ))
        cmds.parent(parent, root)

    ### parent zennNode under rigGRP ###
    rigGRP = '%s_rig_GRP' % namespace
    if cmds.objExists(rigGRP):
        conGroup = cmds.listRelatives(rigGRP, p=True)
        if conGroup:
            cmds.parent(root, conGroup[0])
            cmds.setAttr('%s.t' % root, 0, 0, 0)
            cmds.setAttr('%s.r' % root, 0, 0, 0)
            cmds.setAttr('%s.s' % root, 1, 1, 1)
            # connect initScale
            if cmds.attributeQuery('initScale', n=conGroup[0], ex=True):
                for i in ['sx', 'sy', 'sz']:
                    cmds.connectAttr('%s.initScale' % conGroup[0], '%s.%s' % (root, i))
    cmds.select(cl=True)

### IMPORT CACHE ###
def importCache(
        startFrame=0, endFrame=0, camData={}, assemData={}, geoData={}, zennData={},
        alembicOpt='mesh', worldOpt = 'none', shot='' ):

    plugins = ['AbcImport', 'backstageMenu', 'ZENNForMaya', 'ZMayaTools',]
    for p in plugins:
        if not cmds.pluginInfo(p, q=True, l=True):
            cmds.loadPlugin(p)

    ### CAMERA ###
    if camData:
        panzoomPath = None
        for cameraPath in camData['camera_path']:
            if os.path.exists(cameraPath):
                if 'panzoom_json_path' in camData:
                    panzoomPath = camData['panzoom_json_path'][0]
                importCamera(cameraPath, panzoomPath)

    ### ASSEMBLY ###
    if assemData:
        for assemPath in assemData['path']:
            if os.path.exists(assemPath) and os.path.splitext(assemPath)[-1] == '.asb':
                importAssem(assemPath)

    ### GEOCACHE ###
    if geoData:
        for asset in geoData.keys():
            if not asset in ['maya_files','maya_dev_file']:
                if type(geoData[asset]) == dict and geoData[asset].has_key('path'):
                    assetPath = geoData[asset]['path'][0]
                    if os.path.exists(assetPath):
                        importGeo( assetPath, worldOpt, alembicOpt)

    ### ZENN ###
    if zennData:
        try:
            LGTplugins = ['RenderMan_for_Maya', 'backstageLight']
            for lp in LGTplugins:
                if not cmds.pluginInfo( lp, q=True, l=True):
                    cmds.loadPlugin(lp)
        except:
            print 'CANNOT LOAD RENDERMAN'
            return

        for asset in zennData:
            fn = zennData[asset]['zenn_path']
            ### hair simulation ###
            if not os.path.exists(fn[0]):
                print '### zenn file not exist !!!'
                continue
            if zennData[asset]['hairTask']:
                importZenn(fn[0])

            ### hair static ###
            if not zennData[asset]['hairTask']:
                ### find geoCache and zenn ###
                if asset in geoData:
                    abcPath = geoData[asset]['path'][0]
                    zennPath = zennData[asset]['zenn_path'][0]
                    zennNode = importZennStatic(asset, abcPath, zennPath)
                if not asset in geoData:
                    sel = cmds.ls('%s_rig_GRP' % asset)
                    if sel:
                        abcPath = cmds.getAttr( '%s.abcFileName' % sel[0] )
                    else:
                        continue
                    zennPath = zennData[asset]['zenn_path'][0]
                    zennNode = importZennStatic(asset, abcPath, zennPath)


    ### FRAME ###
    cmds.playbackOptions(minTime=startFrame)
    cmds.playbackOptions(maxTime=endFrame)
    cmds.playbackOptions(animationStartTime=startFrame)
    cmds.playbackOptions(animationEndTime=endFrame)
    cmds.currentTime(startFrame)

### IMPORT ASSET
def importAssetCache(assetData=None):
    plugins = ['AbcImport', 'backstageMenu', 'ZENNForMaya', 'ZMayaTools']
    for p in plugins:
        if not cmds.pluginInfo(p, q=True, l=True):
            cmds.loadPlugin(p)

    ### MODEL
    if 'model_path' in assetData:
        if os.path.exists( assetData['model_path'][0] ):
            cmd = '''AbcImport -mode import "%s"''' % assetData['model_path'][0]
            mel.eval(cmd)
    if 'assembly_path' in assetData:
        for i in assetData['assembly_path']:
            if os.path.exists(i):
                importAssem(i)
    if 'rig_path' in assetData:
        if os.path.exists( assetData['rig_path'][0] ):
            namespace = assetData['name']
            mel.eval('''file -r -gl -namespace \"%s\" -lrd \"all\" -options \"v=0\" \"%s\"'''
                % (namespace, assetData['rig_path'][0]))


### UPDATE CACHE ###
def updateCache(
        startFrame=None, endFrame=None, camData={}, assemData={}, geoData={}, zennData={},
        alembicOpt='mesh', worldOpt = 'none', shot='' ):

    plugins = ['AbcImport', 'backstageMenu', 'ZENNForMaya', 'ZMayaTools']
    for p in plugins:
        if not cmds.pluginInfo(p, q=True, l=True):
            cmds.loadPlugin(p)

    ### CHECK CAMERA ###
    if camData:
        dxcamNode = cmds.ls(type='dxCamera')
        if dxcamNode:
            camExistPath = cmds.getAttr( '%s.fileName' %dxcamNode[0] )
            for cameraPath in camData['camera_path']:
                if os.path.exists(cameraPath) and cameraPath != camExistPath:
                    dxCameraUI.import_cameraFile('%s.fileName' %dxcamNode[0], cameraPath)
        else:
            for cameraPath in camData['camera_path']:
                if os.path.exists(cameraPath):
                    importCamera(cameraPath)

    ### CHECK ASSEMBLY ###
    if assemData:
        assemNode = cmds.ls(type='ZAssemblyArchive')
        if assemNode:
            assemExistPath = cmds.getAttr('%s.asbFilePath'%assemNode[0])
            for assemPath in assemData['path']:
                if os.path.exists(assemPath) and assemPath != assemExistPath:
                    cmds.setAttr('%s.asbFilePath' %assemNode[0], assemPath, type='string')
        else:
            for assemPath in assemData['path']:
                if os.path.exists(assemPath):
                    importAssem(assemPath)

    ### CHECK GEO ###
    if geoData:
        for asset in geoData.keys():
            assetPath = geoData[asset]['path'][0]
            assetNode = cmds.ls('%s*' %asset, type='dxComponent')
            if os.path.exists(assetPath):
                if assetNode:
                    assetExistPath = cmds.getAttr( '%s.abcFileName' %assetNode[0] )
                    if assetPath != assetExistPath:
                        # sgcomponent update
                        sgComponent.componentImport('%s.abcFileName' %assetNode[0],
                                     assetPath)
                else:
                    importGeo(assetPath, worldOpt, alembicOpt)

    ### CHECK ZENN ###
    if zennData:
        LGTplugins = ['RenderMan_for_Maya', 'backstageLight']
        for lp in LGTplugins:
            if not cmds.pluginInfo(lp, q=True, l=True):
                cmds.loadPlugin(lp)

        for asset in zennData:
            zennPath = zennData[asset]['zenn_path'][0]
            zennNode = sel = cmds.ls( '%s_zennGrp' %asset )
            if os.path.exists(zennPath):
                if zennNode:
                    viewers = cmds.listRelatives( sel[0], typ='ZN_StrandsArchive', ad=1, pa=1)
                    ### static hair cache update ###
                    if viewers:
                        # delete and load
                        cmds.delete(zennNode)

                        if zennData[asset]['hairTask']:
                            importZenn(zennPath)

                        if not zennData[asset]['hairTask']:
                            if geoData:
                                geoPath = geoData[asset]['path'][0]
                            else:
                                continue
                            zennPath = zennData[asset]['zenn_path'][0]
                            zennNode = importZennStatic(asset, geoPath, zennPath)


                    ### simulation hair cache update ###
                    if not viewers:
                        if cmds.objExists('%sShape' % zennNode[0]):
                            zennExistPath = cmds.getAttr('%sShape.cachePath' % zennNode[0])
                            if zennExistPath != zennPath:
                                cmds.setAttr('%sShape.cachePath' % zennNode[0], zennPath, type='string')

                        if not zennData[asset]['hairTask']:
                            cmds.delete(zennNode)
                            if geoData:
                                geoPath = geoData[asset]['path'][0]
                            else:
                                continue
                            zennPath = zennData[asset]['zenn_path'][0]
                            zennNode = importZennStatic(asset, geoPath, zennPath)

                else:
                    ### import simulation hair cache ###
                    if zennData[asset]['hairTask']:
                        importZenn(zennPath)
                    ### import static hair cache ###
                    if not zennData[asset]['hairTask']:
                        ### get geoCache data ###
                        if geoData:
                            abcPath = geoData[asset]['path'][0]
                        if not geoData:
                            sel = cmds.ls( '%s_rig_GRP' %asset )
                            if sel:
                                abcPath = cmds.getAttr('%s.abcFileName' % sel[0])
                            else:
                                continue
                        zennPath = zennData[asset]['zenn_path'][0]
                        zennNode = importZennStatic(asset, abcPath, zennPath)










