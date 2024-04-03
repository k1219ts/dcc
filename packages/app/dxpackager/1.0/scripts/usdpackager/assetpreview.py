#coding:utf-8
import maya.cmds as cmds
import maya.mel as mel

import os, sys, shutil

def createPreviewCameras(cams=4, levels=3, levelInitScale=1, levelXZScale=2, levelYScale=2.5):
    cameraPrefix = 'previewCamera'
    cameraGroup = cameraPrefix+'_group'
    locatorName = cameraPrefix+'AimLocator'
    levelPrefix = cameraPrefix+'Level'
    levelGroup = levelPrefix+'_group'
    cmds.spaceLocator(n=locatorName)
    for l in range(1, levels+1):
        camLevelNum = str(l).zfill(2)
        curveName = levelPrefix+camLevelNum
        cmds.circle(ch=False, sw=360, s=8, r=levelInitScale, nr=(0, 1, 0), n=curveName)
        for n in range(1, cams+1):
            camNum = str(n).zfill(2)
            camName = cameraPrefix+camNum+'_'+camLevelNum
            camAimName = camName+'_aim'
            camGroupName = camName+'_group'
            cmds.camera(name=camName)
            mel.eval('cameraMakeNode 2 ""')
            camSh = cmds.listRelatives(camName, shapes=True)[0]
            cmds.setAttr(camSh+'.renderable', 1)
            cmds.createNode('pointOnCurveInfo', n=camName+'_pointOnCurveInfo')
            cmds.connectAttr(curveName+'.worldSpace[0]', camName+'_pointOnCurveInfo.inputCurve', f=True)
            cmds.setAttr(camName+'_pointOnCurveInfo.turnOnPercentage', 1)
            cmds.connectAttr(camName+'_pointOnCurveInfo.position', camName+'.translate', f=True)
            cmds.setAttr(camName+'_pointOnCurveInfo.parameter', float(1)/cams*n)
            cmds.parentConstraint(locatorName, camAimName, weight=1)

    firstBb = []
    minBb = [9999, 9999, 9999]
    maxBb = [-9999, -9999, -9999]
    centerBb = [0, 0, 0]
    scaleBb = [0, 0, 0]

    for sh in cmds.ls(type='mesh', long=True, visible=True)+cmds.ls(type='pxrUsdProxyShape', long=True, visible=True):
        if sh.count('Orig') > 0:
            continue
        meshTr = cmds.listRelatives(sh, f=True, parent=True)[0]
        shBb = cmds.xform(meshTr, bb=True, ws=True, q=True)
        if len(firstBb) < 1:
            firstBb = shBb
        shMinBb = shBb[0:3]
        shMaxBb = shBb[3:6]
        
        shScaleBb = [0, 0, 0]
        for shbmx in range(len(shMinBb)):
            shScaleBb[shbmx] = abs(shMaxBb[shbmx]-shMinBb[shbmx]) / 2
        
        if (shScaleBb[0]+shScaleBb[1])*10 < shScaleBb[2]:
            continue
        if (shScaleBb[0]+shScaleBb[2])*10 < shScaleBb[1]:
            continue
        if (shScaleBb[1]+shScaleBb[2])*10 < shScaleBb[0]:
            continue
                   
        for bmn in range(len(minBb)):
            if minBb[bmn] > shMinBb[bmn]:
                minBb[bmn] = shMinBb[bmn]
                
        for bmx in range(len(maxBb)):
            if maxBb[bmx] < shMaxBb[bmx]:
                maxBb[bmx] = shMaxBb[bmx]
    
    if minBb[0] == 9999 or maxBb[0] == -9999:
        minBb = firstBb[0:3]
        maxBb = firstBb[3:6]
        
    for bmx in range(len(maxBb)):
        centerBb[bmx] = sum([maxBb[bmx], minBb[bmx]]) / 2

    for bmx in range(len(maxBb)):
        scaleBb[bmx] = abs(maxBb[bmx]-minBb[bmx]) / 1
        
    pclList = cmds.ls(levelPrefix+'*', type='transform')
    reposList = pclList+[locatorName]
    rpScale = abs(scaleBb[0]) if abs(scaleBb[0]) > abs(scaleBb[2]) else abs(scaleBb[2])
    rpScale = rpScale if rpScale > abs(scaleBb[1]) else abs(scaleBb[1])
    for rp in reposList:
        cmds.setAttr(rp+'.tx', centerBb[0])
        cmds.setAttr(rp+'.ty', centerBb[1])
        cmds.setAttr(rp+'.tz', centerBb[2])
        rpxz = rpScale/levelInitScale*levelXZScale
        cmds.setAttr(rp+'.sx', rpxz)
        cmds.setAttr(rp+'.sz', rpxz)

    heightBb = maxBb[1] - minBb[1]
    pclLen = len(pclList)
    HeightStepBb = heightBb / (levels-1)
    for pcl in range(pclLen):
        cmds.setAttr(pclList[pcl]+'.ty', minBb[1]+(HeightStepBb*pcl))
        if pcl == 0 or pcl == pclLen-1:
            rpxzt = rpScale/levelInitScale*levelXZScale/1.25
            cmds.setAttr(pclList[pcl]+'.sx', rpxzt)
            cmds.setAttr(pclList[pcl]+'.sz', rpxzt)

    cmds.group(em=True, n=cameraGroup)
    cmds.delete(cmds.parentConstraint((locatorName, cameraGroup), w=1))
    cmds.makeIdentity(cameraGroup, apply=True, t=1, r=1, s=1, n=0, pn=1)
    cmds.parent(cmds.ls(cameraPrefix+'*_*_group')+[cameraGroup])

    cmds.group(em=True, n=levelGroup)
    cmds.delete(cmds.parentConstraint((locatorName, levelGroup), w=1))
    cmds.makeIdentity(levelGroup, apply=True, t=1, r=1, s=1, n=0, pn=1)
    cmds.parent(pclList+[locatorName, levelGroup])

    cmds.group(em=True, n=cameraPrefix+'_set')
    cmds.delete(cmds.parentConstraint((locatorName, cameraPrefix+'_set'), w=1))
    cmds.makeIdentity(cameraPrefix+'_set', apply=True, t=1, r=1, s=1, n=0, pn=1)
    cmds.parent([cameraGroup, levelGroup, cameraPrefix+'_set'])
    cmds.setAttr(levelGroup+'.sy', levelYScale)

def makePreviews(prevDir, lightIntensity=5, renderCams=[]):
    camShList = cmds.ls('previewCamera*_*', type='camera')
    for camSh in camShList:
        camShTr = cmds.listRelatives(camSh, parent=True)[0]
        camShTrLight = cmds.directionalLight(n=camShTr+'_light')
        cmds.setAttr(camShTr+'_lightShape.intensity', float(lightIntensity)/len(camShList))
        lightConst = cmds.parentConstraint((camShTr, camShTr+'_light'), weight=1)
        cmds.parent(camShTr+'_light', camShTr)
        cmds.delete(lightConst)

    cmds.setAttr('defaultRenderGlobals.currentRenderer', 'mayaSoftware', type='string')
    cmds.setAttr('defaultRenderGlobals.imageFormat', 8)
    cmds.setAttr('defaultResolution.width', 1280)
    cmds.setAttr('defaultResolution.height', 720)
    cmds.setAttr('defaultResolution.deviceAspectRatio', 1.777)
    # Get defaultRenderQuality attrs(Production, Highest)
    # for qv in cmds.listAttr('defaultRenderQuality'):
    #     try: print "cmds.setAttr('defaultRenderQuality."+qv+"', "+str(cmds.getAttr('defaultRenderQuality.'+qv))+")"
    #     except:pass
    cmds.setAttr('defaultRenderQuality.caching', False)
    cmds.setAttr('defaultRenderQuality.frozen', False)
    cmds.setAttr('defaultRenderQuality.isHistoricallyInteresting', 2)
    cmds.setAttr('defaultRenderQuality.nodeState', 0)
    # cmds.setAttr('defaultRenderQuality.binMembership', None)
    cmds.setAttr('defaultRenderQuality.reflections', 10)
    cmds.setAttr('defaultRenderQuality.refractions', 10)
    cmds.setAttr('defaultRenderQuality.shadows', 10)
    cmds.setAttr('defaultRenderQuality.rayTraceBias', 0.0)
    cmds.setAttr('defaultRenderQuality.edgeAntiAliasing', 0)
    cmds.setAttr('defaultRenderQuality.renderSample', False)
    cmds.setAttr('defaultRenderQuality.useMultiPixelFilter', True)
    cmds.setAttr('defaultRenderQuality.pixelFilterType', 2)
    cmds.setAttr('defaultRenderQuality.pixelFilterWidthX', 2.20000004768)
    cmds.setAttr('defaultRenderQuality.pixelFilterWidthY', 2.20000004768)
    cmds.setAttr('defaultRenderQuality.plugInFilterWeight', 1.0)
    cmds.setAttr('defaultRenderQuality.shadingSamples', 2)
    cmds.setAttr('defaultRenderQuality.maxShadingSamples', 8)
    cmds.setAttr('defaultRenderQuality.visibilitySamples', 1)
    cmds.setAttr('defaultRenderQuality.maxVisibilitySamples', 4)
    cmds.setAttr('defaultRenderQuality.volumeSamples', 1)
    cmds.setAttr('defaultRenderQuality.particleSamples', 1)
    cmds.setAttr('defaultRenderQuality.enableRaytracing', False)
    cmds.setAttr('defaultRenderQuality.redThreshold', 0.40000000596)
    cmds.setAttr('defaultRenderQuality.greenThreshold', 0.300000011921)
    cmds.setAttr('defaultRenderQuality.blueThreshold', 0.600000023842)
    cmds.setAttr('defaultRenderQuality.coverageThreshold', 0.125)
    
    for camSh in camShList:
        camTr = cmds.listRelatives(camSh, parent=True)[0]
        if len(renderCams) == 0 or camTr.split(':')[-1] in renderCams:
            outPath = cmds.render(camSh)
            try: os.makedirs(prevDir)
            except: pass
            imgPath = prevDir+'/'+camTr+'.jpg'
            print outPath, imgPath
            shutil.copy(outPath, imgPath)

# if __name__ == '__main__':
#     import maya.standalone
#     maya.standalone.initialize()

#     cmds.loadPlugin('pxrUsd')

#     if len(sys.argv) < 3:
#         os._exit(1)

#     srcPath = sys.argv[1]
#     imgDir = sys.argv[2]

#     if not os.path.isfile(srcPath):
#         os._exit(1)

#     try: os.makedirs(imgDir)
#     except: pass

#     cmds.file(srcPath, r=True, f=True)
#     # if not srcPath.endswith('.usd'):
#     #     cmds.file(srcPath, i=True, f=True, ignoreVersion=True, options='v=0;', mergeNamespacesOnClash=False, pr=True)
#     # else:
#     #     # pxrUsdRef = 'pxrUsdProxy'
#     #     pxrUsdSh = cmds.createNode('pxrUsdProxyShape')
#     #     cmds.setAttr(pxrUsdSh+'.filePath', srcPath, type='string')
#     #     pxrUsdTr = cmds.listRelatives(pxrUsdSh, parent=True)[0]
#     #     # cmds.rename(pxrUsdTr, pxrUsdRef)

#     createPreviewCameras()
#     srcFn = os.path.basename(df)
#     srcFnNoext = os.path.splitext(srcFn)[0]
#     cmds.file(rename=imgDir+'/'+srcFnNoext+'-preview.mb')
#     # cmds.file(s=True, f=True)
#     prevDir = imgDir+'/'+os.path.splitext(srcFn)[0]
#     makePreviews(prevDir)
#     os._exit(0)

# /backstage/dcc/DCC rez-env dxpackager pylibs-2.7 assetbrowser baselib-2.5 maya-2018 maya_animation maya_asset maya_layout maya_matchmove maya_rigging maya_toolkit dxrulebook-1.0.0 python-2 dxusd-2.0.0 dxusd_maya-1.0.0 usd_maya-19.11 -- mayapy /stdrepo/CSP/jungsup.han/scripts/usdpackager/assetpreview.py /show/pipe/_3d/asset/bariFace/rig/scenes/bariFace_rig_v003.mb /stdrepo/CSP/jungsup.han/test/assetpreview
# /backstage/dcc/DCC rez-env dxpackager pylibs-2.7 assetbrowser baselib-2.5 maya-2018 maya_animation maya_asset maya_layout maya_matchmove maya_rigging maya_toolkit dxrulebook-1.0.0 python-2 dxusd-2.0.0 dxusd_maya-1.0.0 usd_maya-19.11 -- mayapy /stdrepo/CSP/jungsup.han/scripts/usdpackager/assetpreview.py /show/cdh1/_3d/asset/aptBullet/aptBullet.usd /stdrepo/CSP/jungsup.han/test/assetpreview
# /backstage/dcc/DCC rez-env dxpackager pylibs-2.7 assetbrowser baselib-2.5 maya-2018 maya_animation maya_asset maya_layout maya_matchmove maya_rigging maya_toolkit dxrulebook-1.0.0 python-2 dxusd-2.0.0 dxusd_maya-1.0.0 usd_maya-19.11 -- mayapy /stdrepo/CSP/jungsup.han/scripts/usdpackager/assetpreview.py /show/slc/_3d/asset/e8dogA/e8dogA.usd /stdrepo/CSP/jungsup.han/test/assetpreview
# /backstage/dcc/DCC rez-env dxpackager pylibs-2.7 assetbrowser baselib-2.5 maya-2018 maya_animation maya_asset maya_layout maya_matchmove maya_rigging maya_toolkit dxrulebook-1.0.0 python-2 dxusd-2.0.0 dxusd_maya-1.0.0 usd_maya-19.11 -- mayapy /stdrepo/CSP/jungsup.han/scripts/usdpackager/assetpreview.py /show/cdh1/_3d/asset/endFactory/endFactory.usd /stdrepo/CSP/jungsup.han/test/assetpreview