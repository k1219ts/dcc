__author__ = 'gyeongheon.jeong'

import maya.cmds as cmds
import maya.mel as mm
import os
import json

MAYA_VER = os.getenv('MAYA_VER')

def DoRetime(SstartFrame_, SendFrame_, TimeScale_, camName):
    CamTransName = cmds.listRelatives(camName, parent=True, type='transform')[0]
    cmds.select(CamTransName)
    lockedAttrList = cmds.listAttr(CamTransName, locked=1)
    if lockedAttrList != None:
        for a in lockedAttrList:
            cmds.setAttr(CamTransName + "." + a ,lock=False)

    cmds.scaleKey(CamTransName, timeScale=TimeScale_, timePivot=int(SstartFrame_) )

    try:
        impName = cmds.listConnections(camName, type='imagePlane')[0]
        impNameShape = cmds.listRelatives(impName, s=1)
        expTemp = cmds.listConnections(impNameShape, type='expression')[0]
        cmds.delete(expTemp)

        cmds.setKeyframe(impName + ".frameExtension", v=float(SstartFrame_), t=float(SstartFrame_),
                         inTangentType = "linear", outTangentType = "linear" )
        cmds.setKeyframe( impName + ".frameExtension", v=float(SendFrame_), t=float(SendFrame_) )
        cmds.keyTangent(itt='linear', ott='linear')
        cmds.scaleKey(impName + ".frameExtension", timeScale=TimeScale_, timePivot=int(SstartFrame_) )
    except:
        pass

def writeCamLog(project=None, filmBack_inch=list(), filmBack_mm=list(), resolution_gate=list(),
                preview_size=list(), letterBox_size=list(), lens_main=list(), lens_others=list(),
                jsonFile=str()):
    cameraInfo = dict()
    cameraInfo["filmBack_inch"] = [1.007229, 0.531156]
    cameraInfo["filmBack_mm"] = [25.5836, 13.4914]
    cameraInfo["resolution_gate"] = [2048, 1080]
    cameraInfo["preview_size"] = [1280, 676]
    cameraInfo["letterBox_size"] = [2048, 878]
    cameraInfo["lens_main"] = [12, 18, 21, 25, 35]
    cameraInfo["lens_others"] = [8, 10, 12, 14, 16, 18, 21, 25, 27, 32, 35,
                                 40, 50, 65, 75, 85, 100, 135, 150, 200, 290, 340]

    projectInfo = dict()
    projectInfo["project"] = dict()
    project = projectInfo["project"]
    project["WZK"] = cameraInfo

    jsonFile = "/home/gyeongheon.jeong/Desktop/WKZ_redDragon5KFF_camera_v01.info"

    with open(jsonFile, 'w') as f:
        json.dump(projectInfo, f, indent=4)
        f.close()

def createShotCam( shotName, camInfoJson, camtype, flens, isRenderCam):

    camName = shotName + "_ani_renderCam"
    camExistsNum = len(cmds.ls(camName + "*", type="camera"))

    if camExistsNum >= 1:
        camName = camName + str(camExistsNum)

    groupNodeName = camName + "_GRP"
    camTranslateLOCName = camName + "_TRA"
    camRotateLOCName = camName + "_ROT"
    # camLOCName = camName + "_LOC"
    camInfoDic = camInfoJson["CAMERAS"][camtype]
    horizontalFilmAperture = camInfoDic["filmBack_inch"][0]
    verticalFilmAperture = camInfoDic["filmBack_inch"][1]

    cmds.select(cl=True)
    camGroupNode = str(cmds.createNode("transform", n=groupNodeName))
    camTranslateLOC = cmds.spaceLocator(n=camTranslateLOCName)[0]
    camRotateLOC = cmds.spaceLocator(n=camRotateLOCName)[0]

    oldcamNode = cmds.createNode("camera")
    camNodeTrans = cmds.listRelatives(oldcamNode, p=True)[0]
    camRenamed = cmds.rename(camNodeTrans, camName)
    camNode = cmds.listRelatives(camRenamed, s=True)[0]

    if isRenderCam:
        camList = cmds.ls(type="camera")
        for cam in camList:
            cmds.setAttr(cam + ".renderable", False)
        cmds.setAttr(camNode + ".renderable", True)

        cmds.setAttr("defaultResolution.width",
                     camInfoJson["CAMERAS"][camtype]["resolution_gate"][0])
        cmds.setAttr("defaultResolution.height",
                     camInfoJson["CAMERAS"][camtype]["resolution_gate"][1])
        cmds.setAttr("defaultResolution.deviceAspectRatio",
                     camInfoJson["CAMERAS"][camtype]["resolution_gate"][0] / float(
                         camInfoJson["CAMERAS"][camtype]["resolution_gate"][1]))
    else:
        cmds.setAttr(camNode + ".renderable", False)

    imageplaneNode = cmds.imagePlane(camera=camNode)
    cmds.imagePlane(imageplaneNode[1], e=1, fileName=camInfoJson["CAMERAS"][camtype]["letter_box"])
    cmds.connectAttr(camNode + ".message", imageplaneNode[1] + ".lookThroughCamera", f=True)
    cmds.setAttr(imageplaneNode[1] + ".displayOnlyIfCurrent", True)
    cmds.setAttr(imageplaneNode[1] + ".fit", 4)
    cmds.setAttr(imageplaneNode[1] + ".sizeX", horizontalFilmAperture)
    cmds.setAttr(imageplaneNode[1] + ".sizeY",
                 horizontalFilmAperture \
                 / (camInfoJson["CAMERAS"][camtype]["resolution_gate"][0] / float(
                     camInfoJson["CAMERAS"][camtype]["resolution_gate"][1])))
    cmds.setAttr(imageplaneNode[1] + ".depth", 1)
    imgpNodeTrans = cmds.listRelatives(imageplaneNode[1], p=True)[0]
    cmds.rename(imgpNodeTrans, camRenamed + "_letterBox")

    cmds.parent(camRenamed, camRotateLOC)
    cmds.parent(camRotateLOC, camTranslateLOC)
    cmds.parent(camTranslateLOC, camGroupNode)

    cmds.addAttr(groupNodeName, longName="notes", dataType="string")
    cmds.setAttr(groupNodeName + ".notes", camInfoDic["Note"], type="string")
    cmds.setAttr(camRenamed + ".rotateOrder", 2)
    cmds.setAttr(camNode + ".focalLength", float(flens))
    cmds.setAttr(camNode + ".horizontalFilmAperture", horizontalFilmAperture, lock=True)
    cmds.setAttr(camNode + ".verticalFilmAperture", verticalFilmAperture, lock=True)
    cmds.setAttr(camNode + ".farClipPlane", 100000)
    cmds.setAttr(camTranslateLOC + ".rx", lock=True)
    cmds.setAttr(camTranslateLOC + ".rz", lock=True)
    cmds.setAttr(camTranslateLOC + ".sx", lock=True, keyable=False)
    cmds.setAttr(camTranslateLOC + ".sy", lock=True, keyable=False)
    cmds.setAttr(camTranslateLOC + ".sz", lock=True, keyable=False)
    cmds.setAttr(camTranslateLOC + ".visibility", lock=True, keyable=False)
    cmds.setAttr(camRotateLOC + ".sx", lock=True, keyable=False)
    cmds.setAttr(camRotateLOC + ".sy", lock=True, keyable=False)
    cmds.setAttr(camRotateLOC + ".sz", lock=True, keyable=False)
    cmds.setAttr(camRotateLOC + ".visibility", lock=True, keyable=False)
    cmds.setAttr(camRotateLOC + ".rz", lock=True)

    cmds.select(camRenamed)

def doCamProjection(selObj, camName ):
    imagePlaneName = cmds.listConnections(camName, type='imagePlane')[0]
    ImageFileLoc = cmds.getAttr('%s.imageName' % imagePlaneName)
    FOffset = cmds.getAttr('%s.frameOffset' % imagePlaneName)

    # create surface shader
    surfShaderName = cmds.shadingNode('surfaceShader', asShader=True)
    ShadingGroupName = cmds.sets(renderable=True,
                                 noSurfaceShader=True,
                                 empty=True, name=surfShaderName + "SG")
    cmds.connectAttr('%s.outColor' % surfShaderName,
                     '%s.surfaceShader' % ShadingGroupName, f=True)

    # create projection imageFile
    if float(MAYA_VER) >= 2016:
        prjFileName = cmds.shadingNode('file',
                                       name="PrjImgFile",
                                       asTexture=True,
                                       isColorManaged=True)
        cmds.setAttr('{0}.colorSpace'.format(prjFileName), 'sRGB', type='string')
    else:
        prjFileName = cmds.shadingNode('file', name="PrjImgFile", asTexture=True)

    cmds.setAttr('%s.fileTextureName' % prjFileName, ImageFileLoc, type="string")
    cmds.setAttr('%s.useFrameExtension' % prjFileName, 1)

    if FOffset != 0:
        cmds.setAttr('%s.frameOffset' % prjFileName, FOffset)

    P2T = cmds.shadingNode('place2dTexture', asUtility=True)

    cmds.connectAttr('%s.coverage' % P2T, '%s.coverage' % prjFileName, f=True)
    cmds.connectAttr('%s.translateFrame' % P2T, '%s.translateFrame' % prjFileName, f=True)
    cmds.connectAttr('%s.rotateFrame' % P2T, '%s.rotateFrame' % prjFileName, f=True)
    cmds.connectAttr('%s.mirrorU' % P2T, '%s.mirrorU' % prjFileName, f=True)
    cmds.connectAttr('%s.mirrorV' % P2T, '%s.mirrorV' % prjFileName, f=True)
    cmds.connectAttr('%s.stagger' % P2T, '%s.stagger' % prjFileName, f=True)
    cmds.connectAttr('%s.wrapU' % P2T, '%s.wrapU' % prjFileName, f=True)
    cmds.connectAttr('%s.wrapV' % P2T, '%s.wrapV' % prjFileName, f=True)
    cmds.connectAttr('%s.repeatUV' % P2T, '%s.repeatUV' % prjFileName, f=True)
    cmds.connectAttr('%s.offset' % P2T, '%s.offset' % prjFileName, f=True)
    cmds.connectAttr('%s.rotateUV' % P2T, '%s.rotateUV' % prjFileName, f=True)
    cmds.connectAttr('%s.noiseUV' % P2T, '%s.noiseUV' % prjFileName, f=True)
    cmds.connectAttr('%s.vertexUvOne' % P2T, '%s.vertexUvOne' % prjFileName, f=True)
    cmds.connectAttr('%s.vertexUvTwo' % P2T, '%s.vertexUvTwo' % prjFileName, f=True)
    cmds.connectAttr('%s.vertexUvThree' % P2T, '%s.vertexUvThree' % prjFileName, f=True)
    cmds.connectAttr('%s.vertexCameraOne' % P2T, '%s.vertexCameraOne' % prjFileName, f=True)
    cmds.connectAttr( '%s.outUV' % P2T, '%s.uv' % prjFileName)
    cmds.connectAttr( '%s.outUvFilterSize' % P2T, '%s.uvFilterSize' % prjFileName)

    #create projection node
    projectionNodeName = cmds.shadingNode('projection', asUtility = True)

    cmds.connectAttr('%s.outColor' % prjFileName, '%s.image' % projectionNodeName, f = True)
    cmds.connectAttr('%s.outColor' % projectionNodeName, '%s.outColor' % surfShaderName, f = True)

    if ImageFileLoc.endswith("png"):
        cmds.connectAttr('%s.outTransparency' % prjFileName, '%s.transparency' % projectionNodeName, f = True)
        cmds.connectAttr('%s.outTransparency' % projectionNodeName, '%s.outTransparency' % surfShaderName, f = True)

    cmds.setAttr('%s.projType' % projectionNodeName, 8)
    cmds.connectAttr('%s.message' % camName, '%s.linkedCamera' % projectionNodeName, f = True)
    #cmds.listConnections(projectionNodeName + ".image", source = True, destination = False)
    #cmds.listConnections(projectionNodeName + ".outColor", source = True, destination = False)
    #cmds.listConnections(projectionNodeName + ".linkedCamera", source = True, destination = False)

    for obj in selObj:
        cmds.sets(obj, e = True, forceElement = ShadingGroupName)

    cmds.select(prjFileName)
    mm.eval("openAEWindow;")


# ======================================================================================================================= #

def DoBakeTexture(SelMaterial, BakeRes):
    SceneFullName_ = cmds.file(q=1, sn=1)
    BakedPrjImgPath = []
    for i in range( len(SelMaterial) ):
        indexNum = SelMaterial[i].find(":")

        if indexNum != -1:
            indexNum += 1
            NewSelMaterial = SelMaterial[i][indexNum:]

        else:
            NewSelMaterial = SelMaterial[i]

        BakedPrjImgPath.append("%s/data/convertedTexture/%s/" % ( os.sep.join( SceneFullName_.split(os.sep)[:-2]), NewSelMaterial ))

        if not os.path.exists( BakedPrjImgPath[i] ):os.makedirs(BakedPrjImgPath[i])

    PrjStartFrame_ = int(cmds.playbackOptions(q=1, min = 1))
    PrjEndFrame_ = int(cmds.playbackOptions(q=1, max = 1))

    for Ctime in range(PrjStartFrame_, PrjEndFrame_ + 1):
        if Ctime == PrjStartFrame_:cmds.undoInfo( state=True )
        else:
            cmds.undoInfo( state=False )

        cmds.currentTime(Ctime, e = 1)

        for objNum in range( len(SelMaterial) ):
            BakedPrjImgName = '%s_BakedProjectionImage.%04d.jpg' % ( SelMaterial[objNum], Ctime )

            SelMaterialShape = cmds.listRelatives( SelMaterial[objNum] )[0]

            shaderSG = cmds.listConnections(SelMaterialShape, type = "shadingEngine", source =0, destination = 1)[0]

            shader = cmds.listConnections(shaderSG, type = "surfaceShader")[0]

            projectionNode = cmds.listConnections(shader, type = "projection")[0]

            TempTextureFile = cmds.convertSolidTx( projectionNode + '.outColor', SelMaterial[objNum], antiAlias = 0,
                                                                    bm = 1, fts = 1, sp = 0, sh = 0,
                                                                    alpha = 1,
                                                                    doubleSided = 0,
                                                                    componentRange = 0,
                                                                    resolutionX = BakeRes, resolutionY = BakeRes,
                                                                    fileFormat = "png",
                                                                    fileImageName = BakedPrjImgPath[objNum] + BakedPrjImgName )

            cmds.delete(TempTextureFile)

    cmds.undoInfo( state=True )

    NewSurfShaderName = []
    NewShadingGroupName = []
    NewBakedTextureFileName = []
    NewP2T = []
    cmtr = []

    for objs in range( len(SelMaterial) ):
        NewSurfShaderName.append( cmds.shadingNode('surfaceShader', asShader = True) )
        NewShadingGroupName.append( cmds.sets(renderable = True, noSurfaceShader = True, empty = True, name = NewSurfShaderName[objs] + "SG") )
        cmds.connectAttr('%s.outColor' % NewSurfShaderName[objs], '%s.surfaceShader' % NewShadingGroupName[objs], f = True)

        NewBakedTextureFileName.append( cmds.shadingNode('file', asTexture=1 ) )
        cmds.setAttr(NewBakedTextureFileName[objs] + ".fileTextureName", BakedPrjImgPath[objs] + "%s_BakedProjectionImage.%04d.jpg" % (SelMaterial[objs], PrjStartFrame_), type = "string")
        cmds.setAttr("%s.useFrameExtension" % NewBakedTextureFileName[objs], 1)

        NewP2T.append( cmds.shadingNode('place2dTexture', asUtility = True) )

        cmds.connectAttr('%s.coverage' % NewP2T[objs], '%s.coverage' % NewBakedTextureFileName[objs], f=True)
        cmds.connectAttr('%s.translateFrame' % NewP2T[objs], '%s.translateFrame' % NewBakedTextureFileName[objs], f=True)
        cmds.connectAttr('%s.rotateFrame' % NewP2T[objs], '%s.rotateFrame' % NewBakedTextureFileName[objs], f=True)
        cmds.connectAttr('%s.mirrorU' % NewP2T[objs], '%s.mirrorU' % NewBakedTextureFileName[objs], f=True)
        cmds.connectAttr('%s.mirrorV' % NewP2T[objs], '%s.mirrorV' % NewBakedTextureFileName[objs], f=True)
        cmds.connectAttr('%s.stagger' % NewP2T[objs], '%s.stagger' % NewBakedTextureFileName[objs], f=True)
        cmds.connectAttr('%s.wrapU' % NewP2T[objs], '%s.wrapU' % NewBakedTextureFileName[objs], f=True)
        cmds.connectAttr('%s.wrapV' % NewP2T[objs], '%s.wrapV' % NewBakedTextureFileName[objs], f=True)
        cmds.connectAttr('%s.repeatUV' % NewP2T[objs], '%s.repeatUV' % NewBakedTextureFileName[objs], f=True)
        cmds.connectAttr('%s.offset' % NewP2T[objs], '%s.offset' % NewBakedTextureFileName[objs], f=True)
        cmds.connectAttr('%s.rotateUV' % NewP2T[objs], '%s.rotateUV' % NewBakedTextureFileName[objs], f=True)
        cmds.connectAttr('%s.noiseUV' % NewP2T[objs], '%s.noiseUV' % NewBakedTextureFileName[objs], f=True)
        cmds.connectAttr('%s.vertexUvOne' % NewP2T[objs], '%s.vertexUvOne' % NewBakedTextureFileName[objs], f=True)
        cmds.connectAttr('%s.vertexUvTwo' % NewP2T[objs], '%s.vertexUvTwo' % NewBakedTextureFileName[objs], f=True)
        cmds.connectAttr('%s.vertexUvThree' % NewP2T[objs], '%s.vertexUvThree' % NewBakedTextureFileName[objs], f=True)
        cmds.connectAttr('%s.vertexCameraOne' % NewP2T[objs], '%s.vertexCameraOne' % NewBakedTextureFileName[objs], f=True)
        cmds.connectAttr( '%s.outUV' % NewP2T[objs], '%s.uv' % NewBakedTextureFileName[objs])
        cmds.connectAttr( '%s.outUvFilterSize' % NewP2T[objs], '%s.uvFilterSize' % NewBakedTextureFileName[objs])

        cmds.connectAttr( '%s.outColor' % NewBakedTextureFileName[objs], '%s.outColor' % NewSurfShaderName[objs], f =True )

        cmds.sets(SelMaterial[objs], e = 1, forceElement = NewShadingGroupName[objs])

        cmtr.append( cmds.listConnections(NewSurfShaderName[objs] + ".outColor", type = "file")[0] )

        messagePlug = cmds.listConnections(NewSurfShaderName[objs], type = "materialInfo")

        connections = cmds.listConnections("%s.texture" % messagePlug[0], c = 1)

        tempNumA = len(connections)
        while tempNumA > 0:
            cmds.disconnectAttr(connections[tempNumA-1] + ".message", connections[tempNumA-2])
            tempNumA -= 2

        cmds.connectAttr( cmtr[objs] + ".message", messagePlug[0]+".texture[0]")
        #mm.eval('AEhardwareTextureChannelCB "outColor outTransparency" %s.message' % NewSurfShaderName)

    mm.eval('hyperShadePanelMenuCommand("hyperShadePanel1", "deleteUnusedNodes");')

    for i in range( len(SelMaterial) ):
        cmds.select( NewSurfShaderName[i] )
        cmds.select( cmtr[i] )
        mm.eval("showEditor %s" % cmtr[i])

def DoCreateImageplane(imagePlaneName, selCam):
    if not selCam:
        cmds.confirmDialog( title='Error',
                            message='Select Camera First',
                            button=['I Got It'],
                            defaultButton='I Got It' )
        return

    objType = cmds.objectType(selCam)

    if objType != "camera":
        cmds.confirmDialog( title='Error',
                            message='Select Camera First',
                            button=['OK'], defaultButton='OK' )
        return

    GBLBarImgplane = cmds.imagePlane(name = "BAR_plane_", camera = selCam, fileName = imagePlaneName)
    cmds.imagePlane( GBLBarImgplane[1], e = 1, lookThrough = selCam, showInAllViews = False)
    impSize = cmds.getAttr(GBLBarImgplane[-1] + '.size')[0]
    panzoomToSizeMultiplydivide = cmds.createNode( 'multiplyDivide', n='panToSize_MTV' )
    panzoomToSizeCondition = cmds.createNode( 'condition', n='panToSize_CDN' )
    panzoomToOffsetCondition = cmds.createNode( 'condition', n='panToOffset_CDN' )

    cmds.setAttr(panzoomToSizeMultiplydivide + '.input1X', impSize[0])
    cmds.setAttr(panzoomToSizeMultiplydivide + '.input1Y', impSize[1])
    cmds.setAttr(panzoomToSizeCondition + '.secondTerm', 1)
    cmds.setAttr(panzoomToOffsetCondition + '.secondTerm', 1)
    cmds.setAttr(panzoomToOffsetCondition + '.colorIfFalseR', 0)
    cmds.setAttr(panzoomToOffsetCondition + '.colorIfFalseG', 0)
    cmds.setAttr( GBLBarImgplane[1] + ".depth", 1 )

    cmds.connectAttr(selCam + '.panZoomEnabled', panzoomToOffsetCondition + '.firstTerm')
    cmds.connectAttr( selCam + '.horizontalPan', panzoomToOffsetCondition + '.colorIfTrueR' )
    cmds.connectAttr( selCam + '.verticalPan', panzoomToOffsetCondition + '.colorIfTrueG' )

    cmds.connectAttr(selCam + '.panZoomEnabled', panzoomToSizeCondition + '.firstTerm')
    cmds.connectAttr(selCam + '.zoom', panzoomToSizeMultiplydivide + '.input2X')
    cmds.connectAttr(selCam + '.zoom', panzoomToSizeMultiplydivide + '.input2Y')

    cmds.connectAttr(panzoomToSizeMultiplydivide + '.outputX', panzoomToSizeCondition + '.colorIfTrueR')
    cmds.connectAttr(panzoomToSizeMultiplydivide + '.outputY', panzoomToSizeCondition + '.colorIfTrueG')
    cmds.connectAttr(panzoomToSizeMultiplydivide + '.input1X', panzoomToSizeCondition + '.colorIfFalseR')
    cmds.connectAttr(panzoomToSizeMultiplydivide + '.input1Y', panzoomToSizeCondition + '.colorIfFalseG')

    cmds.connectAttr(panzoomToSizeCondition + '.outColorR', GBLBarImgplane[-1] + '.sizeX')
    cmds.connectAttr(panzoomToSizeCondition + '.outColorG', GBLBarImgplane[-1] + '.sizeY')
    cmds.connectAttr(panzoomToOffsetCondition + '.outColorR', GBLBarImgplane[-1] + '.offsetX')
    cmds.connectAttr(panzoomToOffsetCondition + '.outColorG', GBLBarImgplane[-1] + '.offsetY')



def DoTextureResolution(objectName, resolution):
    objectName = cmds.ls(sl=1)

    for obj in objectName:
        objShape = cmds.listRelatives(obj, s=1)[0]
        SGname = cmds.listConnections(objShape, type = "shadingEngine")

        S_Shader = cmds.listConnections(SGname, d=0, type = "surfaceShader")

        if "resolution" not in cmds.listAttr(S_Shader[0]):
            cmds.addAttr(S_Shader[0], ln="resolution", nn = "Resolution")
        cmds.setAttr(S_Shader[0] + ".resolution", resolution)

        print S_Shader[0] + ".resolution", resolution

        """
        try:
            prjNodeName = cmds.listConnections(S_Shader, d=0)
            txtr_filename = cmds.listConnections(prjNodeName, d=0, type = "file")[0]

            if "resolution" not in cmds.listAttr(txtr_filename):
                cmds.addAttr(txtr_filename, ln="resolution", nn = "Resolution")
            cmds.setAttr(txtr_filename + ".resolution", resolution)
            print txtr_filename + ".resolution"

        except:
            txtr_filename = cmds.listConnections(S_Shader, d=0, type = "file")[0]

            if "resolution" not in cmds.listAttr(txtr_filename):
                cmds.addAttr(txtr_filename, ln="resolution", nn = "Resolution")
            cmds.setAttr(txtr_filename + ".resolution", resolution)
            #print txtr_filename + ".resolution"
            """


def createFollowCam(object, Moves=[True, True, True], newCam=True, selCam = None):
    pluginLoaded = cmds.pluginInfo("matrixNodes", q=True, loaded=True)

    if not pluginLoaded:
        cmds.loadPlugin("matrixNodes")

    panelName = cmds.getPanel(wf=1)
    camT = cmds.modelPanel(panelName, q=1, cam=1)
    camT_world = cmds.xform(camT, q=1, rp=1, ws=1)
    camR = cmds.xform(camT, q=1, ro=1)

    if newCam:
        camName = cmds.camera(name="GH_followCAM")
        grpName = cmds.group(camName[0], name="GH_followCAM_GRP")

        cmds.addAttr(grpName, ln="muteX", at="long", min=0, max=1, dv=0)
        cmds.setAttr("%s.muteX" % grpName, e=1, keyable=1)
        cmds.addAttr(grpName, ln="muteY", at="long", min=0, max=1, dv=0)
        cmds.setAttr("%s.muteY" % grpName, e=1, keyable=1)
        cmds.addAttr(grpName, ln="muteZ", at="long", min=0, max=1, dv=0)
        cmds.setAttr("%s.muteZ" % grpName, e=1, keyable=1)

        mmxNode = cmds.shadingNode("multMatrix", name="%s_GHFC_MMX" % grpName, asUtility=True)
        dcmNode = cmds.shadingNode("decomposeMatrix", name="%s_GHFC_DCM" % grpName, asUtility=True)

        XmuteNode = cmds.createNode("mute", name="mute_%s_tx" % grpName)
        YmuteNode = cmds.createNode("mute", name="mute_%s_ty" % grpName)
        ZmuteNode = cmds.createNode("mute", name="mute_%s_tz" % grpName)

        cmds.connectAttr("%s.worldMatrix[0]" % object, "%s.matrixIn[0]" % mmxNode, f=True)
        cmds.connectAttr("%s.matrixSum" % mmxNode, "%s.inputMatrix" % dcmNode, f=True)
        cmds.connectAttr("%s.outputTranslateX" % dcmNode, "%s.input" % XmuteNode, f=True)
        cmds.connectAttr("%s.outputTranslateY" % dcmNode, "%s.input" % YmuteNode, f=True)
        cmds.connectAttr("%s.outputTranslateZ" % dcmNode, "%s.input" % ZmuteNode, f=True)

        cmds.connectAttr("%s.output" % XmuteNode, "%s.translateX" % grpName, f=True)
        cmds.connectAttr("%s.output" % YmuteNode, "%s.translateY" % grpName, f=True)
        cmds.connectAttr("%s.output" % ZmuteNode, "%s.translateZ" % grpName, f=True)

        cmds.setAttr("%s.muteX" % grpName, int(not Moves[0]))
        cmds.setAttr("%s.muteY" % grpName, int(not Moves[1]))
        cmds.setAttr("%s.muteZ" % grpName, int(not Moves[2]))

        cmds.connectAttr("%s.muteX" % grpName, "%s.mute" % XmuteNode, f=1)
        cmds.connectAttr("%s.muteY" % grpName, "%s.mute" % YmuteNode, f=1)
        cmds.connectAttr("%s.muteZ" % grpName, "%s.mute" % ZmuteNode, f=1)

        cmds.connectAttr("%s.parentInverseMatrix[0]" % grpName, "%s.matrixIn[1]" % mmxNode)

        cmds.xform(camName[0], t=(camT_world[0], camT_world[1], camT_world[2]), ro=(camR[0], camR[1], camR[2]), ws=1)

        cmds.select(grpName)
    else:
        grpName = cmds.listRelatives(selCam, p=1)
        AllConnectedNodes = cmds.listConnections(grpName[0])
        mmxNode = None

        for i in AllConnectedNodes:
            if cmds.nodeType(i) == "multMatrix":
                mmxNode = i

        cmds.connectAttr("%s.worldMatrix[0]" % object, "%s.matrixIn[0]" % mmxNode, f=True)

        cmds.setAttr("%s.muteX" % grpName[0], int(not Moves[0]))
        cmds.setAttr("%s.muteY" % grpName[0], int(not Moves[1]))
        cmds.setAttr("%s.muteZ" % grpName[0], int(not Moves[2]))

        cmds.xform(selCam, t=(camT_world[0], camT_world[1], camT_world[2]), ro=(camR[0], camR[1], camR[2]), ws=1)
