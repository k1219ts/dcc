


def camShake2D_Build():
    import maya.cmds as cmds
    import maya.mel as mel

    if cmds.window("camShake2D_", q=True, ex=True):
        cmds.deleteUI("camShake2D_", window=True)

    cmds.window("camShake2D_", sizeable=False, mb=1)
    cmds.menu("camShake2D_DispMenuSet", label='Display', tearOff=False)
    rsm1 = cmds.menuItem(label='Film Gate', checkBox=0, en=False)
    rsm2 = cmds.menuItem(label='Resolution Gate', checkBox=0, en=0)
    rsm3 = cmds.menuItem(label='Field Chart', checkBox=0, en=0)
    rsm4 = cmds.menuItem(label='Safe Action', checkBox=0, en=0)
    rsm5 = cmds.menuItem(label='Safe Title', checkBox=0, en=0)
    rsm6 = cmds.menuItem(label='View Window', checkBox=0, en=0)
    cmds.menu("camShake2D_ToolsMenuSet", label='Tools', tearOff=False)
    cmds.menuItem(divider=True)
    tempMenuItem = cmds.menuItem(label='Add/Remove Camera Shake')


    rsCamUIMainColumn = cmds.columnLayout(columnAttach=('both', 10), adj=True)

    cmds.columnLayout(columnAttach=('both', 1), adj=True)

    cmds.separator(style="none", h=10)
    cOpt = cmds.optionMenu(label='Camera')
    cmds.menuItem(label='--Unlink Camera--')
    tempAllCams = cmds.ls(cameras=True)
    for tempCam in tempAllCams:
        if cmds.getAttr(tempCam + ".orthographic") == 0:
            tempShape = tempCam
            tempTrans = (mel.eval('listTransforms "' + tempCam + '";'))
            cmds.menuItem(label=tempTrans[0])

    cmds.optionMenu(cOpt, e=1, cc=(
        'import maya.cmds as cmds; camShake2D.camShake2D_Fill((cmds.optionMenu("' + cOpt + '",q=1,v=1)), "' + rsm1 + '","' + rsm2 + '","' + rsm3 + '","' + rsm4 + '","' + rsm5 + '","' + rsm6 + '","' + rsCamUIMainColumn + '")'))
    cmds.setParent("..")
    cmds.separator(style="none", h=10)
    cmds.showWindow("camShake2D_")
    cmds.window("camShake2D_", e=1, wh=[235, 305])
    cmds.menuItem(tempMenuItem, e=1, c=(
        'import maya.cmds as cmds; camShake2D.camShake2D_ShakeStart("' + cOpt + '",(cmds.optionMenu("' + cOpt + '",q=1,v=1)), "' + rsm1 + '","' + rsm2 + '","' + rsm3 + '","' + rsm4 + '","' + rsm5 + '","' + rsm6 + '","' + rsCamUIMainColumn + '")'))


def camShake2D_ShowWindow(rsm6, camT, rsCamUIMainColumn):
    import maya.cmds as cmds

    tempTrial = cmds.about(v=True)
    if len(tempTrial) < 4:
        tempTrial = tempTrial + "____"
    tempTrialer = tempTrial[0:4]
    cmds.setParent(rsCamUIMainColumn)
    tempX = cmds.window("camShake2D_", q=1, h=True)
    tempY = cmds.menuItem(rsm6, q=1, checkBox=1)
    if tempY == 1:
        if tempTrialer == "2010" or tempTrialer == "2009":
            cmds.window("camShake2D_", e=1, h=(tempX + 228))
            cmds.paneLayout("camShake2D_play", w=206, h=188)
        else:
            cmds.window("camShake2D_", e=1, h=(tempX + 208))
            cmds.paneLayout("camShake2D_play", w=206, h=166)
        shotCamera = cmds.modelPanel(mbv=0, camera=camT)
        cmds.modelEditor(shotCamera, edit=True, grid=False, da="smoothShaded")
        cmds.modelEditor(shotCamera, edit=True, allObjects=False)
        cmds.modelEditor(shotCamera, edit=True, nurbsSurfaces=True)
        cmds.modelEditor(shotCamera, edit=True, polymeshes=True)
        cmds.modelEditor(shotCamera, edit=True, subdivSurfaces=True)
        if tempTrialer == "2010" or tempTrialer == "2009":
            cmds.setParent("..")
            cmds.setParent("..")
        cmds.setParent("..")
        cmds.setParent("..")
        cmds.setParent("..")
        myTimeCtrl = cmds.timePort('myTimePort', w=100, h=20)


    else:
        if tempTrialer == "2010" or tempTrialer == "2009":
            cmds.window("camShake2D_", e=1, h=(tempX - 228))
        else:
            cmds.window("camShake2D_", e=1, h=(tempX - 208))
        cmds.deleteUI("camShake2D_play", layout=True)


def camShake2D_Fill(camTrans, rsm1, rsm2, rsm3, rsm4, rsm5, rsm6, rsCamUIMainColumn):
    import maya.cmds as cmds

    cmds.setParent(rsCamUIMainColumn)

    cmds.window("camShake2D_", e=1, wh=[235, 305])

    if camTrans == '--Unlink Camera--':
        if cmds.control('rsCameraReplaceColumn', query=True, exists=True):
            cmds.deleteUI('rsCameraReplaceColumn', layout=True)
            return

    camShape = cmds.listRelatives(camTrans, shapes=True)

    cmds.menuItem(rsm1, e=1, checkBox=cmds.getAttr(camShape[0] + ".displayFilmGate"), en=True)
    cmds.menuItem(rsm2, e=1, checkBox=cmds.getAttr(camShape[0] + ".displayResolution"), en=True)
    cmds.menuItem(rsm3, e=1, checkBox=cmds.getAttr(camShape[0] + ".displayFieldChart"), en=True)
    cmds.menuItem(rsm4, e=1, checkBox=cmds.getAttr(camShape[0] + ".displaySafeAction"), en=True)
    cmds.menuItem(rsm5, e=1, checkBox=cmds.getAttr(camShape[0] + ".displaySafeTitle"), en=True)
    cmds.menuItem(rsm6, e=1, checkBox=cmds.getAttr(camShape[0] + ".displaySafeTitle"), en=True)

    cmds.menuItem(rsm1, e=1, c=('import maya.cmds as cmds;cmds.setAttr("' + camShape[
        0] + '.displayFilmGate", cmds.menuItem("' + rsm1 + '", q=1,checkBox=1))'))
    cmds.menuItem(rsm2, e=1, c=('import maya.cmds as cmds;cmds.setAttr("' + camShape[
        0] + '.displayResolution", cmds.menuItem("' + rsm2 + '", q=1,checkBox=1))'))
    cmds.menuItem(rsm3, e=1, c=('import maya.cmds as cmds;cmds.setAttr("' + camShape[
        0] + '.displayFieldChart", cmds.menuItem("' + rsm3 + '", q=1,checkBox=1))'))
    cmds.menuItem(rsm4, e=1, c=('import maya.cmds as cmds;cmds.setAttr("' + camShape[
        0] + '.displaySafeAction", cmds.menuItem("' + rsm4 + '", q=1,checkBox=1))'))
    cmds.menuItem(rsm5, e=1, c=('import maya.cmds as cmds;cmds.setAttr("' + camShape[
        0] + '.displaySafeTitle", cmds.menuItem("' + rsm5 + '", q=1,checkBox=1))'))
    cmds.menuItem(rsm6, e=1, c=('import maya.cmds as cmds; camShake2D.camShake2D_ShowWindow("' + rsm6 + '","' + camShape[
        0] + '","' + rsCamUIMainColumn + '")'))

    if cmds.control('rsCameraReplaceColumn', query=True, exists=True):
        cmds.deleteUI('rsCameraReplaceColumn', layout=True)
    cmds.columnLayout('rsCameraReplaceColumn', columnAttach=('both', 1), adj=True)


    ##
    ##### SHAKE
    ##
    if cmds.objExists(camTrans + "_ShakeControl"):
        cmds.window("camShake2D_", e=1, h=355)
        tempEC = 'import maya.cmds as cmds; x=cmds.window("camShake2D_",q=1,h=1);cmds.window("camShake2D_",e=1,h=(x+395));'
        tempCC = 'import maya.cmds as cmds; x=cmds.window("camShake2D_",q=1,h=1);cmds.window("camShake2D_",e=1,h=(x-395));'
        cmds.separator(style="double", h=20)
        cmds.frameLayout(li=60, fn='boldLabelFont', l='SHAKE', la='center', bv=False, bs='etchedOut', ec=tempEC,
                         cc=tempCC, cl=1, cll=1)
        cmds.columnLayout(cat=['both', 1], adj=1)
        cmds.separator(style="none", h=5)
        cmds.separator(style="none", h=10)
        shakeOpt = cmds.optionMenu(label='Presets')
        cmds.optionMenu(shakeOpt, edit=True,
                        cc='camShake2D.camShake2D_KeyShakePreset("' + shakeOpt + '","' + camTrans + '")')
        cmds.menuItem(label='-Reset-')
        cmds.menuItem(label='Slow Drift')
        cmds.menuItem(label='Exaggerated Drift')
        cmds.menuItem(label='Extreme Drift')
        cmds.menuItem(label='Standard Ride')
        cmds.menuItem(label='Bumpy Ride')
        cmds.menuItem(label='Rough Ride')
        cmds.separator(style="none", h=3)
        # shake mag
        cmds.columnLayout(cat=['both', 1], cal="center", adj=1)
        cmds.separator(style="none", h=3)
        cmds.separator(h=10)
        shk1AttFld = cmds.attrControlGrp(attribute=camTrans + '_ShakeControl.seed', hmb=True)
        shk3AttFld = cmds.attrControlGrp(attribute=camTrans + '_ShakeControl.offset', hmb=True)
        cmds.separator(h=10)
        shk4AttFld = cmds.attrControlGrp(attribute=camTrans + '_ShakeControl.frequencyTransX', hmb=True)
        shk5AttFld = cmds.attrControlGrp(attribute=camTrans + '_ShakeControl.frequencyTransY', hmb=True)
        shk6AttFld = cmds.attrControlGrp(attribute=camTrans + '_ShakeControl.amplitudeTransX', hmb=True)
        shk7AttFld = cmds.attrControlGrp(attribute=camTrans + '_ShakeControl.amplitudeTransY', hmb=True)
       
        cmds.separator(h=10)
        cmds.separator(style="none", h=3)
        shakeKeyButton = cmds.button(l="Key Camera Shake", c=('camShake2D.camShake2D_KeyShake("' + camTrans + '")'))
        cmds.setParent("..")
        cmds.setParent("..")
        cmds.setParent("..")
    ##
    ##### CAMERA
    ##
    tempEC = 'import maya.cmds as cmds; x=cmds.window("camShake2D_",q=1,h=1);cmds.window("camShake2D_",e=1,h=(x+150));'
    tempCC = 'import maya.cmds as cmds; x=cmds.window("camShake2D_",q=1,h=1);cmds.window("camShake2D_",e=1,h=(x-150));'
    cmds.separator(style="double", h=20)
    cmds.frameLayout(li=60, fn='boldLabelFont', l='CAMERA', la='center', bv=False, bs='etchedOut', ec=tempEC, cc=tempCC,
                     cl=1, cll=1, )
    cmds.columnLayout(cat=['both', 1], adj=1)
    cmds.separator(style="none", h=10)
    pOpt = cmds.optionMenu(label='Primes', )
    cmds.menuItem(label='-Choose-')
    cmds.menuItem(label='15mm')
    cmds.menuItem(label='20mm')
    cmds.menuItem(label='28mm')
    cmds.menuItem(label='35mm')
    cmds.menuItem(label='45mm')
    cmds.menuItem(label='50mm')
    cmds.menuItem(label='60mm')
    cmds.menuItem(label='75mm')
    cmds.menuItem(label='80mm')
    cmds.menuItem(label='90mm')
    cmds.menuItem(label='105mm')
    cmds.menuItem(label='120mm')
    cmds.menuItem(label='150mm')
    cmds.menuItem(label='210mm')
    cmds.menuItem(label='300mm')
    cmds.menuItem(label='500mm')
    cmds.menuItem(label='1000mm')

    cmds.separator(style="none", h=3)
    lSldr = cmds.attrFieldSliderGrp(l="Lens: ", smn=15, smx=120.0, fmn=2.5, fmx=3500, adj=3, cw4=[34, 42, 40, 10],
                                    pre=1, at=(camShape[0] + ".focalLength"), hmb=True, cc='print "Boob"')

    cmds.separator(style="none", h=3)
    cmds.rowLayout(w=220, numberOfColumns=5, columnWidth5=(15, 23, 110, 23, 15), adj=3,
                   columnAttach5=['both', 'both', 'both', 'both', 'both'],
                   cl5=["center", "center", "center", "center", "center"])

    cmds.setParent("..")

    ncpAttFld = cmds.attrControlGrp(attribute=camShape[0] + '.nearClipPlane', hmb=True)
    fcpAttFld = cmds.attrControlGrp(attribute=camShape[0] + '.farClipPlane', hmb=True)
    oAttFld = cmds.attrControlGrp(attribute=camShape[0] + '.overscan', hmb=True)
    cmds.setParent("..")
    cmds.setParent("..")


    cmds.optionMenu(pOpt, e=1, cc='camShake2D.camShake2D_Opt("' + pOpt + '","' + camShape[0] + '")')



def camShake2D_Opt(pOpt, camShape):
    import maya.cmds as cmds

    if cmds.optionMenu(pOpt, q=1, v=1) != '-Choose-':
        temp = cmds.optionMenu(pOpt, q=1, v=1)
        temp = int(temp.replace("mm", ""))
        cmds.setAttr(camShape + '.focalLength', temp)
        cmds.optionMenu(pOpt, e=1, v='-Choose-')



def camShake2D_ShakeStart(cOpt, camTrans, rsm1, rsm2, rsm3, rsm4, rsm5, rsm6, rsCamUIMainColumn):

    import maya.cmds as cmds

    if cmds.optionMenu(cOpt, q=True, v=True) == "--Unlink Camera--":
        return

    if cmds.objExists(camTrans + "_ShakeControl") == True:
        camShake2D_ShakeRemove(camTrans)
    else:
        camShake2D_ShakeAdd(camTrans)
        camShake2D_ShakeReAdd(camTrans)
    camShake2D_Fill(camTrans, rsm1, rsm2, rsm3, rsm4, rsm5, rsm6, rsCamUIMainColumn)


def camShake2D_ShakeRemove(currentCamera):
    import maya.cmds as cmds

    if cmds.objExists(currentCamera + "_ShakeControl") == False:
        cmds.confirmDialog(title='No Shaker', message='There is no camera shake!!\n\n', button=['Yes', 'No'],
                           defaultButton='Abort')
        return

    cmds.select(currentCamera + "_ShakeControl", d=True)
    cmds.currentTime((cmds.playbackOptions(q=True, minTime=True)))


    tempChanList = ["frequencyRot", "amplitudeTransY", "amplitudeTransX", "frequencyTransY", "frequencyTransX",
                    "seedTrans", "amplitudeRot", "seedRot", "magnitude"]
    tcl = ""
    for tcl in tempChanList:
        try:
            cmds.disconnectAttr((currentCamera + "_ShakeControl_" + tcl + ".output"),
                                (currentCamera + "_ShakeControl." + tcl))
        except:
            print " - There is no animation on " + currentCamera + "_ShakeControl_" + tcl

    tempChanList = ["tx", "ty", "tz", "sx", "sy", "sz"]
    tcl = ""
    for tcl in tempChanList:
        tempConncts = cmds.listConnections((currentCamera + "_ShakeControl." + tcl), d=False, s=True)
        tempConnct = ""
        if tempConncts != None:
            for tempConnct in tempConncts:
                if cmds.nodeType(tempConnct) == "animCurveTL":
                    cmds.cutKey(currentCamera + "_ShakeControl", t=(":",), f=(":",), at=tcl)
                    cmds.pasteKey(currentCamera, connect=True, at=tcl)

    tempChanList = ["rx", "ry", "rz"]
    tcl = ""
    for tcl in tempChanList:
        tempConncts = cmds.listConnections((currentCamera + "_ShakeControl." + tcl), d=False, s=True)
        tempConnct = ""
        if tempConncts != None:
            for tempConnct in tempConncts:
                if cmds.nodeType(tempConnct) == "animCurveTA":
                    cmds.cutKey(currentCamera + "_ShakeControl", t=(":",), f=(":",), at=tcl)
                    cmds.pasteKey(currentCamera, connect=True, at=tcl)

    tempChanList = ["rotateX", "rotateY", "rotateZ", "translateX", "translateY", "translateZ"]
    tcl = ""
    for tcl in tempChanList:
        try:
            cmds.delete(currentCamera + "_ShakeControl_" + tcl)
        except:
            print (currentCamera + "_ShakeControl_" + tcl + " doesn't exist")


    cmds.delete(currentCamera + "_ShakeControl")

    if(cmds.objExists(currentCamera + "_ShakeLayer")):
        cmds.delete(currentCamera + "_ShakeLayer")


def camShake2D_KeyShakePreset(shakeOpt, currentCamera):
    import maya.cmds as cmds

    tsp = cmds.optionMenu(shakeOpt, q=1, v=1)
    tempChannels = ["seed", "offset", "frequencyTransX", "frequencyTransY", "amplitudeTransX", "amplitudeTransY"]
                    

    tempValues = []

    if tsp == "Slow Drift":
        tempValues = [0.0, 0.0, 0.5, 0.7, 0.7, 0.7]
    if tsp == "Exaggerated Drift":
        tempValues = [0.0, 0.0, 1, 1.2, 1, 1]
    if tsp == "Extreme Drift":
        tempValues = [0.0, 0.0, 1.3, 1.4, 1.3, 1.3]
    if tsp == "Standard Ride":
        tempValues = [0.0, 0.0, 2.0, 3.0, 0.3, 0.4]
    if tsp == "Bumpy Ride":
        tempValues = [0.0, 0.0, 2.8, 3.7, 0.5, 0.5]
    if tsp == "Rough Ride":
        tempValues = [0.0, 0.0, 3, 4, 0.8, 0.9]
    if tsp == "-Reset-":
        tempValues = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]



    tempCounter = 0
    for tempChan in tempChannels:
        cmds.setAttr(currentCamera + "_ShakeControl." + tempChan, tempValues[tempCounter])
        tempCounter = tempCounter + 1


def camShake2D_KeyShake(currentCamera):
    import maya.cmds as cmds

    tempChanList = ["frequencyTransY", "amplitudeTransY", "frequencyTransX", "amplitudeTransX", "seed", "offset"]

    tcl = ""
    for tcl in tempChanList:
        cmds.setKeyframe(currentCamera + "_ShakeControl." + tcl)


def camShake2D_ShakeReAdd(currentCamera):
    import maya.cmds as cmds
    import maya.mel as mel

    tempChanList = ["frequencyRot", "frequencyTransY", "amplitudeTransY", "frequencyTransX", "amplitudeTransX",
                    "seedTrans", "amplitudeRot", "seedRot", "magnitude"]

    tcl = ""
    previousShaker = 0
    for tcl in tempChanList:
        if cmds.objExists(currentCamera + "_ShakeControl_" + tcl):
            previousShaker = 1
    if previousShaker == 1:
        tempDialog = cmds.confirmDialog(title='Confirm', message='Previous Shaker settings found!',
                                        button=['Use Previous Settings', 'ignore'], defaultButton='Yes',
                                        cancelButton='Re-Initialize', dismissString='Re-Initialize')
        if tempDialog == 'ignore':
            for tcl in tempChanList:
                if cmds.objExists(currentCamera + "_ShakeControl_" + tcl):
                    cmds.delete(currentCamera + "_ShakeControl_" + tcl)
        else:
            for tcl in tempChanList:
                try:
                    cmds.connectAttr((currentCamera + "_ShakeControl_" + tcl + ".output"),
                                     (currentCamera + "_ShakeControl." + tcl))
                except:
                    print " - There is no animation on " + tcl


def camShake2D_ShakeAdd(currentCamera):
    import maya.cmds as cmds
    import maya.mel as mel

    if cmds.objExists(currentCamera + "_ShakeControl") == True:
        cmds.confirmDialog(title='Naming Conflict',
                           message='ABORTING ADDING SHAKE\n\nThere is already an object nameb ' + currentCamera + '_ShakeControl!!\n\n',
                           button=['Yes', 'No'], defaultButton='Abort')
        return


    camShakerNodes = cmds.spaceLocator(p=(0, 0, 0), n=(currentCamera + "_ShakeControl"))
    currentCameraCon = camShakerNode = camShakerNodes[0]

    ####
    cmds.parentConstraint(currentCamera, currentCameraCon)
   

    # Add Attribs

    cmds.addAttr(currentCameraCon, ln="CameraShake", at="enum", en="========:")
    cmds.setAttr((currentCameraCon + ".CameraShake"), e=True, keyable=True, lock=True)
    cmds.addAttr(currentCameraCon, ln="seed", at="double", min=-99999, max=99999, dv=0)
    cmds.setAttr((currentCameraCon + ".seed"), e=True, keyable=True)
    cmds.addAttr(currentCameraCon, ln="offset", at="double", min=-99999, max=99999, dv=0)
    cmds.setAttr((currentCameraCon + ".offset"), e=True, keyable=True)
    cmds.addAttr(currentCameraCon, ln="frequencyTransX", at="double", min=-500, max=500, dv=0)
    cmds.setAttr((currentCameraCon + ".frequencyTransX"), e=True, keyable=True)
    cmds.addAttr(currentCameraCon, ln="frequencyTransY", at="double", min=-500, max=500, dv=0)
    cmds.setAttr((currentCameraCon + ".frequencyTransY"), e=True, keyable=True)
    cmds.addAttr(currentCameraCon, ln="amplitudeTransX", at="double", min=-99999, max=99999, dv=0)
    cmds.setAttr((currentCameraCon + ".amplitudeTransX"), e=True, keyable=True)
    cmds.addAttr(currentCameraCon, ln="amplitudeTransY", at="double", min=-99999, max=99999, dv=0)
    cmds.setAttr((currentCameraCon + ".amplitudeTransY"), e=True, keyable=True)


    currentCameraShape = cmds.listRelatives(currentCamera, s=True, type='camera')
    cmds.setAttr(currentCameraShape[0] + '.hpn', k=True)
    cmds.setKeyframe(currentCameraShape[0] + '.hpn', t=cmds.playbackOptions(q=True, min=True))
    cmds.setAttr(currentCameraShape[0] + '.vpn', k=True)
    cmds.setKeyframe(currentCameraShape[0] + '.vpn', t=cmds.playbackOptions(q=True, min=True))
    cmds.setAttr(currentCameraShape[0] + '.zoom', k=True)
    cmds.setKeyframe(currentCameraShape[0] + '.zoom', t=cmds.playbackOptions(q=True, min=True))
    cmds.setAttr(currentCameraShape[0] + '.panZoomEnabled', True)

    animLayerName = currentCamera+"_ShakeLayer"


    if(cmds.objExists(animLayerName)):
        cmds.delete(animLayerName)

    cmds.animLayer(animLayerName, at=[currentCameraShape[0] + '.hpn', currentCameraShape[0] + '.vpn'])

    animBlendNodeH = cmds.listConnections(currentCameraShape[0] + '.hpn', s=True, d=False)
    animBlendNodeV = cmds.listConnections(currentCameraShape[0] + '.vpn', s=True, d=False)

    expressionString = 'expression -s (\"float $ampX = ' + currentCameraCon + '.amplitudeTransX;\\n'
    expressionString += 'float $ampY = ' + currentCameraCon + '.amplitudeTransY;\\n'
    expressionString += 'float $seed = ' + currentCameraCon + '.seed;\\n'
    expressionString += 'float $freqTX = ' + currentCameraCon + '.frequencyTransX;\\n'
    expressionString += 'float $freqTY = ' + currentCameraCon + '.frequencyTransY;\\n'
    expressionString += 'float $ampOffset = ' + currentCameraCon + '.offset/100;\\n'
    expressionString += '// compute input value for noise function\\n'
    expressionString += 'float $noiseTransX = (frame * ($freqTX * .1) + ($seed +3));\\n'
    expressionString += 'float $noiseTransY = (frame * ($freqTY * .1) + ($seed +4));\\n'
    expressionString += '// noise amplitudeX\\n'
    expressionString += '$ampX = $ampX + (noise($noiseTransX) * $ampOffset);\\n'
    expressionString += '$ampY = $ampY + (noise($noiseTransY) * $ampOffset);\\n'
    expressionString += '// Translations\\n// transX is sin wave * amplitude\\n'
    expressionString += 'float $sin_input_frequency = noise($noiseTransX) * .05 * 3.14;\\n'
    expressionString += 'float $cameraShakeTransX = sin($sin_input_frequency) * $ampX;\\n'
    expressionString += '$sin_input_frequency = noise($noiseTransY) * .05 * 3.14;\\n'
    expressionString += 'float $cameraShakeTransY = sin($sin_input_frequency) * $ampY;\\n'
    expressionString += animBlendNodeH[0] + '.inputB = $cameraShakeTransX;\\n'
    expressionString += animBlendNodeV[0] + '.inputB = $cameraShakeTransY;\\n\") -o '
    expressionString += currentCameraCon + ' -ae 1 -uc all;'
    mel.eval(expressionString)

    cmds.select(currentCameraCon, r=True)