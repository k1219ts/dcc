##################################################
# Description: Bake selected camera.
# Version: 1.0
# Written by Daehwan Jang
# final Corected by kwantae Kim
# Last updated: 2019. 3. 26
##################################################

# Change Log
# 0.8
# FilmOffset attributes are added.
# 0.9
# bake Focal length key. (dongho.Cha)
# 1.0
# bug fix, imageplane add & Add camera key for motion blur. (kwantae.Kim)

import maya.cmds as cmds
import maya.mel as mm
import addCameraKey as camKey

def copyImageplene(selectcameraShape, newcameraShape):
    a = cmds.listRelatives(selectcameraShape, ad=True)
    ipList = cmds.ls(a[0], type="imagePlane")

    ipAttrs = ['displayMode', 'type', 'textureFilter', 'imageName',
               'offsetX', 'offsetY', 'useFrameExtension', 'frameOffset',
               'frameCache',
               'fit', 'displayOnlyIfCurrent', 'depth', 'frameExtension',
               'coverageX', 'coverageY', 'coverageOriginX', 'coverageOriginY',
               'imageCenterX', 'imageCenterY', 'imageCenterZ',
               'width', 'height', 'maintainRatio']

    renewIpList = []
    for i in ipList:

        cmds.select(i)
        mm.eval('autoUpdateAttrEd;')
        attrResult = dict()
        for a in ipAttrs:
            gv = cmds.getAttr('%s.%s' % (i, a))
            gt = cmds.getAttr('%s.%s' % (i, a), type=True)
            attrResult[a] = {'value': gv, 'type': gt}

        newtr, newshape = cmds.imagePlane(camera=newcameraShape)

        for a in attrResult:
            obj = attrResult[a]
            if obj['type'] == 'string':
                cmds.setAttr('%s.%s' % (newshape, a), obj['value'], type='string')
            else:
                cmds.setAttr('%s.%s' % (newshape, a), obj['value'])
        if attrResult['useFrameExtension']['value']:
            cmds.expression(n='frame_ext_expression',
                            s='%s.frameExtension=frame;' % newshape)

        if attrResult['displayOnlyIfCurrent']['value']:
            try:
                mm.eval(
                    'source AEimagePlaneTemplate.mel; optionMenu -edit -enable true AELookThroughCameraMenu; AEchangeLookThroughCamera %s;' % newshape)
            except:
                pass

        try:
            cmds.copyKey(i + ".depth")
            cmds.pasteKey(newshape + ".depth")  # edit by ani
        except:
            pass

        renewIpList.append(newshape)

    cmds.select(cl=True)
    return renewIpList


def bakeCam():
    selectedCamera = cmds.ls(sl=True, dag=True)
    # a transform node of the selected Camera: selectedCamera[0]
    # a shape node of the selected camera: selectedCamera[1]

    newCamera = cmds.camera(name="bakedCam")
    # a transform node of the created camera: newCamera[0]
    # a shape node of the created camera: newCamera[1]

    # Connect basic attributes from the selected camera to new camera. Other attributes will be added.
    # horizontalFilmAperture, verticalFilmAperture, lensSqueezeRatio
    cmds.setAttr(newCamera[0] + ".rotateOrder", cmds.getAttr(selectedCamera[0] + ".rotateOrder"), lock=True)

    cmds.setAttr(newCamera[1] + ".focalLength", cmds.getAttr(selectedCamera[1] + ".focalLength"))
    cmds.setAttr(newCamera[1] + ".nearClipPlane", cmds.getAttr(selectedCamera[1] + ".nearClipPlane"))
    cmds.setAttr(newCamera[1] + ".farClipPlane", cmds.getAttr(selectedCamera[1] + ".farClipPlane"))

    cmds.setAttr(newCamera[1] + ".horizontalFilmAperture", cmds.getAttr(selectedCamera[1] + ".horizontalFilmAperture"),
                 lock=True)
    cmds.setAttr(newCamera[1] + ".verticalFilmAperture", cmds.getAttr(selectedCamera[1] + ".verticalFilmAperture"),
                 lock=True)
    cmds.setAttr(newCamera[1] + ".filmFit", cmds.getAttr(selectedCamera[1] + ".filmFit"), lock=True)
    cmds.setAttr(newCamera[1] + ".lensSqueezeRatio", cmds.getAttr(selectedCamera[1] + ".lensSqueezeRatio"), lock=True)
    cmds.setAttr(newCamera[1] + ".horizontalFilmOffset", cmds.getAttr(selectedCamera[1] + ".horizontalFilmOffset"),
                 lock=True)
    cmds.setAttr(newCamera[1] + ".verticalFilmOffset", cmds.getAttr(selectedCamera[1] + ".verticalFilmOffset"),
                 lock=True)

    cmds.setAttr(newCamera[1] + ".preScale", cmds.getAttr(selectedCamera[1] + ".preScale"), lock=True)
    cmds.setAttr(newCamera[1] + ".postScale", cmds.getAttr(selectedCamera[1] + ".postScale"), lock=True)

    cmds.setAttr(newCamera[1] + ".shutterAngle", cmds.getAttr(selectedCamera[1] + ".shutterAngle"), lock=True)
    cmds.setAttr(newCamera[1] + ".fStop", cmds.getAttr(selectedCamera[1] + ".fStop"), lock=True)

    # Constrain transform data of new camera to the selected camera by parent contraint.
    constInfo = cmds.parentConstraint(selectedCamera[0], newCamera[0], weight=1, maintainOffset=0, skipTranslate="none",
                                      skipRotate="none")

    # copy focalLength key.
    connList = cmds.listConnections(selectedCamera[1])
    if connList != None:
        count = 0;
        for i in connList:
            print i
            if (cmds.nodeType(i) == "animCurveTU"):
                if "focalLength" in i:
                    cmds.connectAttr(i + ".output", newCamera[1] + ".focalLength")

    # Bake transform data of new camera.
    cmds.bakeResults(newCamera[0], t=(cmds.playbackOptions(q=True, ast=True), cmds.playbackOptions(q=True, aet=True)),
                     pok=True, at=["tx", "ty", "tz", "rx", "ry", "rz"])
    cmds.bakeResults(newCamera[1], t=(cmds.playbackOptions(q=True, ast=True), cmds.playbackOptions(q=True, aet=True)),
                     pok=True, at=["fl"])

    # Delete parentConstraint
    cmds.delete(constInfo)

    # imageplane copy
    copyImageplene(selectedCamera[1], newCamera[1])

    # Add camera key for motion blur.
    cmds.select(newCamera[0])
    camKey.offsetKey()

    # lock Attributes
    cmds.setAttr(newCamera[0] + ".translate", lock=True)
    cmds.setAttr(newCamera[0] + ".rotate", lock=True)
    cmds.setAttr(newCamera[1] + ".focalLength", lock=True)