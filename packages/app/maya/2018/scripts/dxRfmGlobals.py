'''
output Attribute
# sampling 
getAttr "renderManRISGlobals.rman__riopt__Hider_minsamples";
getAttr "renderManRISGlobals.rman__riopt__Hider_maxsamples";
getAttr "renderManRISGlobals.rman__riopt___PixelVariance";
getAttr "renderManRISGlobals.rman__riopt__Hider_darkfalloff";
getAttr "renderManRISGlobals.rman__riopt__Hider_incremental";

# integrator
getAttr "rman__riopt__Integrator_name"

"PxrDebugShadingContext"
getAttr "PxrDebugShadingContext.viewchannel"

"PxrDefualt"

"PxrDirectLighting"
getAttr "PxrDirectLighting.numLightSamples"
getAttr "PxrDirectLighting.numBxdfSamples"

"PxrOcclusion"
getAttr "PxrOcclusion.numSamples"
getAttr "PxrOcclusion.numBxdfSamples"
getAttr "PxrOcclusion.distribution"
    "Uniform"
    getAttr "PxrOcclusion.falloff"
    getAttr "PxrOcclusion.maxDistance"
    getAttr "PxrOcclusion.useAlbedo"

    "Cosine"
    getAttr "PxrOcclusion.cosineSpread"
    getAttr "PxrOcclusion.falloff"
    getAttr "PxrOcclusion.maxDistance"
    getAttr "PxrOcclusion.useAlbedo"

    "Reflection"
    getAttr "PxrOcclusion.falloff"
    getAttr "PxrOcclusion.maxDistance"
    getAttr "PxrOcclusion.useAlbedo"    

"PxrPathTracer"
getAttr "PxrPathTracer.maxPathLength"
getAttr "PxrPathTracer.maxContinuationLength"
getAttr "PxrPathTracer.sampleMode"
    "manual"
    getAttr "PxrPathTracer.numLightSamples"
    getAttr "PxrPathTracer.numBxdfSamples"
    getAttr "PxrPathTracer.numDiffuseSamples"
    getAttr "PxrPathTracer.numSpecularSamples"
    getAttr "PxrPathTracer.numSubsurfaceSamples"
    getAttr "PxrPathTracer.numRefractionSamples"
    getAttr "PxrPathTracer.allowCaustics"
    getAttr "PxrPathTracer.accumOpacity"

    "bxdf"
    getAttr "PxrPathTracer.numLightSamples"
    getAttr "PxrPathTracer.numBxdfSamples"
    getAttr "PxrPathTracer.numIndirectSamples"
    getAttr "PxrPathTracer.allowCaustics"
    getAttr "PxrPathTracer.accumOpacity"

"PxrVCM"
getAttr "PxrVCM.connectPaths"
getAttr "PxrVCM.mergePaths"
getAttr "PxrVCM.numLightSamples"
getAttr "PxrVCM.numBxdfSamples"
getAttr "PxrVCM.maxPathLength"
getAttr "PxrVCM.rouletteDepth"
getAttr "PxrVCM.rouletteThreshold"
getAttr "PxrVCM.clampDepth"
getAttr "PxrVCM.clampLuminance"
getAttr "PxrVCM.mergeRadius"
getAttr "PxrVCM.timeRadius"
getAttr "PxrVCM.photonGuiding"
getAttr "PxrVCM.photonGuidingBBoxMinX"
getAttr "PxrVCM.photonGuidingBBoxMinY"
getAttr "PxrVCM.photonGuidingBBoxMinZ"

getAttr "PxrVCM.photonGuidingBBoxMaxX"
getAttr "PxrVCM.photonGuidingBBoxMaxY"
getAttr "PxrVCM.photonGuidingBBoxMaxZ"

getAttr "PxrVCM.specularCurvatureFilter"

"PxrValidateBxdf"
getAttr "PxrValidateBxdf.numSamples"

"PxrVisualizer"
getAttr "PxrVisualizer.style"
getAttr "PxrVisualizer.wireframe"
getAttr "PxrVisualizer.normalCheck"
getAttr "PxrVisualizer.matCap"

# Filter
getAttr "rmanFinalOutputGlobals0.rman__riopt__Display_filter"
getAttr "rmanFinalOutputGlobals0.rman__riopt__Display_filterwidth0"
getAttr "rmanFinalOutputGlobals0.rman__riopt__Display_filterwidth1"
'''

import maya.cmds as cmds
import maya.app.renderSetup.model.renderSetup as renderSetup
import time
import getpass
import os
import json
import re

def getSamplingData():
    print "getSamplingData call"

    minSamples = cmds.getAttr("renderManRISGlobals.rman__riopt__Hider_minsamples")
    maxSamples = cmds.getAttr("renderManRISGlobals.rman__riopt__Hider_maxsamples")
    pixelVariance = cmds.getAttr("renderManRISGlobals.rman__riopt___PixelVariance")
    darkFallOff = cmds.getAttr("renderManRISGlobals.rman__riopt__Hider_darkfalloff")
    incremental = cmds.getAttr("renderManRISGlobals.rman__riopt__Hider_incremental")

    samplingDict = {"renderManRISGlobals.rman__riopt__Hider_minsamples" : minSamples,
                    "renderManRISGlobals.rman__riopt__Hider_maxsamples" : maxSamples,
                    "renderManRISGlobals.rman__riopt___PixelVariance" : pixelVariance,
                    "renderManRISGlobals.rman__riopt__Hider_darkfalloff" : darkFallOff,
                    "renderManRISGlobals.rman__riopt__Hider_incremental" : incremental}

    return "Sampling", samplingDict

def getIntegratorData():
    print "getIntegratorData call"

    integratorName = cmds.getAttr("renderManRISGlobals.rman__riopt__Integrator_name")
    integratorInfoDict = {}

    if "PxrDebugShadingContext" == integratorName:
        viewChannel = cmds.getAttr("PxrDebugShadingContext.viewchannel")
        integratorInfoDict[integratorName] = {"PxrDebugShadingContext.viewchannel" : viewChannel}

    elif "PxrDefault" == integratorName:
        pass

    elif "PxrDirectLighting" == integratorName:
        numLightSamples = cmds.getAttr("PxrDirectLighting.numLightSamples")
        numBxdfSamples = cmds.getAttr("PxrDirectLighting.numBxdfSamples")

        integratorInfoDict[integratorName] = {"PxrDirectLighting.numLightSamples" : numLightSamples,
                                          "PxrDirectLighting.numBxdfSamples": numBxdfSamples
                                          }

    elif "PxrOcclusion" == integratorName:
        numSamples = cmds.getAttr("PxrOcclusion.numSamples")
        numBxdfSamples = cmds.getAttr("PxrOcclusion.numBxdfSamples")
        distribution = cmds.getAttr("PxrOcclusion.distribution")

        falloff = cmds.getAttr("PxrOcclusion.falloff")
        maxDistance = cmds.getAttr("PxrOcclusion.maxDistance")
        useAlbedo = cmds.getAttr("PxrOcclusion.useAlbedo")

        integratorInfoDict[integratorName] = {"PxrOcclusion.numSamples" : numSamples,
                                          "PxrOcclusion.numBxdfSamples" : numBxdfSamples,
                                          "PxrOcclusion.distribution" : {distribution : {"PxrOcclusion.falloff" : falloff,
                                                                                        "PxrOcclusion.maxDistance" : maxDistance,
                                                                                        "PxrOcclusion.useAlbedo" : useAlbedo
                                                                                        }
                                                                         }
                                          }

        if "Cosine" == distribution:
            cosineSpread = cmds.getAttr("PxrOcclusion.cosineSpread")
            integratorInfoDict[integratorName]["PxrOcclusion.distribution"][distribution]["PxrOcclusion.cosineSpread"] = cosineSpread

    elif "PxrPathTracer" == integratorName:
        maxPathLength = cmds.getAttr("PxrPathTracer.maxPathLength")
        maxContinuationLength = cmds.getAttr("PxrPathTracer.maxContinuationLength")
        sampleMode = cmds.getAttr("PxrPathTracer.sampleMode")

        numLightSamples = cmds.getAttr("PxrPathTracer.numLightSamples")
        numBxdfSamples = cmds.getAttr("PxrPathTracer.numBxdfSamples")
        allowCaustics = cmds.getAttr("PxrPathTracer.allowCaustics")
        accumOpacity = cmds.getAttr("PxrPathTracer.accumOpacity")

        integratorInfoDict[integratorName] = {"PxrPathTracer.maxPathLength" : maxPathLength,
                                          "PxrPathTracer.maxContinuationLength" : maxContinuationLength,
                                          "PxrPathTracer.sampleMode" : {sampleMode : { "PxrPathTracer.numLightSamples" : numLightSamples,
                                                                                       "PxrPathTracer.numBxdfSamples" : numBxdfSamples,
                                                                                       "PxrPathTracer.allowCaustics" : allowCaustics,
                                                                                       "PxrPathTracer.accumOpacity" : accumOpacity
                                                                                    }
                                                                        }
                                          }

        if "manual" == sampleMode:
            numDiffuseSamples = cmds.getAttr("PxrPathTracer.numDiffuseSamples")
            numSpecularSamples = cmds.getAttr("PxrPathTracer.numSpecularSamples")
            numSubsurfaceSamples = cmds.getAttr("PxrPathTracer.numSubsurfaceSamples")
            numRefractionSamples = cmds.getAttr("PxrPathTracer.numRefractionSamples")

            integratorInfoDict[integratorName]["PxrPathTracer.sampleMode"][sampleMode]["PxrPathTracer.numDiffuseSamples"] = numDiffuseSamples
            integratorInfoDict[integratorName]["PxrPathTracer.sampleMode"][sampleMode]["PxrPathTracer.numSpecularSamples"] = numSpecularSamples
            integratorInfoDict[integratorName]["PxrPathTracer.sampleMode"][sampleMode]["PxrPathTracer.numSubsurfaceSamples"] = numSubsurfaceSamples
            integratorInfoDict[integratorName]["PxrPathTracer.sampleMode"][sampleMode]["PxrPathTracer.numRefractionSamples"] = numRefractionSamples

        elif "bxdf" == sampleMode:
            numIndirectSamples = cmds.getAttr("PxrPathTracer.numIndirectSamples")

            integratorInfoDict[integratorName]["PxrPathTracer.sampleMode"][sampleMode]["PxrPathTracer.numIndirectSamples"] = numIndirectSamples

    elif "PxrVCM" == integratorName:
        connectPaths = cmds.getAttr("PxrVCM.connectPaths")
        mergePaths = cmds.getAttr("PxrVCM.mergePaths")
        numLightSamples = cmds.getAttr("PxrVCM.numLightSamples")
        numBxdfSamples = cmds.getAttr("PxrVCM.numBxdfSamples")
        maxPathLength = cmds.getAttr("PxrVCM.maxPathLength")
        rouletteDepth = cmds.getAttr("PxrVCM.rouletteDepth")
        rouletteThreshold = cmds.getAttr("PxrVCM.rouletteThreshold")
        clampDepth = cmds.getAttr("PxrVCM.clampDepth")
        clampLuminance = cmds.getAttr("PxrVCM.clampLuminance")
        mergeRadius = cmds.getAttr("PxrVCM.mergeRadius")
        timeRadius = cmds.getAttr("PxrVCM.timeRadius")
        photonGuiding = cmds.getAttr("PxrVCM.photonGuiding")
        photonGuidingBBoxMinX = cmds.getAttr("PxrVCM.photonGuidingBBoxMinX")
        photonGuidingBBoxMinY = cmds.getAttr("PxrVCM.photonGuidingBBoxMinY")
        photonGuidingBBoxMinZ = cmds.getAttr("PxrVCM.photonGuidingBBoxMinZ")
        photonGuidingBBoxMaxX = cmds.getAttr("PxrVCM.photonGuidingBBoxMaxX")
        photonGuidingBBoxMaxY = cmds.getAttr("PxrVCM.photonGuidingBBoxMaxY")
        photonGuidingBBoxMaxZ = cmds.getAttr("PxrVCM.photonGuidingBBoxMaxZ")
        specularCurvatureFilter = cmds.getAttr("PxrVCM.specularCurvatureFilter")

        integratorInfoDict[integratorName] = {"PxrVCM.connectPaths": connectPaths,
                                          "PxrVCM.mergePaths": mergePaths,
                                          "PxrVCM.numLightSamples": numLightSamples,
                                          "PxrVCM.numBxdfSamples": numBxdfSamples,
                                          "PxrVCM.maxPathLength": maxPathLength,
                                          "PxrVCM.rouletteDepth": rouletteDepth,
                                          "PxrVCM.rouletteThreshold": rouletteThreshold,
                                          "PxrVCM.clampDepth": clampDepth,
                                          "PxrVCM.clampLuminance": clampLuminance,
                                          "PxrVCM.mergeRadius": mergeRadius,
                                          "PxrVCM.timeRadius": timeRadius,
                                          "PxrVCM.photonGuiding": photonGuiding,
                                          "PxrVCM.photonGuidingBBoxMinX": photonGuidingBBoxMinX,
                                          "PxrVCM.photonGuidingBBoxMinY": photonGuidingBBoxMinY,
                                          "PxrVCM.photonGuidingBBoxMinZ": photonGuidingBBoxMinZ,
                                          "PxrVCM.photonGuidingBBoxMaxX": photonGuidingBBoxMaxX,
                                          "PxrVCM.photonGuidingBBoxMaxY": photonGuidingBBoxMaxY,
                                          "PxrVCM.photonGuidingBBoxMaxZ": photonGuidingBBoxMaxZ,
                                          "PxrVCM.specularCurvatureFilter": specularCurvatureFilter}

    elif "PxrValidateBxdf" == integratorName:
        numSamples = cmds.getAttr("PxrValidateBxdf.numSamples")

        integratorInfoDict[integratorName] = {"PxrValidateBxdf.numSamples": numSamples}

    elif "PxrVisualizer" == integratorName:
        style = cmds.getAttr("PxrVisualizer.style")
        wireframe = cmds.getAttr("PxrVisualizer.wireframe")
        normalCheck = cmds.getAttr("PxrVisualizer.normalCheck")
        matCap = cmds.getAttr("PxrVisualizer.matCap")

        integratorInfoDict[integratorName] = {"PxrVisualizer.style": style,
                                          "PxrVisualizer.wireframe": wireframe,
                                          "PxrVisualizer.normalCheck": normalCheck,
                                          "PxrVisualizer.matCap": matCap}

    integratorDict = {"renderManRISGlobals.rman__riopt__Integrator_name": integratorInfoDict }

    return "Integrator", integratorDict

def getFilterData():
    print "getFilterData call"

    filter = cmds.getAttr("rmanFinalOutputGlobals0.rman__riopt__Display_filter")
    filterwidth0 = cmds.getAttr("rmanFinalOutputGlobals0.rman__riopt__Display_filterwidth0")
    filterwidth1 = cmds.getAttr("rmanFinalOutputGlobals0.rman__riopt__Display_filterwidth1")

    filterDict = {"rmanFinalOutputGlobals0.rman__riopt__Display_filter": filter,
                    "rmanFinalOutputGlobals0.rman__riopt__Display_filterwidth0": filterwidth0,
                    "rmanFinalOutputGlobals0.rman__riopt__Display_filterwidth1": filterwidth1}

    return "Filter", filterDict

def getStupidAOVData():
    stupidAOVNode = cmds.ls(type = 'stupidAOV')

    if len(stupidAOVNode) == 0:
        cmds.confirmDialog(title="Warning!",
                             message="not find stupidAOV Node",
                             messageAlign='center',
                             icon="warning",
                             button=["OK"],
                             backgroundColor=[.5, .5, .5])
        return None

    attrNameList = cmds.listAttr(stupidAOVNode[0])

    customAttrIndex = attrNameList.index('channelSize')
    attrDict = {}
    for attr in attrNameList[customAttrIndex:]:
        attrDict[attr] = cmds.getAttr('%s.%s' % (stupidAOVNode[0], attr))

    return {"StupidAOV" : attrDict}

def createHeader():
    header = {}
    header['created'] = time.asctime()
    header['author']  = getpass.getuser()
    header['context'] = cmds.file( q=True, sn=True )
    return header

def exportRmanGlobals(filePath):
    try:
        # renderLayer set MasterLayer
        renderSetup.instance().getDefaultRenderLayer().makeVisible()

        # get Sampling Attr Info
        sampleKey, sampleDict = getSamplingData()

        # get Integrator Attr Info
        integratorKey, integratorDict = getIntegratorData()

        # get Filter Attr Info
        filterKey, filterDict = getFilterData()

        # make header label
        headerDict = createHeader()

        # make data of json
        rmanGlobalDict = {"rmanGlobals" : {sampleKey : sampleDict,
                                        integratorKey : integratorDict,
                                        filterKey : filterDict},
                         "_Header": headerDict}

        if filePath.startswith('/netapp/dexter/show'):
            filePath.replace('/netapp/dexter/show', '/show')

        dirPath = os.path.dirname(filePath)

        if not os.path.exists(dirPath):
            # print "instal : ", "install -d -m 755 {0}".format(dirPath)
            os.system("install -d -m 755 {0}".format(dirPath))

        with open(filePath, 'w') as f:
            json.dump(rmanGlobalDict, f, indent=4)
            f.close()

        print "Debug : rmanGlobals Export success"

        return True

    except Exception as e:
        print "Error : %s" % e.message
        return False

def recursiveSetAttr(attrDict):
    for attr in attrDict.keys():
        if type(attrDict[attr]) == dict:
            value = attrDict[attr].keys()[0]
            # print attr, value
            # print "1"
            setAttr(attr, value)
            recursiveSetAttr(attrDict[attr][value])
        else:
            # print attr, attrDict[attr]
            # print "2"
            setAttr(attr, attrDict[attr])


def setAttr(attr, value):
    print "setAttr :", attr, value
    if re.compile(r"(-|)\d.\d").match(str(value)):
        # print "type : float"
        cmds.setAttr(attr, value)
    elif re.compile(r"(-|)\d").match(str(value)):
        # print "type : long"
        cmds.setAttr(attr, value)
    elif re.compile(r"(False|True)").match(str(value)):
        cmds.setAttr(attr, value)
    else:
        # print "type : string"
        cmds.setAttr(attr, value, type = "string")

def importRmanGlobals(filePath):
    try:
        jsonData = open(filePath, 'r').read()
        rmanGlobalData = json.loads(jsonData)

        # set masterLayer because export when masterLayer status
        renderSetup.instance().getDefaultRenderLayer().makeVisible()

        # import Sampling
        recursiveSetAttr(rmanGlobalData["rmanGlobals"]["Sampling"])

        # import Integrator
        recursiveSetAttr(rmanGlobalData["rmanGlobals"]["Integrator"])

        # import Filter
        recursiveSetAttr(rmanGlobalData["rmanGlobals"]["Filter"])

        print "Debug : rmanGlobals Import success"
        return True
    except Exception as e:
        print "Error : %s" % e.message
        msg = cmds.confirmDialog(title="Warning!",
                                 message="import renderGlobal fail",
                                 messageAlign='center',
                                 icon="warning",
                                 button=["OK"],
                                 backgroundColor=[.5, .5, .5])
        return False

def exportStupidAOV(filePath):
    data = getStupidAOVData()
    if data == None:
        return

    if filePath.startswith('/netapp/dexter/show'):
        filePath.replace('/netapp/dexter/show', '/show')

    dirPath = os.path.dirname(filePath)

    if not os.path.exists(dirPath):
        # print "instal : ", "install -d -m 755 {0}".format(dirPath)
        os.system("install -d -m 755 {0}".format(dirPath))

    with open(filePath, 'w') as f:
        json.dump(data, f, indent=4)
        f.close()

    print "Debug : stupidAOV export success"

def importStupidAOV(filePath):
    try:
        jsonData = open(filePath, 'r').read()
        stupidData = json.loads(jsonData)

        stupidAOVNode = cmds.ls(type='stupidAOV')

        if len(stupidAOVNode) == 0:
            import lgtCommon
            stupidAOVNode = lgtCommon.createUniqueNode('stupidAOV')
        else:
            stupidAOVNode = stupidAOVNode[0]

        print stupidData

        for attr in stupidData["StupidAOV"].keys():
            # print '%s.%s' % (stupidAOVNode, attr)
            # print stupidData["StupidAOV"][attr]
            setAttr('%s.%s' % (stupidAOVNode, attr), stupidData["StupidAOV"][attr])

    except Exception as e:
        print "Error : %s" % e.message
        msg = cmds.confirmDialog(title="Warning!",
                                 message="import stupidAOV fail",
                                 messageAlign='center',
                                 icon="warning",
                                 button=["OK"],
                                 backgroundColor=[.5, .5, .5])
        return False