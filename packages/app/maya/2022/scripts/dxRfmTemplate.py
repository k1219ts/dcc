import rfm.rmanAssetsMaya as ram
import rfmShading
# from rfm.rmanAssets import RmanAsset
import dxRmanAsset
import os
import getpass
import copy

import dxRSCommon
import dxRfmGlobals

import maya.cmds as cmds

lightTypeList = ['PxrAovLight', 'PxrDiskLight', 'PxrDistantLight', 'PxrDomeLight', 'PxrEnvDayLight', 'PxrMeshLight', 'PxrPortalLight', 'PxrRectLight', 'PxrSphereLight',
                 'PxrBarnLightFilter', 'PxrBlockerLightFilter', 'PxrCookieLightFilter', 'PxrGoboLightFilter', 'PxrIntMultLightFilter', 'PxrRampLightFilter', 'PxrRodLightFilter']


# rmantd_edit @2017.05.18 $1
def exportAssetDexter( nodes, atype, infodict, outfile, alwaysOverwrite=True ):
    label = infodict['label']
    Asset = dxRmanAsset.dxRmanAsset( atype, label )

    for k, v in infodict.iteritems():
        if k == 'label':
            continue
        Asset.addMetadata(k, v)

    prmanVersion = (cmds.rman('getversion')).split(' ')[0]
    Asset.setCompatibility(hostName='Maya',
                           hostVersion=ram.mayaVersion(),
                           rendererVersion=prmanVersion)

    if atype is "nodeGraph":
        ram.parseNodeGraph( nodes, Asset )
    elif atype is "envMap":
        ram.parseTexture( nodes, Asset )
    else:
        raise ram.RmanAssetMayaError( "%s is not a known asset type !" % atype )

    if os.path.exists( outfile ):
        if cmds.about(batch=True) or alwaysOverwrite:
            cmds.warning( 'Replacing existing file: %s' % outfile )
        else:
            replace = cmds.confirmDialog(title='This file already exists !',
                                       message='Do you want to overwrite it ?',
                                       button=['Overwrite', 'Cancel'],
                                       defaultButton='Replace',
                                       cancelButton='Cancel',
                                       dismissString='Cancel')
            if replace == 'Cancel':
                return

    Asset.save( outfile, False )

##
# @brief      Import an asset into maya
#
# @param      filepath  full path to a *.rma directory
#
# @return     none
#
def importAssetDexter(filepath):
    # early exit
    if not os.path.exists(filepath):
        raise ram.RmanAssetMayaError("File doesn't exist: %s" % filepath)

    Asset = dxRmanAsset.dxRmanAsset()
    Asset.load(filepath, localizeFilePaths=True)
    assetType = Asset.type()
    outHierachy = copy.deepcopy(Asset.getMetadata("hierachy"))

    # compatibility check
    #
    if not ram.compatibilityCheck(Asset):
        return

    if assetType == "nodeGraph":
        # internalNodes are conversion nodes needed by prman but not by maya.
        # we will collect these nodes in createNodes to re-establish the
        # original connections.
        internalNodes = {}
        newNodes = ram.createNodes(Asset, internalNodes)
        ram.connectNodes(Asset, newNodes, internalNodes)
        for v in newNodes.itervalues():
            if cmds.objExists(v) and cmds.nodeType(v) == 'shadingEngine':
                return v

    elif assetType == "envMap":
        selectedLights = cmds.ls(sl=True, dag=True, shapes=True)
        # nothing selected ?
        if not len(selectedLights):
            domeLights = cmds.ls(type='PxrDomeLight')
            numDomeLights = len(domeLights)
            # create a dome light if there isn't already one in the scene !
            if numDomeLights == 0:
                selectedLights.append(
                    cmds.eval('rmanCreateNode -asLight "" PxrDomeLight'))
            # if there is only one, use that.
            elif numDomeLights == 1:
                selectedLights.append(domeLights[0])
        if len(selectedLights):
            envMapPath = Asset.envMapPath()
            for light in selectedLights:
                nt = cmds.nodeType(light)
                if nt == 'PxrDomeLight':
                    try:
                        cmds.setAttr('%s.lightColorMap' % light, envMapPath,
                                   type='string')
                    except:
                        msg = 'Failed to set %s.lightColorMap\n' % light
                        msg += ram.sysErr()
                        raise ram.RmanAssetMayaError(msg)
                else:
                    raise ram.RmanAssetMayaError("We only support PxrDomeLight !")
        else:
            raise ram.RmanAssetMayaError('Select a PxrDomeLight first !')
        # print("not implemented yet")
    else:
        raise ram.RmanAssetMayaError("Unknown asset type : %s" % assetType)

    return outHierachy


def recursiveNode(hiearchy, hierachyDict, lightNode):
    '''
    get hierachy info
    :param hiearchy: current position ~ parent of lightNode
    :param dict: save hierachy info
    :param lightNode:
    :return:
    '''
    if len(hiearchy) == 0:
        if not hierachyDict.has_key('nodes'):
            hierachyDict['nodes'] = []
        hierachyDict['nodes'].append(lightNode)
        return

    if not hierachyDict.has_key(hiearchy[0]):
        hierachyDict[hiearchy[0]] = {}

    recursiveNode(hiearchy[1:], hierachyDict[hiearchy[0]], lightNode)

def recursiveHiearchy(nodes, hiearchy, parent):
    '''
    make hierachy of nodes
    :param nodes: makes node
    :param hiearchy: hiearchy info
    :param parent: parent of node
    :return:
    '''
    for key in hiearchy.keys():
        if key == 'nodes':
            continue
        print(key)
        if not cmds.objExists(key):
            if parent == None:
                cmds.createNode('transform', n=key)
            else:
                cmds.createNode('transform', n=key, p=parent)
        recursiveHiearchy(hiearchy[key]['nodes'], hiearchy[key], key)

    if parent == None or nodes == None:
        return

    cmds.parent(nodes, parent)


def getPxrLightingObject():
    '''
    get Pxr Light & filter NodesList
    :return: lightList and light hiearachy
    '''

    dict = {}
    lightList = cmds.ls(type = lightTypeList)
    for shape in cmds.ls(type=lightTypeList, l=True):
        hierachyNode = shape.split('|')
        recursiveNode(hierachyNode[1:-2], dict, hierachyNode[-1])

    return lightList, dict

def getBindingRules():
    '''
    get binding Rules data in maya scene
    :return:
    '''
    scopeData = rfmShading.getCurrentRlf()
    rules = scopeData.GetRules()
    rules.reverse()
    ruleData = list()
    sgObj = list()
    for r in rules:
        payloadID = r.GetPayloadId()
        ruleData.append((r.GetRuleString(), payloadID))
        sgObj.append(payloadID)

    return ruleData, sgObj

##
# @brief      Import an asset into maya
#
# @param      filepath  full path to a *.rma directory
#
# @return     none
#
def importAssetLight(filepath):
    outHierachy = importAssetDexter(filepath)

    print(outHierachy, "in importAssetLight")

    # hierachyPath = filepath.replace('ris_light', 'light_hierachy')
    # jsonData = open(hierachyPath, 'r').read()
    # hierachyData = json.loads(jsonData)

    recursiveHiearchy(None, outHierachy, None)

def importAssetShader(filepath):
    importAssetDexter(filepath)

def importBindingRule(filepath):
    rlfMode = "rlfAdd"
    rfmShading.importRlf(filepath, rlfMode)

def exportTemplate(dirPath, light = False, shader = False, binding = False):
    if dirPath.startswith('/netapp/dexter/show'):
        dirPath.replace('/netapp/dexter/show', '/show')

    if not os.path.exists(dirPath):
        print("instal : ", "install -d -m 755 {0}".format(dirPath))
        os.system("install -d -m 755 {0}".format(dirPath))

    try:
        shot = dirPath.split('/')[5]
    except:
        shot = dirPath.split('/')[-1]

    outputFiles = {}

    if light:
        lightPath = os.path.join(dirPath, "{SHOT}_ris_light.json".format(SHOT=shot))
        hierachyPath = os.path.join(dirPath, "{SHOT}_light_hierachy.json".format(SHOT=shot))
        lightNodes, hiearachy = getPxrLightingObject()
        label = "{SHOT}_ris_light".format(SHOT = shot)
        metaData = {"label":label,
                    "author":getpass.getuser(),
                    "version":os.path.dirname(dirPath),
                    "hierachy":hiearachy}

        exportAssetDexter(lightNodes, "nodeGraph", metaData, lightPath)

        # jsonFile = open(hierachyPath, 'w')
        # json.dump(hiearachy, jsonFile, indent=4)
        # jsonFile.close()
        outputFiles['light_path'] = [lightPath]
        # outputFiles['light_hierachy_path'] = [hierachyPath]

    if shader or binding:
        bindingRules, sgList = getBindingRules()

        if shader:
            shaderPath = os.path.join(dirPath, "{SHOT}_ris_material.json".format(SHOT=shot))
            label = "{SHOT}_ris_material".format(SHOT=shot)
            metaData = {"label": label,
                        "author": getpass.getuser(),
                        "version": os.path.dirname(dirPath),
                        "RuleString": bindingRules}

            exportAssetDexter(sgList, 'nodeGraph', metaData, shaderPath)

            outputFiles['shader_path'] = [shaderPath]

        if binding:
            bindingPath = os.path.join(dirPath, "{SHOT}_ris_material.xml".format(SHOT=shot))
            rfmShading.exportRlf(bindingPath)

            outputFiles['binding_path'] = [bindingPath]

    renderSetupPath = os.path.join(dirPath, "{SHOT}_renderSetup.json".format(SHOT = shot))
    dxRSCommon.rs_export(renderSetupPath)
    outputFiles['rendersetup_path'] = [renderSetupPath]

    renderGlobalPath = os.path.join(dirPath, "{SHOT}_renderGlobal.json".format(SHOT=shot))
    dxRfmGlobals.exportRmanGlobals(renderGlobalPath)
    outputFiles['renderglobal_path'] = [renderGlobalPath]

    stupidAOVPath = os.path.join(dirPath, "{SHOT}_stupidAOV.json".format(SHOT=shot))
    dxRfmGlobals.exportStupidAOV(stupidAOVPath)
    outputFiles['stupidAOV_path'] = [stupidAOVPath]

    return outputFiles

def importRfmTemplate(show, shot):
    # IP config
    import dxConfig
    DB_IP = dxConfig.getConf('DB_IP')

    import pymongo
    from pymongo import MongoClient
    client = MongoClient(DB_IP)
    db = client['PIPE_PUB']
    coll = db[show]

    ### LIGHTING TEMPLATE
    result = coll.find({"data_type": "template", 'enabled': True, 'shot': shot
                        }).sort('version', pymongo.DESCENDING).limit(1)
    if result.count() > 0:
        # SHADER
        if result[0]['files'].has_key('shader_path'):
            shader_path = result[0]['files']['shader_path'][0]
            importAssetShader(shader_path)

        # LIGHTING
        if result[0]['files'].has_key('light_path'):
            light_path = result[0]['files']['light_path'][0]
            importAssetLight(light_path)

        # BINDING
        if result[0]['files'].has_key('binding_path'):
            binding_path = result[0]['files']['binding_path'][0]
            importBindingRule(binding_path)

        renderSetupPath = result[0]['files']['rendersetup_path'][0]
        renderGlobalPath = result[0]['files']['renderglobal_path'][0]
        stupidAOVPath = result[0]['files']['stupidAOV_path'][0]

        # RENDER SETUP
        dxRSCommon.rs_import(renderSetupPath)
        # RENDER GLOBAL
        dxRfmGlobals.importRmanGlobals(renderGlobalPath)
        # STUPID AOV
        dxRfmGlobals.importStupidAOV(stupidAOVPath)
    else:
        msg = cmds.confirmDialog(title="Warning!",
                                 message="Don't find light keyshot data",
                                 messageAlign='center',
                                 icon="warning",
                                 button=["OK"],
                                 backgroundColor=[.5, .5, .5])
