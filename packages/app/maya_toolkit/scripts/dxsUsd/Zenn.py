'''
USD ZENN Export

1. if not nodes,
    find node by 'ZN_ExportSet' objectSet

Script Command
> dxsUsd.ZennAssetExport(showDir=$showDir, asset=$assetName, version=$ver).doIt()
'''

import os
import re
import glob
import string
import json

from pxr import Sdf, Usd, UsdGeom, UsdShade
import maya.cmds as cmds

import MsgSender
import dxsMsg
import Arguments
import dxsMayaUtils
import dxsUsdUtils
import PathUtils
import SessionUtils
import PackageUtils
import ClipUtils
import EditPrim
import Texture
import Material


def GetZennBaseMesh(nodes=None):
    baseMeshes = list()
    if nodes:
        for z in nodes:
            for n in cmds.listHistory(z):
                if cmds.nodeType(n) == 'ZN_Import':
                    shapes = cmds.listConnections(n, s=True, sh=True, type='surfaceShape')
                    if shapes:
                        baseMeshes += cmds.ls(shapes, l=True)
    else:
        for n in cmds.ls(type='ZN_Import'):
            shapes = cmds.listConnections(n, s=True, sh=True, type='surfaceShape')
            if shapes:
                baseMeshes += cmds.ls(shapes, l=True)
    return list(set(baseMeshes))


class ZennScene:
    def __init__(self, filename, open=False, show='', shot='', user=''):
        cmds.file(new=True, f=True)
        dxsMayaUtils.PluginSetup(['backstageMenu', 'pxrUsd', 'ZENNForMaya'])

        if open:
            args = {
                'o': True, 'type': 'mayaBinary', 'iv': True, 'options': 'v=0;', 'pr': True
            }
        else:
            args = {
                'i': True, 'type': 'mayaBinary', 'iv': True, 'mnc': False,
                'rpr': os.path.splitext(os.path.basename(filename))[0],
                'options': 'v=0;', 'pr': True
            }
        cmds.file(filename, **args)

        # initialize
        if not cmds.ls('ZN_ExportSet', r=True):
            MsgSender.sendMsg("# msg : Not found 'ZN_ExportSet' - %s" % filename, show, shot, user)

        # Member Variables
        self.zennExportNodes   = list()
        self.zennImportNodeMap = dict() # {zn: [zi, ..], ..}
        self.zennBaseMeshes    = list()

        for s in cmds.ls('ZN_ExportSet', r=True):
            self.zennExportNodes += cmds.sets(s, q=True)

        if not self.zennExportNodes:
            return

        self.initialize()
        self.getZennBaseMesh()


    def initialize(self):
        for znode in self.zennExportNodes:
            for n in cmds.listHistory(znode):
                if cmds.nodeType(n) == 'ZN_Import':
                    restTime = cmds.getAttr('%s.restTime' % n)
                    cmds.currentTime(restTime)
                    cmds.setAttr('%s.updateMesh' % n, 1)

                    if not self.zennImportNodeMap.has_key(znode):
                        self.zennImportNodeMap[znode] = list()
                    self.zennImportNodeMap[znode].append(n)

            cmds.dgeval(znode)

    def getZennBaseMesh(self):
        baseMeshes = list()
        for znode in self.zennImportNodeMap:
            for zi in self.zennImportNodeMap[znode]:
                shapes = cmds.listConnections(zi, s=True, sh=True, type='surfaceShape')
                if shapes:
                    baseMeshes += cmds.ls(shapes, l=True)
        if baseMeshes:
            self.zennBaseMeshes = list(set(baseMeshes))




class GetExportData:
    '''
    Returns:
        (dict): {
            node: {'output': output base path name, 'attr': {'txAttrs': {}, 'modelVersion': ''}}
        }
    '''
    def __init__(self, outDir, nodes, excludeAttrs = ['txmultiUV', 'txVersion']):
        self.outDir = outDir
        self.nodes  = nodes
        self.excludeAttrs = excludeAttrs
        self._msg   = list()

    def doIt(self):
        result = dict() # {'node': {'output': xxx, 'attr': dict()}}
        for n in self.nodes:
            result[n] = dict()
            name   = n.split(':')[-1]
            output = '{DIR}/{NODE}/{NODE}'.format(DIR=self.outDir, NODE=name)
            result[n]['output'] = output
            result[n]['attr']   = self.getMeshAttributes(n)
        if self._msg:
            dxsMsg.Print('dialog', list(set(self._msg)))
        return result

    def getMeshAttributes(self, node):
        zimport_nodes = list()
        for zn in cmds.listHistory(node, bf=True):
            if cmds.nodeType(zn) == 'ZN_Import':
                zimport_nodes.append(zn)
        if not zimport_nodes:
            return
        result = dict()

        inBodyMesh = cmds.listConnections('%s.inBodyMesh' % zimport_nodes[0], sh=True)[0]
        result['inBodyMesh'] = inBodyMesh

        if cmds.attributeQuery('modelVersion', n=inBodyMesh, ex=True):
            getVal = cmds.getAttr(inBodyMesh + '.modelVersion')
            if getVal:
                result['modelVersion'] = getVal
        if not result.has_key('modelVersion'):
            _msg = "[Zenn.GetExportData] - Not found 'modelVersion' attribute -> %s" % inBodyMesh
            self._msg.append(_msg)

        txAttrs = dict()
        # excludeAttrs = ['txmultiUV', 'txVersion']
        attrs = cmds.listAttr(inBodyMesh, ud=True, st=['rman__riattr__user_tx*', 'tx*'])
        if attrs:
            for ln in attrs:
                attrName = ln.replace('rman__riattr__user_', '')
                if not attrName in self.excludeAttrs:
                    attrVal  = cmds.getAttr(inBodyMesh + '.' + ln)
                    if attrName == 'txAssetName':
                        attrVal = 'asset/' + attrVal.split('/')[-1] + '/texture'
                    if attrName == 'txLayerName':
                        attrVal = attrVal + '_ZN'
                    txAttrs[attrName] = attrVal
        result['txAttrs'] = txAttrs
        return result



class ZennGeomExport(Arguments.CommonArgs):
    '''
    Args:
        outDir (str): version directory path
        customData (dict): {'layer': {...}, 'prim': {...}}
    '''
    def __init__(self, outDir, nodes=None, customData=None, **kwargs):
        self.outDir = outDir
        self.nodes  = nodes
        self.customData = customData
        Arguments.CommonArgs.__init__(self, **kwargs)

        self.showDir = None
        if '/show/' in self.outDir:
            splitPath = self.outDir.split('/')
            index = splitPath.index('show')
            self.showDir = string.join(splitPath[:index+2], '/')
            # self.showDir, self.showName = PathUtils.GetRootPath(self.showDir)

    def walkTimeline(self, stopFrame):
        startTime = int(cmds.currentTime(q = True))
        for i in range(startTime, stopFrame):
            print "# Debug : walkFrame (%s)" % i
            cmds.currentTime(i)

    def warningCheck(self):
        if not self.nodes:
            if not cmds.objExists('ZN_ExportSet'):
                dxsMsg.Print('warning', "[Zenn.ZennGeomExport] - not found 'ZN_ExportSet'")
                return False
            self.nodes = cmds.sets('ZN_ExportSet', q=True)
        if not self.nodes:
            dxsMsg.Print('warning', "[Zenn.ZennGeomExport] - not found 'ZENN' nodes")
            return False
        return True

    def doIt(self, postProcess=1):
        '''
        Zenn geom data export
        :param postProcess: 2 : both & merge file
                            1 : static
                            0 : dynamic
        :return:
        '''
        if not self.warningCheck():
            return
        dxsMsg.Print('warning', "[Zenn.ZennGeomExport Nodes] -> %s" % string.join(self.nodes, ', '))

        excludeAttrs = ['txmultiUV', 'txVersion']
        if postProcess == 2:
            excludeAttrs.remove("txVersion")

        # pre-frame setup
        print postProcess
        if postProcess != 1:
            self.walkTimeline(self.fr[0])

        jobCmd = list()
        outData = GetExportData(self.outDir, self.nodes, excludeAttrs).doIt()
        for node in outData:
            cmd = '-f %s' % outData[node]['output']
            cmd+= ' -n %s' % node
            # -ct -> static : only attribute, dynamic : perframe data, both : static, dynamic
            if self.fr[0] != self.fr[1]:
                if postProcess == 2:
                    cmd += ' -ct both'
                    cmd += ' -mgf'
                elif postProcess == 1:
                    cmd += ' -ct static'
                else:
                    cmd += ' -ct dynamic'
            jobCmd.append(cmd)

        cmds.ZN_ExportUSDCmd(j=jobCmd, startFrame=self.fr[0], endFrame=self.fr[1], step=self.step, v=True)
        dxsMsg.Print('info', "Creating stage path '%s'" % self.outDir)

        if postProcess == 2:
            geomData = {'high':list(), 'low':list()}
            for node in outData:
                output = outData[node]['output']
                highfile = output + ".high_geom.usd"
                UpdateAttributes(highfile, outData, self.customData, "")
                lowfile = output + ".low_geom.usd"
                UpdateAttributes(lowfile, outData, self.customData, "")
                geomData["high"].append(highfile)
                geomData["low"].append(lowfile)
            return geomData
        elif postProcess == 1:
            highgeom, lowgeom = self.outDataCollapsed(outData)
            geomFile = '{DIR}/zn_deforms.usd'.format(DIR=self.outDir)
            args = {'SdfPath': '/ZN_Global/zenn', 'comment': self.comment}
            if self.customData:
                if self.customData.has_key('layer'):
                    args['customLayerData'] = self.customData['layer']
                if self.customData.has_key('prim'):
                    args['customPrimData'] = self.customData['prim']
            SessionUtils.MakeReferenceStage(geomFile, [(highgeom, 'render'), (lowgeom, 'proxy')], **args)
            dxsMsg.Print('info', "Creating stage file '%s'" % geomFile)
            Material.Main(geomFile)
            return geomFile


    def outDataCollapsed(self, outData):    # outData: GetExportData
        highRefs = list()
        lowRefs  = list()
        for node in outData:
            output = outData[node]['output']
            if self.fr[0] == self.fr[1]:    # just one frame
                highfile = output + '.high_geom.usd'
                lowfile  = output + '.low_geom.usd'
            else:
                # high
                srcfile  = output + '.high_geom.*.usd'
                highfile = output + '.high_geom.usd'
                ClipUtils.MergeCoalesceFiles(srcfile, self.fr, step=self.step, outFile=highfile, mergeFrames=[], modify=False).doIt()
                # low
                srcfile = output + '.low_geom.*.usd'
                lowfile = output + '.low_geom.usd'
                ClipUtils.MergeCoalesceFiles(srcfile, self.fr, step=self.step, outFile=lowfile, mergeFrames=[], modify=False).doIt()

            highRefs.append((highfile, None))
            lowRefs.append((lowfile, None))
        highgeom = '{DIR}/zn_deforms.high_geom.usd'.format(DIR=self.outDir)
        SessionUtils.MakeReferenceStage(highgeom, highRefs, SdfPath='/ZN_Global', addChild=True, comment=self.comment)
        UpdateAttributes(highgeom, outData, self.customData)
        lowgeom  = '{DIR}/zn_deforms.low_geom.usd'.format(DIR=self.outDir)
        SessionUtils.MakeReferenceStage(lowgeom, lowRefs, SdfPath='/ZN_Global', addChild=True, comment=self.comment)
        UpdateAttributes(lowgeom, outData, self.customData)
        return highgeom, lowgeom


class UpdateAttributes:
    def __init__(self, geomfile, outData, customData, prefixPath='/ZN_Global'):
        self.geomfile  = geomfile
        self.outData   = outData
        self.customData= customData
        self.prefixPath= prefixPath

        self.parsePath()
        # Member variables
        # rootDir

        self.doIt()

    def parsePath(self):
        splitPath = self.geomfile.split('/')

        if self.customData and self.customData.has_key('layer'):
            if self.customData['layer'].has_key('zennAssetFile'):
                zennAssetFile = self.customData['layer']['zennAssetFile']
                splitPath = zennAssetFile.split('/')

        if "show" in splitPath:
            self.rootDir = '/'.join(splitPath[:splitPath.index('show') + 2])
            try:
                self.rootDir, self.showName= PathUtils.GetRootPath(self.rootDir)
            except:
                pass
        elif "assetlib" in splitPath:
            self.rootDir = '/'.join(splitPath[:splitPath.index('assetlib') + 2])

    def doIt(self):
        stage = Usd.Stage.Open(self.geomfile)

        for node in self.outData:
            output  = self.outData[node]['output']
            nodeData= self.outData[node]['attr']
            if nodeData:
                modelVersion = 'v001'
                if nodeData.has_key('modelVersion'):
                    modelVersion = nodeData['modelVersion']

                primPath = '/' + os.path.basename(output)
                if self.prefixPath:
                    primPath = self.prefixPath + primPath
                prim = stage.GetPrimAtPath(primPath)
                if prim:
                    if nodeData.has_key('txAttrs'):
                        txAttrs = nodeData['txAttrs']
                        if txAttrs.has_key('txBasePath') and txAttrs.has_key('txLayerName'):
                            txBasePath = txAttrs['txBasePath']
                            txLayerName= txAttrs['txLayerName']

                            splitPath = txBasePath.split('/')
                            rootName  = splitPath[splitPath.index('asset') + 1]

                            # Class Prim
                            cprimPath = Sdf.Path('/_{NAME}_{LAYER}_txAttr'.format(NAME=rootName, LAYER=txLayerName))
                            if not stage.GetPrimAtPath(cprimPath):
                                self.CreateClassPrim(stage, cprimPath, txBasePath, txLayerName, modelVersion)
                            prim.GetInherits().AddInherit(cprimPath)

                    # userProperties:MaterialSet
                    dxsUsdUtils.CreateUserProperty(prim, 'MaterialSet', 'fur', Sdf.ValueTypeNames.String)
                    # for denoise
                    dxsUsdUtils.CreateConstPrimvar(prim, 'useHairNormal', 1, Sdf.ValueTypeNames.Float)

        stage.GetRootLayer().Save()
        del stage

    def CreateClassPrim(self, stage, primPath, txBasePath, txLayerName, modelVersion):
        if not txBasePath.startswith('/'):
            txBasePath = self.rootDir + '/' + txBasePath

        txAttrMap = {txBasePath: [(txLayerName, False)]}
        Texture.MakeTexAttr(txAttr=txAttrMap, modelVersion=modelVersion).doIt()
        Texture.MakePrvMtl(txAttr=txAttrMap).doIt()

        txAttrFile = '{PATH}/tex/tex.attr.usd'.format(PATH=txBasePath)
        cprim = stage.CreateClassPrim(primPath)
        cprim.SetPayload(Sdf.Payload(PathUtils.GetRelPath(self.geomfile, txAttrFile), Sdf.Path('/' + txLayerName)))
        vset = cprim.GetVariantSets().GetVariantSet('modelVersion')
        vset.SetVariantSelection(modelVersion)



#-------------------------------------------------------------------------------
#
#   ASSET
#
#-------------------------------------------------------------------------------
class ZennAssetExport(Arguments.AssetArgs):
    '''
    ZENN assert export
    '''
    def __init__(self, nodes=list(), **kwargs):
        dxsMayaUtils.PluginSetup(['backstageMenu', 'pxrUsd'])
        if not cmds.pluginInfo('ZENNForMaya', q=True, l=True):
            MsgSender.sendMsg('# msg : ZENNForMaya plugin setup error!')

        self.nodes = nodes
        Arguments.AssetArgs.__init__(self, **kwargs)
        if not self.outDir and self.assetDir:
            self.outDir = self.assetDir + '/zenn'
        self.computeVersion()

        self.mayafile = dxsMayaUtils.GetMayaFilename()
        if self.mayafile:
            self.comment = 'Generated with %s' % self.mayafile

    def warningCheck(self):
        if not self.nodes:
            if not cmds.objExists('ZN_ExportSet'):
                dxsMsg.Print('warning', "[Zenn.ZennAssetExport] - not found 'ZN_ExportSet'")
                return False
            self.nodes = cmds.sets('ZN_ExportSet', q=True)
        if not self.nodes:
            dxsMsg.Print('warning', "[Zenn.ZennAssetExport] - not found 'ZENN' nodes")
            return False
        return True

    def doIt(self):
        if not self.warningCheck():
            return

        ctime = int(cmds.currentTime(q=True))
        geomDir = '{DIR}/{VER}'.format(DIR=self.outDir, VER=self.version)
        geomFile= ZennGeomExport(geomDir, nodes=self.nodes, fr=(ctime, ctime), comment=self.comment).doIt()
        geomPayloadFile = geomFile.replace('.usd', '.payload.usd')
        SessionUtils.MakeReferenceStage(geomPayloadFile, [(geomFile, None)], SdfPath='/%s{zennVersion=%s}' % (self.assetName, self.version))
        self.makePackage(geomPayloadFile)

        self.zennAssetSetup()

    def makePackage(self, sourceFile):
        taskFile = '{DIR}/zenn.usd'.format(DIR=self.outDir)
        SessionUtils.MakeSubLayerStage(taskFile, [sourceFile])
        taskPayloadFile = taskFile.replace('.usd', '.payload.usd')
        SessionUtils.MakeReferenceStage(taskPayloadFile, [(taskFile, None)], SdfPath='/%s{taskVariant=model}' % self.assetName)
        if self.showDir:
            PackageUtils.AssetPackage(self.showDir, self.assetName, taskPayloadFile)

    def zennAssetSetup(self):
        # more first connected ZN_Generate to ZN_Deform(export nodes) by CVs Sample initialize 1
        orgCvsRatioDict = {}
        for node in self.nodes:
            deformNode = node
            if cmds.nodeType(node) == "ZN_FeatherInstance":
                deformNode = cmds.listConnections("%s.inStrands" % node)[0]
            if cmds.nodeType(deformNode) == "ZN_Deform":
                generateNode = cmds.listConnections("%s.inStrands" % deformNode)
                if generateNode and cmds.nodeType(generateNode[0]) == "ZN_Generate":
                    orgCvsRatioDict[generateNode[0]] = cmds.getAttr("%s.cvSamplingRatio" % generateNode[0])
                    cmds.setAttr("%s.cvSamplingRatio" % generateNode[0], 1)

        # first save as scene.
        if not os.path.exists(os.path.join(self.outDir, "scenes")):
            os.makedirs(os.path.join(self.outDir, "scenes"))

        self.zennPubSceneFile = os.path.join(self.outDir, "scenes",
                                "{ASSET}_hair_{VER}.mb".format(ASSET=self.assetName, VER=self.version))
        # cmds.file(rename=filename)
        cmds.file(save=True)
        orgFileName = cmds.file(q=True, sn=True)
        cmd = "cp -rf %s %s" % (orgFileName, self.zennPubSceneFile)
        os.system(cmd)

        # insert 190614, if save maya scene file, write json file from ZN_ExportSet
        print "# Debug : ZENN Nodes", self.nodes
        exportSetJsonData = {"ZN_ExportSet":self.nodes}
        jsonfilename = self.zennPubSceneFile.replace(".mb", ".json")
        with open(jsonfilename, "w") as fw:
            json.dump(exportSetJsonData, fw, indent = 4)

        # recorvery
        orgCvsRatioDict = {}
        for generateNodeName in orgCvsRatioDict.keys():
            cmds.setAttr("%s.cvSamplingRatio" % generateNodeName, orgCvsRatioDict[generateNodeName])

        # second get rig version
        rigVersion = ""
        xBlockNodes = cmds.ls(type = 'xBlock')
        if xBlockNodes:
            if len(xBlockNodes) > 1:
                dxsMsg.Print('dialog', "[xBlock is more than one] - xBlock use %s?" % xBlockNodes[0])

            xBlock = xBlockNodes[0]

            rigFilePath = cmds.getAttr("%s.importFile" % xBlock)

            if not rigFilePath:
                dxsMsg.Print('dialog', "[Get rig version failed] - Not found rig file")
                return

            rigVersion = os.path.dirname(rigFilePath).split("/")[-1]

        assetInfoFile = '{DIR}/_config/maya/AssetInfo.json'.format(DIR=self.showDir.replace("_pub", ""))

        if not os.path.exists(os.path.dirname(assetInfoFile)):
            os.makedirs(os.path.dirname(assetInfoFile))

        data = {}
        if os.path.exists(assetInfoFile):
            f = open(assetInfoFile, 'r')
            data = eval(f.read())

        if not data.has_key('zenn'):
            data['zenn'] = {}
        if not data['zenn'].has_key(self.assetName):
            data['zenn'][self.assetName] = {}

        task = "modelHair"
        if "hairSim" in self.mayafile:
            task = "rigHair"

        if not data['zenn'][self.assetName].has_key(task):
            data['zenn'][self.assetName][task] = {}
        checkKey = "{ASSET}_hair_{VER}".format(ASSET = self.assetName, VER = self.version)
        if not data['zenn'][self.assetName][task].has_key(checkKey):
            data['zenn'][self.assetName][task][checkKey] = {}

        data['zenn'][self.assetName][task][checkKey]["assetFile"] = self.zennPubSceneFile
        if not data['zenn'][self.assetName][task][checkKey].has_key("rigVersion"):
            data['zenn'][self.assetName][task][checkKey]['rigVersion'] = []
        if rigVersion and not rigVersion in data['zenn'][self.assetName][task][checkKey]["rigVersion"]:
            data['zenn'][self.assetName][task][checkKey]["rigVersion"].append(rigVersion)

        if task == "rigHair":
            data['zenn'][self.assetName][task][checkKey]["shotList"] = []

        with open(assetInfoFile, "w") as fw:
            json.dump(data, fw, indent = 4)


#-------------------------------------------------------------------------------
#
#   SHOT
#
#-------------------------------------------------------------------------------
class ZennShotExport(Arguments.ShotArgs):
    '''
    Args:
        customData (dict): {
            'layer': {...}, 'prim': {...},
            'dependency': {'task': [], 'version': []},
        }
    '''
    def __init__(self, nsLayer=None, zennNodes=None, customData=None, **kwargs):
        dxsMayaUtils.PluginSetup(['backstageMenu', 'pxrUsd', 'ZENNForMaya'])
        self.nsLayer   = nsLayer
        self.zennNodes = zennNodes
        self.customData= customData

        Arguments.ShotArgs.__init__(self, **kwargs)
        if not self.outDir and self.shotDir:
            self.outDir = self.shotDir + '/zenn'
            self.outDir+= '/' + nsLayer
        else:
            self.nsLayer = os.path.basename(self.outDir)
        self.computeVersion()

        self.expfr = self.fr
        if self.fr[0] != self.fr[1]:
            self.expfr = (self.fr[0] - 1, self.fr[1] + 1)
        self.geomDir = '{DIR}/{VER}'.format(DIR=self.outDir, VER=self.version)
        self.fps = dxsMayaUtils.GetFPS()

    def warningCheck(self):
        if not self.zennNodes:
            if not cmds.objExists('ZN_ExportSet'):
                dxsMsg.Print('warning', "[Zenn.ZennShotExport] - not found 'ZN_ExportSet'")
                return False
            self.zennNodes = cmds.sets('ZN_ExportSet', q=True)
        if not self.zennNodes:
            dxsMsg.Print('warning', "[Zenn.ZennShotExport] - not found 'ZENN' nodes")
            return False
        return True

    def doIt(self, postProcess=1):
        if not self.warningCheck():
            return

        geomFile = ZennGeomExport(self.geomDir, nodes=self.zennNodes, customData=self.customData, fr=self.expfr, step=self.step, comment=self.comment).doIt(postProcess)
        if postProcess:
            geomPayloadFile = self.makeGeomMaster(geomFile)
            self.makePackage(geomPayloadFile)


    def makeGeomMaster(self, sourceFile):
        geomPayloadFile = sourceFile.replace('.usd', '.payload.usd')
        stage = SessionUtils.MakeInitialStage(geomPayloadFile, clear=True, fr=self.fr, fps=self.fps)
        dprim = stage.DefinePrim('/' + self.nsLayer, 'Xform')
        dxsUsdUtils.SetModelAPI(dprim, kind='component', name=self.nsLayer)
        stage.SetDefaultPrim(dprim)

        if not self.customData:
            payload = Sdf.Payload(PathUtils.GetRelPath(geomPayloadFile, sourceFile))
            dxsUsdUtils.SetPayload(dprim, payload)
            stage.GetRootLayer().Save()
            return geomPayloadFile

        dependency = self.customData['dependency']
        dependency['task'].append('zenn')
        dependency['version'].append(self.version)

        def AddVariantSet(editcontext, prim, name, value):
            if editcontext:
                editcontext.__enter__()
            vset = dxsUsdUtils.VariantSelection(prim, name, value)
            return vset.GetVariantEditContext()

        ectx = None
        for i in range(len(dependency['task'])):
            task = dependency['task'][i]
            ver  = dependency['version'][i]
            ectx = AddVariantSet(ectx, dprim, '%sVersion' % task, ver)
        ectx.__enter__()
        payload = Sdf.Payload(PathUtils.GetRelPath(geomPayloadFile, sourceFile))
        dxsUsdUtils.SetPayload(dprim, payload)
        ectx.__exit__(None, None, None)
        stage.GetRootLayer().Save()
        return geomPayloadFile


    def makePackage(self, sourceFile):
        # NSLayer
        layerFile = '{DIR}/{NS}.usd'.format(DIR=self.outDir, NS=self.nsLayer)
        SessionUtils.MakeSubLayerStage(layerFile, [sourceFile])
        layerPayloadFile = layerFile.replace('.usd', '.payload.usd')
        SessionUtils.MakeReferenceStage(layerPayloadFile, [(layerFile, None)], SdfPath='/rig/%s' % self.nsLayer, Kind='assembly', clear=True, fr=self.fr, fps=self.fps)
        # ZENN
        zennFile = '{DIR}/zenn/zenn.usd'.format(DIR=self.shotDir)
        SessionUtils.MakeSubLayerStage(zennFile, [layerPayloadFile])
        zennPayloadFile = zennFile.replace('.usd', '.payload.usd')
        SessionUtils.MakeReferenceStage(zennPayloadFile, [(zennFile, None)], SdfPath='/shot/rig', Name=self.shotName, Kind='assembly', clear=True, fr=self.fr, fps=self.fps)
        # shot
        PackageUtils.ShotPackage(self.showDir, self.seqName, self.shotName, zennPayloadFile, fr=self.fr, fps=self.fps)
        self.overrideVersion()

    def overrideVersion(self):
        shotFile = '{DIR}/shot/{SEQ}/{SHOT}/{SHOT}.usd'.format(DIR=self.showDir, SEQ=self.seqName, SHOT=self.shotName)
        PackageUtils.VersionSelect(shotFile, '/shot/rig/' + self.nsLayer, 'zennVersion', self.version)

        shotLgtFile = shotFile.replace('.usd', '.lgt.usd')
        if os.path.exists(shotLgtFile):
            PackageUtils.VersionSelect(shotLgtFile, '/shot/rig/' + self.nsLayer, 'zennVersion', self.version)



class ZennClipExport(Arguments.AssetArgs):
    def __init__(self, nodes=None, fr=(0, 0), step=1.0, overWrite=True, nsLayer = "", **kwargs):
        dxsMayaUtils.PluginSetup(['backstageMenu', 'pxrUsd', 'ZENNForMaya'])
        self.isPurpose = True
        self.overWrite = overWrite
        self.nsLayer = nsLayer

        self.nodes = nodes

        Arguments.AssetArgs.__init__(self, **kwargs)
        if not self.outDir and self.assetDir:
            self.outDir = '{DIR}/clip'.format(DIR=self.assetDir)
        self.computeVersion()

        self.fps = dxsMayaUtils.GetFPS()
        # frameRange
        self.fr  = fr
        self.step= step

        # loopScales, loopRange - compute by geom file

        self.mayafile = dxsMayaUtils.GetMayaFilename()
        if self.mayafile:
            self.comment = 'Generated with %s' % self.mayafile

    def parseGeom(self, geomDir):
        # get loopScales
        regexPattern = geomDir.replace('_clip', '_loop*')
        loopScaleList= glob.glob(regexPattern)
        self.loopScales = list()
        for loopScaleDir in loopScaleList:
            loopScaleValue = os.path.basename(loopScaleDir).split('_loop')[-1]
            value = loopScaleValue.replace('_', '.')
            self.loopScales.append(float(value))

        # get geom frameRange
        if not self.fr[0] or not self.fr[1]:
            geomMasterFile = os.path.join(geomDir, geomDir.split('/')[-1] + '.usd')
            geomLayer = Sdf.Layer.FindOrOpen(geomMasterFile)
            if geomLayer:
                self.fr = (int(geomLayer.startTimeCode), int(geomLayer.endTimeCode))
            else:
                self.fr = dxsMayaUtils.GetFrameRange()


    def doIt(self):
        '''
        1. export zenn sequence data.
        2. $assetName_clip.usd insert reference
        3.
        :return:
        '''
        clipLayer = 'default'#  if not self.assetName else self.assetName

        if self.nsLayer:
            clipLayer = self.nsLayer
        elif self.assetName:
            clipLayer = self.assetName
        clipLayer += '_clip'

        geomDir = '{DIR}/{VER}/{LAYER}'.format(DIR=self.outDir, VER=self.version, LAYER=clipLayer)
        self.parseGeom(geomDir)
        geomFileData = ZennGeomExport(geomDir, nodes=self.nodes, fr=(self.fr[0]-1, self.fr[1]+1), step=self.step, comment=self.comment).doIt(postProcess=2)

        # Default Prim Name
        self.rootName = self.assetName
        geomMasterFile = self.makeGeomPackage(geomFileData, clipLayer)
        result = None
        if self.loopScales:
            result = self.makeGeomLoopClip(geomMasterFile, geomFileData)

        if not result:
            return

        taskPayloadFile = self.makeLoopClipPackage(result, clipLayer)
        if self.showDir and self.assetDir:
            PackageUtils.AssetPackage(self.showDir, self.assetName, taskPayloadFile)


    def makeGeomPackage(self, geomFileData, clipLayer):
        outDir = '{DIR}/{VER}/{LAYER}'.format(DIR=self.outDir, VER=self.version, LAYER=clipLayer)
        highgeom = '{DIR}/zn_deforms.high_geom.usd'.format(DIR=outDir)
        sourceData = list()
        for f in geomFileData['high']:
            sourceData.append((f, None))
        SessionUtils.MakeReferenceStage(highgeom, sourceData, SdfPath='/ZN_Global', addChild=True, comment=self.comment)
        lowgeom = '{DIR}/zn_deforms.low_geom.usd'.format(DIR=outDir)
        sourceData = list()
        for f in geomFileData['low']:
            sourceData.append((f, None))
        SessionUtils.MakeReferenceStage(lowgeom, sourceData, SdfPath='/ZN_Global', addChild=True, comment=self.comment)

        geomFile = '{DIR}/zn_deforms.usd'.format(DIR=outDir)
        SessionUtils.MakeReferenceStage(geomFile, [(highgeom, 'render'), (lowgeom, 'proxy')], SdfPath='/ZN_Global', comment=self.comment)

        masterFile = '{DIR}/{LAYER}.usd'.format(DIR=outDir, LAYER=clipLayer)
        SessionUtils.MakeReferenceStage(masterFile, [(geomFile, None)], SdfPath='/%s/zenn' % self.assetName, comment=self.comment)
        Material.Main(masterFile)
        return masterFile


    def makeGeomLoopClip(self, inputFile, geomFileData):
        layerMasters = list()
        loopExp = ClipUtils.LoopClip(inputFile, scales=self.loopScales)
        loopExp.clip_geomFiles = geomFileData['high'] + geomFileData['low']
        for ts in loopExp.timeScales:
            loopFileData = {'high': list(), 'low': list()}
            scaleName = str(ts).replace('.', '_')
            loopName  = loopExp.loopLayer + scaleName
            geomDir   = '{DIR}/{VER}/{LAYER}'.format(DIR=self.outDir, VER=self.version, LAYER=loopName)

            # reference geom file
            clipTimes = None
            if os.path.exists(geomDir):
                for gf in os.listdir(geomDir):
                    if 'high_geom_loop.usd' in gf:
                        refFile = os.path.join(geomDir, gf)
                        clipTimes = ClipUtils.LoopClip.GetClipTimes(refFile)

            for f in loopExp.clip_geomFiles:
                baseName = os.path.basename(f)
                baseName = baseName.replace('.usd', '_loop.usd')
                geomfile = os.path.join(geomDir, baseName)
                loopExp.makeValueClipGeom(geomfile, f, ts, clipTimes)
                if 'high_geom' in geomfile:
                    loopFileData['high'].append(geomfile)
                elif 'low_geom' in geomfile:
                    loopFileData['low'].append(geomfile)
            masterFile = self.makeGeomPackage(loopFileData, loopName)
            layerMasters.append(masterFile)
        return layerMasters


    def makeLoopClipPackage(self, sourceFiles, clipLayer):
        loopClip = '{DIR}/{VER}/loopClip.usd'.format(DIR=self.outDir, VER=self.version)
        loopLayer= clipLayer.replace('clip', 'loop')
        SessionUtils.MakeSubLayerStage(loopClip, sourceFiles, SdfPath='/%s{loopVariant=%s1_0}' % (self.rootName, loopLayer))
        loopClipPayload = loopClip.replace('.usd', '.payload.usd')
        SessionUtils.MakeReferenceStage(loopClipPayload, [(loopClip, None)], SdfPath='/%s{clipVersion=%s}' % (self.rootName, self.version))

        taskFile = '{DIR}/clip.usd'.format(DIR=self.outDir)
        SessionUtils.MakeSubLayerStage(taskFile, [loopClipPayload])
        taskPayloadFile = taskFile.replace('.usd', '.payload.usd')
        SessionUtils.MakeReferenceStage(taskPayloadFile, [(taskFile, None)], SdfPath='/%s{taskVariant=clip}' % self.assetName, clear=True)
        return taskPayloadFile
