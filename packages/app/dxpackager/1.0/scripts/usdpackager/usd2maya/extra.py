import maya.api.OpenMaya as OpenMaya
import maya.cmds as cmds

import DXUSD_MAYA.Utils as utl
import DXUSD_MAYA.MUtils as mutl
import DXUSD.Message as msg
import subprocess
from dxBlockUtils import Represent

import os
from pxr import Usd, UsdGeom, UsdShade, UsdUtils, Sdf, Kind

TEMP = '/tmp'

# ------------------------------------------------------------------------------
# Utils
# ------------------------------------------------------------------------------
class XformBlock:
    def __init__(self, node):
        self.m_translate = cmds.getAttr('%s.translate' % node)[0]
        self.m_rotate = cmds.getAttr('%s.rotate' % node)[0]
        self.m_scale = cmds.getAttr('%s.scale' % node)[0]

        self.m_rotatePivot = cmds.getAttr('%s.rotatePivot' % node)[0]
        self.m_scalePivot = cmds.getAttr('%s.scalePivot' % node)[0]
        self.m_rotatePivotTranslate = cmds.getAttr('%s.rotatePivotTranslate' % node)[0]
        self.m_scalePivotTranslate = cmds.getAttr('%s.scalePivotTranslate' % node)[0]

    def Set(self, node):
        cmds.setAttr('%s.rotatePivot' % node, *self.m_rotatePivot)
        cmds.setAttr('%s.rotatePivotTranslate' % node, *self.m_rotatePivotTranslate)
        cmds.setAttr('%s.scalePivot' % node, *self.m_scalePivot)
        cmds.setAttr('%s.scalePivotTranslate' % node, *self.m_scalePivotTranslate)

        cmds.setAttr('%s.scale' % node, *self.m_scale)
        cmds.setAttr('%s.rotate' % node, *self.m_rotate)
        cmds.setAttr('%s.translate' % node, *self.m_translate)


def getTime():
    import datetime
    now = datetime.datetime.now()
    time = str(now.year)
    time += str(now.month)
    time += str(now.day)
    time += str(now.hour)
    time += str(now.minute)
    time += str(now.second)
    return time


def deleteCache(tmpfile):
    import os
    if os.path.exists(tmpfile):
        suCmd = 'rm -rf %s' % tmpfile
        os.system(suCmd)
        msg.debug('deleted tmpfile')


def ReplaceToNewGroup(node):
    name = node.split('|')[-1].split(':')[-1]
    parentGrp = cmds.listRelatives(node, p=True, path=True)

    newGrp = cmds.group(em=True, name='%s_tmp' % name)
    if parentGrp:
        cmds.parent(newGrp, parentGrp)
        if newGrp.startswith('|'):
            newGrp = parentGrp[0] + newGrp
        else:
            newGrp = parentGrp[0] + '|' + newGrp

    # copy attribute values
    attrs = ['ro', 'v']
    for attr in attrs:
        v = cmds.getAttr('%s.%s' % (node, attr))
        cmds.setAttr('%s.%s' % (newGrp, attr), v)

    attrs = ['s', 'spt', 'sp', 'r', 'ra', 'rpt', 'rp', 't', 'sh']
    for attr in attrs:
        v = cmds.getAttr('%s.%s' % (node, attr))[0]
        cmds.setAttr('%s.%s' % (newGrp, attr), v[0], v[1], v[2], type='double3')

    # copy attribute connections
    _ = cmds.listConnections(node, s=True, d=False, p=True, c=True) or []
    connections = [[_[i * 2 + 1], _[i * 2]] for i in range(len(_) / 2)]
    connections.append(None)
    _ = cmds.listConnections(node, s=False, d=True, p=True, c=True) or []
    connections += [[_[i * 2], _[i * 2 + 1]] for i in range(len(_) / 2)]

    idx = 1
    for connection in connections:
        if connection == None:
            idx = 0
            continue
        cmds.disconnectAttr(connection[0], connection[1])
        elm = connection[idx].split('.')
        elm[0] = newGrp
        connection[idx] = '.'.join(elm)
        cmds.connectAttr(connection[0], connection[1], f=True)

    print '>>>newGrp:',newGrp


    # reparent children
    for child in cmds.listRelatives(node, c=True, s=False, f=True):
        print '>>>child:', child
        if cmds.objectType(child, isAType='shape'):
            continue
        cmds.parent(child, newGrp)

    return newGrp


def CreatePxrUsdProxyNode(filename):
    nodeName = filename.split("/")[-1].split(".")[0]
    node = cmds.createNode('pxrUsdProxyShape', n="%sShape" % nodeName)
    cmds.setAttr('%s.filePath' % node, filename, type='string')
    return cmds.listRelatives(node, p=True, f=True)[0]


def PrimToMayaPath(node, primPath):
    nodename = node.split('|')[-1]
    sufpath = '/'.join(primPath.split('/')[2:])
    mayapath = os.path.join('/' + nodename)
    if sufpath:
        mayapath = os.path.join('/' + nodename, sufpath)
    mayapath = mayapath.replace('/', '|')
    return mayapath


def GetMatrixByGf(position, orient, scale):
    tmtx = OpenMaya.MTransformationMatrix()
    tmtx.setScale([scale[0], scale[1], scale[2]], OpenMaya.MSpace.kWorld)

    img = orient.imaginary
    quat = OpenMaya.MQuaternion([img[0], img[1], img[2], orient.real])
    tmtx.setRotation(quat.asEulerRotation())

    tmtx.setTranslation(OpenMaya.MVector(*position), OpenMaya.MSpace.kWorld)
    return tmtx.asMatrix()


def SetDefaultTransform(node):
    for m in ['translate', 'rotate', 'scale']:
        cmds.setAttr('%s.%s' % (node, m), 0, 0, 0, type="double3")
        if m == 'scale':
            cmds.setAttr('%s.%s' % (node, m), 1, 1, 1, type="double3")
    cmds.setAttr('%s.%s' % (node, 'shearXY'), 0)
    cmds.setAttr('%s.%s' % (node, 'shearXZ'), 0)
    cmds.setAttr('%s.%s' % (node, 'shearYZ'), 0)

# ------------------------------------------------------------------------------
# Mesh
# ------------------------------------------------------------------------------
class USDProxyShapeToMesh:
    def __init__(self, nodes):
        if not nodes:
            return
        self.nodes = nodes

        RDATA = utl.GetReferenceData()
        self.refData = RDATA.get(self.nodes)
        print self.refData

    def doit(self):
        if not self.refData:
            return
        for node in self.nodes:
            # if 'buslimousine' in node:
            shape = node
            transNode = cmds.listRelatives(shape, p=True, f=True)[0]
            filePath = cmds.getAttr('%s.filePath' % shape)
            excludePaths = cmds.getAttr('%s.excludePrimPaths' % shape)
            primPath = cmds.getAttr('%s.primPath' % shape)
            nodename = transNode.split('|')[-1]
            impnode = self.ImportUsd(nodename,filePath, excludePaths, primPath)
            if impnode:
                transNode = cmds.listRelatives(shape, p=True, f=True)[0]
                try:
                    self.CopyNode(transNode, impnode)
                except:
                    pass
                self.DeleteXform()
        print 'Success'

    def DeleteXform(self):
        if cmds.objExists('*|scatter'):
            cmds.delete('*|scatter')
        if cmds.objExists('*|Looks'):
            cmds.delete('*|Looks')

    def UsdImport(self,filePath,exprims):
        deletelist = []
        impnode = cmds.usdImport(f=filePath, shd=None)[0]

        if exprims:
            for expath in exprims:
                mayapath = PrimToMayaPath(impnode, expath)
                deletelist.append(mayapath)
            cmds.delete(deletelist)
        return impnode


    def ImportUsd(self, nodename ,filePath, excludePaths,primPath):
        if not os.path.exists(filePath):
            return
        impnode = ''
        geomnode = ''
        stage = Usd.Stage.Open(filePath)
        dPrim = stage.GetDefaultPrim()

        primpathlist = []
        refprims = []
        sprims = []
        ptprims = []
        exprims = []
        geomprim = []

        if primPath:
            p = stage.GetPrimAtPath(primPath)
            if not p.GetAllChildren():
                primpathlist.append(primPath)

        if excludePaths:
            excludePaths = excludePaths.split(',')
            self.walk(dPrim, primPath, geomprim, primpathlist, refprims, sprims, ptprims, exprims, excludePaths)
        else:
            # if not primPath:
            self.walk(dPrim,primPath, geomprim, primpathlist, refprims, sprims, ptprims, exprims)

        # print('>>>primpathlist:', primpathlist)
        if not primpathlist:
            return

        try:
            # print '>>>dxusd import'
            impnode = self.UsdCatImport(filePath, primpathlist, nodename)
            # print('impnode:',impnode)
        except:
            # print '>>>usd maya import'
            impnode = self.UsdImport(filePath, exprims)
            self.DeleteXform()


        if not impnode:
            return

        impnode = cmds.rename(impnode,nodename)
        geomnode = cmds.listRelatives(impnode,c=True,f=True)[0]

        if not impnode:
            # pass
            dprimname = dPrim.GetPath().pathString
            mayapath = dprimname.replace('/', '|')
            geomnode = cmds.group(em = True, world=True,n='Geom')
            impnode =cmds.group(geomnode, n=mayapath)
            impnode = cmds.ls(impnode, l=1)[0]
            geomnode = cmds.listRelatives(impnode,c=True,f=True)[0]
            msg.debug('Create null root GRP:',impnode)

        if geomprim:
            p = geomprim[0]
            geompath = p.GetPath().pathString
            translate = p.GetAttribute('xformOp:translate').Get()
            rotate = p.GetAttribute('xformOp:rotateXYZ').Get()
            scale = p.GetAttribute('xformOp:scale').Get()
            geomMayapath = PrimToMayaPath(impnode, geompath)
            node = geomMayapath

            if translate:
                cmds.setAttr('%s.translate' % node, translate[0], translate[1], translate[2], type="double3")
            if rotate:
                cmds.setAttr('%s.rotate' % node, rotate[0], rotate[1], rotate[2], type="double3")
            if scale:
                cmds.setAttr('%s.scale' % node, scale[0], scale[1] , scale[2] , type="double3")


        if refprims:
            msg.debug('[reference import ]:', refprims)
            for p in refprims:
                # p = stage.GetPrimAtPath(primPath)
                path = p.GetPath().pathString
                print 'impnode:',impnode
                mayapath = PrimToMayaPath(impnode, path)

                if cmds.listRelatives(mayapath, c=True, f=True):
                    for n in cmds.listRelatives(mayapath, c=True, f=True):
                        print('childnode :', n)
                        if cmds.objExists(n):
                            cmds.delete(n)

            refData = self.references_doIt(refprims, impnode, type='reference')
            if refData:
                self.ImportUsdChild(refData, impnode)

        if ptprims:
            msg.debug('[pointInstancing import ]:', ptprims)
            self.PointInstance_doit(nodename,stage, ptprims, geomnode)

        if sprims:
            msg.debug('[instancing reference import ]:', sprims)
            refData =self.references_doIt(sprims, impnode, type='specialize')
            if refData:
                self.ImportUsdChild(refData, impnode)

        return impnode


    def PointInstance_doit(self, nodename, stage, ptprims, parentnode):
        ptinfo = []
        impnodes = []
        for p in ptprims:
            nodename = p.GetParent().GetName()
            data={}
            data['name'] = p.GetParent().GetName()
            data['filepath'] = []
            ptgeom = UsdGeom.PointInstancer(p)
            prototypes = ptgeom.GetPrototypesRel().GetTargets()
            for i in xrange(len(prototypes)):
                prim = stage.GetPrimAtPath(prototypes[i])
                stack = prim.GetPrimStack()
                spec = stack[0]
                refs = prim.GetMetadata('references')
                if refs:
                    identifier = spec.layer.identifier
                    assetPath = spec.referenceList.prependedItems[0].assetPath
                    filepath = os.path.abspath(os.path.join(utl.DirName(identifier), assetPath))
                    if not filepath in data['filepath']:
                        data['filepath'].append(filepath)
                    if not filepath in impnodes:
                        impnodes.append(filepath)

            indices = ptgeom.GetProtoIndicesAttr().Get()
            positions = ptgeom.GetPositionsAttr().Get()
            orients = ptgeom.GetOrientationsAttr().Get()
            scales = ptgeom.GetScalesAttr().Get()
            ids = ptgeom.GetIdsAttr().Get()
            data['positions'] = positions
            data['scales'] = scales
            data['orientations'] = orients
            data['indices'] = indices
            data['ids'] = ids
            data['nodename'] = nodename
            ptinfo.append(data)

        impData = {}
        newGrp = ''
        if impnodes:
            for path in impnodes:
                protoname = path.split("/")[-1].split(".")[0]
                try:
                    imp = cmds.usdImport(f=path, shd=None)[0]
                    deleteList=[]
                    for nullnode in cmds.ls('%s|Looks' % imp, type='transform'):
                        deleteList.append(nullnode)
                    if deleteList:
                        cmds.delete(deleteList)
                except:
                    print('pointInstancing import Error')
                    imp = cmds.spaceLocator(n=nodename)[0]

                checkpath = self.SourceTempGrp(imp)
                impnode = checkpath
                impData[path] = impnode

        if ptinfo:
            for i in xrange(len(ptinfo)):
                indices = ptinfo[i]['indices']
                positions = ptinfo[i]['positions']
                orients = ptinfo[i]['orientations']
                scales = ptinfo[i]['scales']
                protofiles = ptinfo[i]['filepath']
                nodename = ptinfo[i]['nodename']

                try:
                    importedPTgrp = cmds.delete(parentnode + '|' + nodename)
                except:
                    pass

                transNode = cmds.group(em=True, world=True, name = nodename)
                for n in xrange(len(indices)):
                    index = indices[n]
                    try:
                        matrix = GetMatrixByGf(positions[n], orients[n], scales[n])
                        filename = protofiles[index]
                        impnode = impData[filename]
                        self.CopyNode3(impnode, matrix, transNode)
                    except:
                        print('nodename:',nodename)
                        return

                tnode = cmds.parent(transNode, parentnode)[0]


    def CopyNode3(self,impnode,matrix,transNode):
        copynode = cmds.instance(impnode)[0]
        # print('[[[[[[[[[[[[[[[[instancing copy]]]]]]]]]]]]]]]]]]]]')
        copynode = cmds.parent(copynode, world=True)
        cmds.xform(copynode, m=matrix, ws=True)
        cmds.parent(copynode, transNode)


    def CopyNode(self, transNode, impnode):
        nodename = transNode.split('|')[-1]
        tmp= ReplaceToNewGroup(transNode)
        print ">>>tmp node:", tmp

        copynode = cmds.duplicate(impnode, n=nodename)[0]
        geomgroup = cmds.listRelatives(copynode, c=True, f=True)[0]
        xb = XformBlock(geomgroup)
        root = cmds.parent(geomgroup, tmp)[0] #Geom
        print 'root:',root
        try:
            SetDefaultTransform(root)
            xb.Set(root)
        except:
            print 'SetDefaultTransform Failed'
            print 'root:',root
            print 'transNode:',transNode
            print 'impnode:',impnode
            print 'copynode:',copynode
            print 'geomgroup:',geomgroup

        cmds.delete(transNode)
        cmds.delete(impnode)
        cmds.delete(copynode)
        cmds.rename(tmp,nodename)


    def CopyNode2(self, transNode, impnode):
        xb = Represent.XformBlock(transNode)
        parentNode = cmds.listRelatives(transNode, p=True, f=True)
        if parentNode:
            parentNode = parentNode[0]
        else:
            parentNode = None

        nodename = transNode.split('|')[-1]
        print 'impnode:',impnode
        try:
            copynode = cmds.instance(impnode, n=nodename)[0]
            print('[[[[[[[[[[[[[[[[instancing copy]]]]]]]]]]]]]]]]]]]]')
            copynode = cmds.ls(copynode, l=1)[0]
            root = cmds.parent(copynode, parentNode)[0]
            xb.Set(root)
            cmds.delete(transNode)
        except:
            print('error: [[[[[[[[[[[[[[[[instancing copy failed]]]]]]]]]]]]]]]]]]]]')



    def ImportUsdChild(self, refData, rootNode):
        print('refData:',refData)

        for k, v in refData.items():
            filePath = v['filePath']
            excludePaths = ''
            primPath = ''
            if v.has_key('primPath'):
                primPath = v['primPath']
            if v.has_key('excludePrimPaths'):
                excludePaths = v['excludePrimPaths']

            print('>>>filePath:', filePath)
            print('>>>excludePaths:', excludePaths)
            print('>>>primPath:', primPath)
            impnode = self.ImportUsd(k,filePath, excludePaths, primPath)

            print('impnode:>>>>',impnode)
            for node in v['nodes']:
                print('>>>node:',node)
                splitMayaPath = node.split('|')
                transNode = rootNode + '|' + '|'.join(splitMayaPath[2:])
                self.CopyNode2(transNode, impnode)

            self.SourceTempGrp(impnode)

    def SourceTempGrp(self, impnode):
        if not impnode:
            return
        newGrp = '|reference_sc_tmp'
        if not cmds.objExists(newGrp):
            newGrp = cmds.group(em=True, name='reference_sc_tmp')

        impname = impnode.split('|')[-1]
        checkpath = newGrp + '|' + impname
        if not cmds.objExists(checkpath):
            cmds.parent(impnode, newGrp)
        else:
            cmds.delete(impnode)

        cmds.setAttr('%s.visibility' % newGrp, 0)
        return checkpath

    def walk(self, prim, primPath,geomprim, primpathlist, refprims, sprims, ptprims, exprims, excludePaths=''):
        for p in prim.GetAllChildren():
            path = p.GetPath().pathString
            exclude = self.match(excludePaths, path, primPath)

            if '/Cam' in path or p.GetTypeName() == "Scope" :
                pass

            else:
                if p.GetTypeName() == 'PointInstancer':
                    msg.debug('[point instancing]:', path)
                    ptprims.append(p)

                elif p.GetTypeName() == 'Mesh':
                    if exclude == False:
                        if not path in primpathlist:
                            primpathlist.append(path)
                    else:
                        if not path in exprims:
                            exprims.append(path)

                # elif p.GetTypeName() == "Scope":
                #     if not path in exprims:
                #         exprims.append(path)

                else:
                    if p.GetParent().GetName() == 'Layout' or p.GetParent().GetName() == 'World':
                        self.walk(p, primPath,geomprim, primpathlist, refprims, sprims, ptprims, exprims, excludePaths)

                    else:
                        if p.HasAuthoredSpecializes():
                            if exclude == False:
                                if not path in primpathlist:
                                    primpathlist.append(path)
                                    # msg.debug('[sceneGraph instancing]:', path)
                                if not p in sprims:
                                    sprims.append(p)
                            else:
                                if not path in exprims:
                                    exprims.append(path)

                        elif p.HasAuthoredReferences():
                            if exclude == False:
                                if not path in primpathlist:
                                    primpathlist.append(path)
                                    # msg.debug('[Reference]:', path)
                                if not p in refprims:
                                    refprims.append(p)
                            else:
                                if not path in exprims:
                                    exprims.append(path)

                        elif p.GetTypeName() == 'Xform':
                            if not '/Looks' in path:
                                self.walk(p, primPath,geomprim, primpathlist, refprims, sprims, ptprims, exprims, excludePaths)


    def match(self, excludePaths, path, primPath):
        # print('path:',path)
        exclude = False

        if primPath:
            if not primPath in path:

                exclude = True

        if excludePaths:
            for ex in excludePaths:
                ex = ex.lstrip()
                if ex == path or '/scatter' in path:
                    exclude = True

        # if exclude == False:
        #     print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>primPath:', primPath)
        #     print('path:', path)

        return exclude


    def references_doIt(self, prims, impnode, type):
        refData = dict()
        nullPrims = list()
        excludeNames = list()
        print('>>>>>>>>>>prims:',prims)
        for prim in prims:
            info =dict()
            primPath = ''
            exps = ''

            path = prim.GetPath().pathString  # /jubangShip/Geom/Render/jubangShip_branch_GRP/boxH_model_GRP/boxH7
            node = path.replace('/', '|')
            stack = prim.GetPrimStack()

            if type == 'reference':
                info = self.references_getInfo(stack[0])
                print('info:', info)
            elif type == 'specialize':
                info = self.specializes_getInfo(stack[-1])

            if not info:
                return

            if info.has_key('primpath'):
                primPath = info['primpath']

            if info.has_key('excludePrimPaths'):
                exps = info['excludePrimPaths']
                print ('exps:',exps)


            # reference source file
            filePath = info.get('assetPath', '')
            baseName = utl.BaseName(filePath).split('.')[0]

            # primPath
            if primPath:
                baseName += '_' + Sdf.Path(primPath).name

            if exps:
                names = list()
                for p in exps.split(','):
                    names.append(Sdf.Path(p.strip()).name)
                name = '_'.join(names)
                if not name in excludeNames:
                    excludeNames.append(name)
                baseName += '_exp%s' % excludeNames.index(name)

            refName = baseName

            if not refData.has_key(refName):
                refData[refName] = {
                    'filePath': filePath, 'nodes': list()}
                refData[refName]['primPath'] = primPath
                refData[refName]['excludePrimPaths'] = exps

            refData[refName]['nodes'].append(node)

        return refData


    def references_getInfo(self, spec):
        identifier = spec.layer.identifier
        try:
            assetPath = spec.referenceList.prependedItems[0].assetPath
        except:
            return
        fullPath = os.path.abspath(os.path.join(utl.DirName(identifier), assetPath))
        data = {
            'assetPath': fullPath,
            'variants': Represent.Expanded().GetVariants(spec)
        }
        # custom data
        if spec.customData.has_key('excludePrimPaths'):
            data['excludePrimPaths'] = spec.customData.get('excludePrimPaths')
        if spec.customData.has_key('primPath'):
            data['primPath'] = spec.customData.get('primPath')
        return data

    def specializes_getInfo(self, spec):
        specializes_DATA = {}
        source = spec.nameChildren.get('source')
        if not source:
            return
        if not specializes_DATA.has_key(spec.path):
            info = self.references_getInfo(source)
            specializes_DATA[spec.path] = info
        return specializes_DATA[spec.path]


    def UsdCatImport(self, filePath, primpathlist,nodename):
        tmppath = '%s/%s_temp_%s.usd' % (TEMP, nodename,getTime())
        # print '>>>tmppath:', tmppath

        if len(primpathlist) ==1:
            primpath = primpathlist[0]
        else:
            primpath = ','.join(primpathlist)

        # print '>>>primpath:', primpath

        cmd = '/backstage/dcc/DCC rez-env usd_core-20.08'
        cmd += ' -- usdcat %s' % filePath
        cmd += ' -o %s' % tmppath
        cmd += ' -f'
        cmd += ' --mask %s' % primpath
        subprocess.Popen(cmd, shell=True).wait()
        # print 'usd cat import:', cmd
        if os.path.exists(tmppath):
            print '>>>import Flattened usd:',tmppath
            impnode = cmds.usdImport(f=tmppath, shd=None)[0]
            deleteCache(tmppath)
            return impnode
        else:
            return

# ------------------------------------------------------------------------------
# Locator
# ------------------------------------------------------------------------------
class USDProxyShapeToLocator(USDProxyShapeToMesh):
    def __init__(self, nodes):
        USDProxyShapeToMesh.__init__(self, nodes)


    def doit(self):
        print('self.refData:',self.refData)
        if not self.refData:
            return

        for shape in  self.nodes:
            filePath = cmds.getAttr('%s.filePath' % shape)
            excludePaths = cmds.getAttr('%s.excludePrimPaths' % shape)
            primPath = cmds.getAttr('%s.primPath' % shape)

            transNode = cmds.listRelatives(shape, p=True, f=True)[0]
            nodename = transNode.split('|')[-1]
            impnode = self.ImportUsd(nodename,filePath, excludePaths, primPath)
            if impnode:
                transNode = cmds.listRelatives(shape, p=True, f=True)[0]
                self.CopyNode(transNode, impnode)
                self.DeleteXform()
        print 'Success'


    def CopyNode(self, transNode, impnode):
        nodename = transNode.split('|')[-1]

        tmp= ReplaceToNewGroup(transNode)

        copynode = cmds.duplicate(impnode, n=nodename)[0]
        geomgroup = cmds.listRelatives(copynode, c=True, f=True)[0]
        xb = XformBlock(geomgroup)
        root =cmds.parent(geomgroup, tmp)[0]
        SetDefaultTransform(root)
        xb.Set(root)

        cmds.delete(transNode)
        cmds.delete(impnode)
        cmds.delete(copynode)

        cmds.rename(tmp,nodename)


    def CopyNode2(self,transNode,filename):
        xb = Represent.XformBlock(transNode)
        parentNode = cmds.listRelatives(transNode, p=True, f=True)
        if parentNode:
            parentNode = parentNode[0]
        else:
            parentNode = None

        nodename = filename.split('/')[-1].split('.')[0]
        locator = cmds.spaceLocator(n=nodename+'_loc')[0]
        locator = cmds.ls(locator, l=1)[0]

        root = cmds.parent(locator, parentNode)[0]
        SetDefaultTransform(root)
        xb.Set(root)
        cmds.delete(transNode)

    def CopyNode3(self,filename,matrix,transNode):
        nodename = filename.split('/')[-1].split('.')[0]
        copynode = cmds.spaceLocator(n=nodename + '_loc')[0]
        copynode = cmds.ls(copynode, l=1)[0]
        cmds.xform(copynode, m=matrix, ws=True)
        cmds.parent(copynode, transNode)



    def ImportUsdChild(self, refData, rootNode):
        for k, v in refData.items():
            filePath = v['filePath']
            excludePaths = ''
            primPath = ''
            if v.has_key('primPath'):
                primPath = v['primPath']
            if v.has_key('excludePrimPaths'):
                excludePaths = v['excludePrimPaths']

            impnode = self.ImportUsd(k,filePath, excludePaths, primPath)

            for node in v['nodes']:
                splitMayaPath = node.split('|')
                transNode = rootNode + '|' + '|'.join(splitMayaPath[2:])
                self.CopyNode2(transNode,filePath)

            self.SourceTempGrp(impnode)
