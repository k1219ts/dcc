#coding=utf-8
import os
import time
import json
import string
import numpy
import getpass
import pprint

import maya.api.OpenMaya as OpenMaya

# for alembic python
from imath import *
from alembic.AbcCoreAbstract import *
from alembic.Abc import *
from alembic.AbcGeom import *
from alembic.Util import *
kWrapExisting = WrapExistingFlag.kWrapExisting

import maya.cmds as cmds
import maya.mel as mel

import sgCommon
import sgComponent

def SetAttr( node, lnData ):
    for ln in lnData:
        if cmds.attributeQuery( ln, n=node, ex=True ):
            if lnData[ln]['type'] == 'string':
                cmds.setAttr( '%s.%s' % (node, ln), lnData[ln]['value'], type='string' )
            else:
                cmds.setAttr( '%s.%s' % (node, ln), lnData[ln]['value'] )

def AddAttr(node, lnData):
    """
    :param node: object name
    :param lnData: {attribute_name:{'value': xx, 'type': xx}, ...}
    :return:
    """
    for ln in lnData:
        typ = lnData[ln]['type']
        # add attribute
        if not cmds.attributeQuery(ln, n=node, ex=True):
            if typ == 'string':
                cmds.addAttr(node, ln=ln, dt='string')
            else:
                cmds.addAttr(node, ln=ln, at=typ)
        # set attribute
        if typ == 'string':
            cmds.setAttr('%s.%s' % (node, ln), lnData[ln]['value'], type='string')
        else:
            cmds.setAttr('%s.%s' % (node, ln), lnData[ln]['value'])


def getZgpuSource(*argv):
    """
    Get dxComponent(ZGpuMeshShape) Layout export node.
    :param argv[0]: selected object list
    :return: {root_group: [objects...], ...}
    """
    if argv and argv[0]:
        nodes = cmds.ls(argv[0], dag=True, l=True, type='dxComponent')
    else:
        nodes = cmds.ls(l=True, type='dxComponent')

    result = dict()
    for n in nodes:
        if cmds.getAttr('%s.action' % n) == 2:
            src = n.split('|')
            if not result.has_key(src[1]):
                result[src[1]] = list()
            result[src[1]].append(n)

    return result


def getZenvSource(*argv):
    """
    Get ZenvGroup child ZEnvPointSet.
    :param argv[0]: selected ZEnvGroup list
    :return: ZEnvPointSet list
    """
    if not cmds.pluginInfo('ZENVForMaya', q=True, l=True):
        return
    if argv and argv[0]:
        nodes = cmds.ls(argv[0], type='ZEnvGroup', dag = True)
    else:
        nodes = cmds.ls(type='ZEnvGroup', dag = True)

    result = list()
    for n in nodes:
        pointSets = cmds.ls(n, dag=True, l=True, type='ZEnvPointSet')
        if len(pointSets) > 1:
            mel.eval('print "## Assembly Export : zenv convention check!\\n"')
            # pass
        else:
            result.append(n)

    return result


def getBound(object):
    """
    Get Boundingbox
    :param object: selected object name
    :return: (xmin, xmax, ymin, ymax, zmin, zmax)
    """
    bmin = cmds.getAttr('%s.boundingBoxMin' % object)
    bmax = cmds.getAttr('%s.boundingBoxMax' % object)
    if bmin and bmax:
        return bmin[0][0], bmax[0][0], bmin[0][1], bmax[0][1], bmin[0][2], bmax[0][2]
    else:
        return None


class AssemblyExport:
    """
    ZEnv and dxComponent(ZGpuMeshShape) Layout export to Alembic Points(Assembly Format).
    need plugins : backstageMenu, ZENVForMaya
    """
    def __init__(self, filename, start, end):
        """
        export json and same directory group name based assembly(*.asb)
        :param filename: json file
        :param start: int(frame)
        :param end: int(frame)
        """
        # plugin setup
        sgCommon.pluginSetup(['backstageMenu'])

        # time
        tunit = cmds.currentUnit(t=True, q=True)
        self.m_fps = sgCommon._timeUnitMap[tunit]

        self.m_infofile = filename
        self.m_outPath = os.path.dirname(filename)
        self.m_start = int(start)
        self.m_end = int(end)

        self.info = dict()

    def doIt(self):
        '''
         Export Assembly files
        :return: Success : [ ], Fail : None
        '''

        if not os.path.exists(self.m_outPath):
            os.makedirs(self.m_outPath)

        startTime = time.time()

        selected = cmds.ls(sl=True)
        self.zgpu_source = getZgpuSource(selected)
        self.zenv_source = getZenvSource(selected)

        if self.zenv_source:
            for z in self.zenv_source:
                self.zenv_export(z)

        if self.zgpu_source:
            for g in self.zgpu_source:
                self.zgpu_export(g)

        if self.info:
            self.writeInfo()
            endTime = time.time()
            mel.eval('print "# AssemblyExport compute time : %s sec"' % (endTime-startTime))
            return self.info['asbfiles'], self.info['abcfiles']
        else:
            mel.eval('print "# AssemblyExport Error!"')
            return None, None

    def writeInfo(self):
        data = dict()
        data['InstanceSetup'] = self.info
        header = dict()
        header['created'] = time.asctime()
        header['author']  = getpass.getuser()
        header['context'] = cmds.file(q=True, sn=True)
        data['_Header'] = header
        f = open(self.m_infofile, 'w')
        json.dump(data, f, indent=4)
        f.close()

    # node - ZEnvPointSet
    def getInstanceSource_for_zenv(self, node):
        """
        ZENV instance source find.
        :param node: ZEnvPointSet name
        :return: files, objectpath, source matrixs
        """
        files = list()
        paths = list()
        matrixs = list()
        objectName_valid = 0

        sourceSets = cmds.listConnections('%s.inSourceSets' % node, sh=True)
        for s in sourceSets:
            sourceShape = cmds.listConnections(s, s=True, d=False, sh=True)
            if sourceShape:
                sourceShape = sourceShape[0]
                sourceTrans = cmds.listRelatives(sourceShape, p=True, f=True)[0]

                fileFormat = cmds.getAttr('%s.fileFormat' % sourceShape)
                objectName = cmds.getAttr('%s.objectName' % sourceShape)
                assetPath  = cmds.getAttr('%s.assetPath' % sourceShape)
                version    = cmds.getAttr('%s.version' % sourceShape)
                if fileFormat == 0:
                    fn = os.path.join(assetPath, 'model', '%s_model_%s.abc' % \
                                      (os.path.basename(assetPath), version))
                else:
                    fn = assetPath
                fn = fn.replace('/netapp/dexter/show/', '/show/')
                if os.path.exists(fn):
                    files.append(fn)
                    objpath = objectName.replace('_low', '')
                    paths.append(objpath)
                    matrixs.append(cmds.xform(sourceTrans, q=True, m=True))
                    if objectName:
                        objectName_valid += 1

                    # update info
                    # abcfiles
                    if not self.info.has_key('abcfiles'):
                        self.info['abcfiles'] = list()
                    if not fn in self.info['abcfiles']:
                        self.info['abcfiles'].append(fn)

                    # locations & boundingbox
                    if not self.info.has_key('locations'):
                        self.info['locations'] = list()
                    if not self.info.has_key('bounds'):
                        self.info['bounds'] = list()
                    bound = getBound(sourceShape)
                    location = os.path.splitext(os.path.basename(fn))[0] + objpath
                    if not location in self.info['locations']:
                        self.info['locations'].append(location)
                        self.info['bounds'].append(bound)

        if objectName_valid == 0:
            return None
        return files, paths, matrixs


    # Export Zenv to ASB
    # node - ZEnvGroup
    def zenv_export(self, zenvgroupnode):
        pointSetNode = cmds.ls(zenvgroupnode, dag=True, l=True, type='ZEnvPointSet')[0]
        source_data = self.getInstanceSource_for_zenv(pointSetNode)
        if not source_data:
            return None

        # debug
        print '# zenv export node : %s -> %s' % (zenvgroupnode, pointSetNode)

        self.src_files   = source_data[0]
        self.src_paths   = source_data[1]
        self.src_matrixs = source_data[2]

        filename = os.path.join(self.m_outPath, '%s.asb' % zenvgroupnode)
        oarch = OArchive(str(filename), asOgawa=True)
        root  = oarch.getTop()
        obj   = OPoints(root, str(zenvgroupnode))
        schema= obj.getSchema()
        arb   = schema.getArbGeomParams()

        self.setZenvConstantData(arb, zenvgroupnode)
        self.setZenvSampleData(pointSetNode, schema, arb)

        print '#\tzenv export file : %s' % filename
        # update info
        if not self.info.has_key('asbfiles'):
            self.info['asbfiles'] = list()
        self.info['asbfiles'].append(filename)


    def setZenvConstantData(self, ArbGeomParams, rootnode):
        d_size = len(self.src_files)

        fileVal = StringArray(d_size)
        pathVal = StringArray(d_size)

        # set value
        for i in range(d_size):
            fileVal[i] = str(self.src_files[i])
            if self.src_paths:
                pathVal[i] = str('/%s%s' % (rootnode, self.src_paths[i]))

        # arcfiles
        fileSamp = OStringGeomParamSample()
        fileSamp.setScope(GeometryScope.kConstantScope)
        fileSamp.setVals(fileVal)
        arcfilesProp = OStringGeomParam(
            ArbGeomParams, 'arcfiles', False, GeometryScope.kConstantScope, 1
        )
        arcfilesProp.set(fileSamp)

        # arcpath
        if self.src_paths:
            pathSamp = OStringGeomParamSample()
            pathSamp.setScope(GeometryScope.kConstantScope)
            pathSamp.setVals(pathVal)
            arcpathProp = OStringGeomParam(
                ArbGeomParams, 'arcpath', False, GeometryScope.kConstantScope, 1
            )
            arcpathProp.set(pathSamp)


    def setZenvSampleData(self, node, PointSchema, ArbGeomParams):
        count = cmds.getAttr('%s.count' % node)

        positions = V3fArray(count)
        ids       = IntArray(count)
        scales    = FloatArray(count*3)
        orients   = QuatfArray(count)
        rtps      = IntArray(count)
        inst      = IntArray(count)
        mtxs      = FloatArray(count*16)

        # source index
        sids = cmds.ZEnvPointInfoCmd(nodeName=node, attribute='sid')
        # get 4x4 matrix
        matrixs = cmds.ZEnvPointInfoCmd(nodeName=node, attribute='tm')

        for i in xrange(count):
            ids[i] = i
            rtp = int(sids[i])
            rtps[i] = rtp

            inst[i] = 1

            z_mtx = OpenMaya.MMatrix(matrixs[i*16:i*16+16])
            src_mtx = OpenMaya.MMatrix(self.src_matrixs[rtp])
            mtx = src_mtx * z_mtx

            for x in xrange(16):
                mtxs[i*16+x] = mtx[x]

            transMtx = OpenMaya.MTransformationMatrix(mtx)
            # translate
            tr = transMtx.translation(OpenMaya.MSpace.kWorld)
            positions[i] = V3f(tr.x, tr.y, tr.z)
            # scale
            sc = transMtx.scale(OpenMaya.MSpace.kWorld)
            scales[i*3] = sc[0]
            scales[i*3+1] = sc[1]
            scales[i*3+2] = sc[2]
            # rotate
            ro = transMtx.rotation(asQuaternion=True)
            orients[i] = Quatf(ro.x, ro.y, ro.z, ro.w)

        # time sample
        ts = TimeSampling(1.0/24.0, 1/24.0)

        psamp = OPointsSchemaSample()
        psamp.setPositions(positions)
        psamp.setIds(ids)
        PointSchema.set(psamp)

        # arbGeomParams
        #   scale
        scaleSamp = OFloatGeomParamSample()
        scaleSamp.setScope(GeometryScope.kVertexScope)
        scaleSamp.setVals(scales)
        ScaleGeomParam = OFloatGeomParam(
            ArbGeomParams, 'scale', False, GeometryScope.kVaryingScope, 3, ts
        )
        ScaleGeomParam.set(scaleSamp)
        #   orient
        orientSamp = OQuatfGeomParamSample()
        orientSamp.setScope(GeometryScope.kVertexScope)
        orientSamp.setVals(orients)
        OrientGeomParam = OQuatfGeomParam(
            ArbGeomParams, 'orient', False, GeometryScope.kVaryingScope, 1, ts
        )
        OrientGeomParam.set(orientSamp)
        #   rtp
        rtpsSamp = OInt32GeomParamSample()
        rtpsSamp.setScope(GeometryScope.kVertexScope)
        rtpsSamp.setVals(rtps)
        rtpsGeomParam = OInt32GeomParam(
            ArbGeomParams, 'rtp', False, GeometryScope.kVaryingScope, 1
        )
        rtpsGeomParam.set(rtpsSamp)

        # inst
        instSamp = OInt32GeomParamSample()
        instSamp.setScope(GeometryScope.kVertexScope)
        instSamp.setVals(inst)
        instGeomParam = OInt32GeomParam(
            ArbGeomParams, 'inst', False, GeometryScope.kVaryingScope, 1
        )
        instGeomParam.set(instSamp)

        #   matrix
        mtxsSamp = OFloatGeomParamSample()
        mtxsSamp.setScope(GeometryScope.kVertexScope)
        mtxsSamp.setVals(mtxs)
        MtxGeomParam = OFloatGeomParam(
            ArbGeomParams, 'mtx', False, GeometryScope.kVaryingScope, 16, ts
        )
        MtxGeomParam.set(mtxsSamp)



    # Export ZGpuMeshShape layout to ASB
    def getZgpuFile(self, node):
        """
        Get render alembic file.
        :param node: dxComponent
        :return: string
        """
        fn = cmds.getAttr('%s.renderFile' % node)
        if not fn:
            fn = cmds.getAttr('%s.abcFileName' % node)
        return fn.replace('/netapp/dexter/show/', '/show/')

    def getInstanceSource_for_zgpu(self, node):
        """
        Get Instance source file and update info
        :param node: root_group
        :return: file list
        """
        result = list()
        for n in self.zgpu_source[node]:
            fn = self.getZgpuFile(n)
            result.append(fn)

            # update info
            # abcfiles
            if not self.info.has_key('abcfiles'):
                self.info['abcfiles'] = list()
            if not fn in self.info['abcfiles']:
                self.info['abcfiles'].append(fn)

            # locations & boundingbox
            if not self.info.has_key('locations'):
                self.info['locations'] = list()
            if not self.info.has_key('bounds'):
                self.info['bounds'] = list()

            zgpuMeshShapeList = cmds.ls(n, dag=True, type='ZGpuMeshShape')
            bound = None
            if len(zgpuMeshShapeList) > 0:
                bound = getBound(zgpuMeshShapeList[0])

            location = os.path.splitext(os.path.basename(fn))[0]
            if not location in self.info['locations']:
                self.info['locations'].append(location)
                if bound != None:
                    self.info['bounds'].append(bound)

        return list(set(result))

    def getRenderManExtraAttributes(self, node):
        result = dict()
        src = node.split('|')
        for i in range(1, len(src)):
            n = string.join(src[:i+1], '|')
            attrs = cmds.listAttr(n, st='rman__riattr__*')
            if attrs:
                for ln in attrs:
                    gv = cmds.getAttr('%s.%s' % (n, ln))
                    gt = cmds.getAttr('%s.%s' % (n, ln), type=True)
                    result[ln] = {'value': gv, 'type': gt}
        return result


    def zgpu_export(self, rootgroupnode):
        """
        Export dxComponent layout group
        :param rootgroupnode:
        :return:
        """
        self.src_files = self.getInstanceSource_for_zgpu(rootgroupnode)
        if not self.src_files:
            return

        # debug
        print '# zgpu export node : %s' % rootgroupnode
        filename = os.path.join(self.m_outPath, '%s.asb' % rootgroupnode)

        # time sample
        self.timeSamp = TimeSampling(1.0/self.m_fps, self.m_start/self.m_fps)
        oarch = OArchive(str(filename), asOgawa=True)
        root  = oarch.getTop()
        obj   = OPoints(root, str(rootgroupnode), self.timeSamp)
        schema= obj.getSchema()
        arb   = schema.getArbGeomParams()

        self.setZgpuConstantData(arb, rootgroupnode)
        self.setZgpuSampleData(rootgroupnode, schema, arb)

        print '#\tzgpu export file : %s' % filename
        # update info
        if not self.info.has_key('asbfiles'):
            self.info['asbfiles'] = list()
        self.info['asbfiles'].append(filename)


    def setZgpuConstantData(self, ArbGeomParams, node):
        """
        write constant data
        :param ArbGeomParams: OCompoundProperty
        :param node:
        :return:
        """
        # arcfiles
        fileVal = StringArray(len(self.src_files))
        for i in range(len(self.src_files)):
            fileVal[i] = str(self.src_files[i])
        fileSamp = OStringGeomParamSample()
        fileSamp.setScope(GeometryScope.kConstantScope)
        fileSamp.setVals(fileVal)
        arcfilesProp = OStringGeomParam(
            ArbGeomParams, 'arcfiles', False, GeometryScope.kConstantScope, 1
        )
        arcfilesProp.set(fileSamp)

        nodes = self.zgpu_source[node]
        n_size = len(nodes)
        objpaths = StringArray(n_size)
        rtps     = IntArray(n_size)
        inst     = IntArray(n_size)

        # RMAN
        attrVals = list()
        attrNames = list()
        attrTypes = list()

        for i in xrange(n_size):
            # object path
            objpaths[i] = str(nodes[i].replace('|', '/'))
            # render type
            rt_index = 0
            arcfn = self.getZgpuFile(nodes[i])
            if arcfn:
                rt_index = self.src_files.index(arcfn)
            rtps[i] = rt_index
            # enable instance
            inst_val = cmds.getAttr('%s.objectInstance' % nodes[i])
            inst[i] = inst_val

            # RenderMan Extra Attributes
            attrs = self.getRenderManExtraAttributes(nodes[i])
            attrVals.append(attrs)
            if attrs:
                for ln in attrs.keys():
                    if not ln in attrNames:
                        attrNames.append(ln)
                        attrTypes.append(attrs[ln]['type'])


        # objpaths
        pathSamp = OStringGeomParamSample()
        pathSamp.setScope(GeometryScope.kVertexScope)
        pathSamp.setVals(objpaths)
        objpathGeomParam = OStringGeomParam(
            ArbGeomParams, 'objpath', False, GeometryScope.kVaryingScope, 1
        )
        objpathGeomParam.set(pathSamp)

        # rtp
        rtpSamp = OInt32GeomParamSample()
        rtpSamp.setScope(GeometryScope.kVertexScope)
        rtpSamp.setVals(rtps)
        rtpsGeomParam = OInt32GeomParam(
            ArbGeomParams, 'rtp', False, GeometryScope.kVaryingScope, 1
        )
        rtpsGeomParam.set(rtpSamp)

        # inst
        instSamp = OInt32GeomParamSample()
        instSamp.setScope(GeometryScope.kVertexScope)
        instSamp.setVals(inst)
        instGeomParam = OInt32GeomParam(
            ArbGeomParams, 'inst', False, GeometryScope.kVaryingScope, 1
        )
        instGeomParam.set(instSamp)

        # RMAN
        if not attrNames:
            return
        attrData = dict()
        ln_size = len(attrNames)
        for i in range(ln_size):
            gt = attrTypes[i]
            if gt == 'long':
                attrData[attrNames[i]] = IntArray(n_size)
        for i in xrange(n_size):
            for ln in attrNames:
                if attrVals[i].has_key(ln):
                    attrData[ln][i] = attrVals[i][ln]['value']
                else:
                    attrData[ln][i] = -1
        for i in range(ln_size):
            gt = attrTypes[i]
            if gt == 'long':
                tmpSamp = OInt32GeomParamSample()
                tmpSamp.setScope(GeometryScope.kVertexScope)
                tmpSamp.setVals(attrData[attrNames[i]])
                tmpGeomParam = OInt32GeomParam(
                    ArbGeomParams, str(attrNames[i]), False, GeometryScope.kVaryingScope, 1
                )
                tmpGeomParam.set(tmpSamp)


    def setZgpuSampleData(self, node, PointSchema, ArbGeomParams):
        """
        write frame data
        :param node:
        :param PointSchema: OPointsSchema
        :param ArbGeomParams: OCompoundProperty
        :return:
        """
        matrixGeomParam = OFloatGeomParam(
            ArbGeomParams, 'mtx', False, GeometryScope.kVaryingScope, 16, self.timeSamp
        )
        scaleGeomParam = OFloatGeomParam(
            ArbGeomParams, 'scale', False, GeometryScope.kVaryingScope, 3, self.timeSamp
        )
        orientGeomParam = OQuatfGeomParam(
            ArbGeomParams, 'orient', False, GeometryScope.kVaryingScope, 1, self.timeSamp
        )
        for f in xrange(self.m_start, self.m_end+1):
            ps, mtx, scale, orient = self.getFrameData(node, f)
            PointSchema.set(ps)
            matrixGeomParam.set(mtx)
            scaleGeomParam.set(scale)
            orientGeomParam.set(orient)

    def getFrameData(self, node, frame):
        """
        :param node: root_group name
        :param frame:
        :return: OPointsSchemaSample, OFloatGeomParamSample(matrix),
                 OFloatGeomParamSample(scale), OQuatfGeomParamSample(orient)
        """
        nodes = self.zgpu_source[node]
        n_size = len(nodes)

        positions = V3fArray(n_size)
        ids       = IntArray(n_size)
        mtxs      = FloatArray(n_size * 16)
        scales    = FloatArray(n_size * 3)
        orients   = QuatfArray(n_size)
        for i in xrange(n_size):
            ids[i] = i

            # get 4x4 matrix
            d_mtx, d_frs = sgCommon.getMtx(nodes[i], frame, frame, 1)
            mtx = OpenMaya.MMatrix(d_mtx[0])
            transMtx = OpenMaya.MTransformationMatrix(mtx)

            for x in xrange(16):
                mtxs[i*16+x] = d_mtx[0][x]

            # translate
            tr = transMtx.translation(OpenMaya.MSpace.kWorld)
            positions[i] = V3f(tr.x, tr.y, tr.z)
            # scale
            sc = transMtx.scale(OpenMaya.MSpace.kWorld)
            for x in xrange(3):
                scales[i*3+x] = sc[x]
            # rotate
            ro = transMtx.rotation(asQuaternion=True)
            orients[i] = Quatf(ro.x, ro.y, ro.z, ro.w)

        # default OPoints data(positions, ids)
        schemaSamp = OPointsSchemaSample()
        schemaSamp.setPositions(positions)
        schemaSamp.setIds(ids)
        # arbGeomParam - matrix(mtxs)
        matrixSamp = OFloatGeomParamSample()
        matrixSamp.setScope(GeometryScope.kVertexScope)
        matrixSamp.setVals(mtxs)
        # arbGeomParam - scale
        scaleSamp = OFloatGeomParamSample()
        scaleSamp.setScope(GeometryScope.kVertexScope)
        scaleSamp.setVals(scales)
        # arbGeomParam - orient
        orientSamp = OQuatfGeomParamSample()
        orientSamp.setScope(GeometryScope.kVertexScope)
        orientSamp.setVals(orients)

        return schemaSamp, matrixSamp, scaleSamp, orientSamp

# def exportDialog():
#     fn = cmds.fileDialog2(
#         fileMode=0,
#         caption='ZENV and LAYOUT export to Assembly',
#         okCaption='export',
#         fileFilter='JSON (*.json)'
#     )
#     if not fn:
#         return
#     asbClass = AssemblyExport(fn[0], 1, 1)
#     asbClass.doIt()
#



class IPointsMatrixSet:
    """
        Alembic IPoints P, scale, orient -> maya matrix set
    """
    def __init__(self, Prop, PointIds, ObjectPath, SourcePath):
        tunit = cmds.currentUnit(t=True, q=True)
        self.m_fps = sgCommon._timeUnitMap[tunit]

        self.m_objpaths = ObjectPath
        self.m_rtp = PointIds
        self.m_sourcePath = SourcePath

        self.m_pointsProp = Prop.getProperty('P')
        geomProp = Prop.getProperty('.arbGeomParams')
        self.m_scaleProp = geomProp.getProperty('scale')
        self.m_orientProp = geomProp.getProperty('orient')

        self.doIt()


    def doIt(self):
        self.getTimes()

        p_const = self.m_pointsProp.isConstant()
        s_const = self.m_scaleProp.isConstant()
        o_const = self.m_orientProp.isConstant()

        self.m_matrixs = list()
        for i in xrange(len(self.m_rtp)):
            self.m_matrixs.append(list())
        self.m_frames = list()

        # not animation
        if p_const and s_const and o_const:
            self.m_P = self.m_pointsProp.getValue()
            self.m_Scale = self.m_scaleProp.getValue()
            self.m_Orient = self.m_orientProp.getValue()
            self.m_frames.append(1)
            self.getMatrixData()
        # animation
        else:
            for f in numpy.arange(self.m_startFrame, self.m_endFrame-1):
                self.m_frames.append(f)
                sampleSelector = ISampleSelector(float(f)/self.m_fps)
                self.m_P = self.m_pointsProp.getValue(sampleSelector)
                self.m_Scale = self.m_scaleProp.getValue(sampleSelector)
                self.m_Orient = self.m_orientProp.getValue(sampleSelector)
                self.getMatrixData()

        for i in xrange(len(self.m_rtp)):
            if self.m_sourcePath == 'objpath':
                node = self.m_objpaths[i].replace('/', '|')
            else:
                node = self.m_objpaths[self.m_rtp[i]].replace('/', '|') + str(i)

            sgCommon.setMtx(node, self.m_matrixs[i], self.m_frames)



    def getTimes(self):
        ts = self.m_pointsProp.getTimeSampling()
        numsamps = self.m_pointsProp.getNumSamples()
        self.m_startFrame = ts.getSampleTime(0) * self.m_fps
        self.m_endFrame   = ts.getSampleTime(numsamps-1) * self.m_fps


    def getMatrixData(self):
        for i in xrange(len(self.m_rtp)):
            trMtx = OpenMaya.MTransformationMatrix()

            trMtx.setScale(self.m_Scale[i*3:i*3+3], OpenMaya.MSpace.kWorld)

            orient = self.m_Orient[i]
            orient = [orient.r()] + list(orient.v())
            quat = OpenMaya.MQuaternion(orient)
            rotate = quat.asEulerRotation()
            trMtx.setRotation(rotate)

            trMtx.setTranslation(OpenMaya.MVector(self.m_P[i]), OpenMaya.MSpace.kWorld)

            self.m_matrixs[i].append(trMtx.asMatrix())



class AssemblyImport:
    """
        AssemblyImport for maya based layout.
        need plugins : backstageMenu, ZMayaTools, AbcImport
    """
    def __init__(self, filename):
        # plug-in setup
        sgCommon.pluginSetup(['AbcImport', 'backstageMenu', 'ZMayaTools'])

        tunit = cmds.currentUnit(t=True, q=True)
        self.m_fps = sgCommon._timeUnitMap[tunit]

        self.m_filename = filename
        self.m_createdMap = dict()
        self.m_root = None

    def doIt(self):
        iarch = IArchive(str(self.m_filename))
        root  = iarch.getTop()
        self.visitObject(root)

    def visitObject(self, iobj):
        for obj in iobj.children:
            ohead = obj.getHeader()
            if IPoints.matches(ohead):
                self.visitPoints(obj)
            self.visitObject(obj)


    def visitPoints(self, iobj):
        icompoundProp = iobj.getProperties()
        prop = icompoundProp.getProperty(0)

        icompoundGeomProp = prop.getProperty('.arbGeomParams')
        geomHeaders = list()
        for i in icompoundGeomProp.propertyheaders:
            geomHeaders.append(i.getName())

        self.sourcePath = 'objpath'

        if not 'objpath' in geomHeaders:
            self.sourcePath = 'arcpath'
            # mel.eval('print "# AssemblyImport Error : not support file, using < ZAssemblyArchive >"')
            # return None

        mel.eval('print "# AssemblyImport : %s"' % self.m_filename)

        # .pointsIds
        idProp = prop.getProperty('.pointIds')
        self.m_ids = idProp.getValue()
        # arcfiles
        self.m_arcfiles = list(icompoundGeomProp.getProperty('arcfiles').getValue())
        # objpath
        self.m_objpaths = list(icompoundGeomProp.getProperty(self.sourcePath).getValue())
        # rtp
        self.m_rtp = list(icompoundGeomProp.getProperty('rtp').getValue())

        # extra attribute
        self.m_extAttrs = dict()
        extraAttributes = ['rman__riattr__user_group_id',
                           'rman__riattr__user_object_id']
        for ln in extraAttributes:
            if ln in geomHeaders:
                attrProp = icompoundGeomProp.getProperty(ln)
                self.m_extAttrs[ln] = list(attrProp.getValue())

        self.createNode()
        # set matrix
        IPointsMatrixSet(prop, self.m_rtp, self.m_objpaths, self.sourcePath)


    def createNode(self):
        """ Create node and instance setup.\n call createTreeNode """
        for i in xrange(len(self.m_ids)):
            self.createTreeNode(i)

        # instance setup
        for c in self.m_createdMap.values():
            zmeshes = cmds.listConnections(c, type='ZGpuMeshShape', s=False, d=True)
            if len(zmeshes) > 1:
                for i in zmeshes:
                    cmds.setAttr('%s.objectInstance' % i, 1)

    def createTreeNode(self, pointIndex):
        """ Create node : dxAssembly, group, dxComponent.\n call sgComponent. """
        if self.sourcePath == 'objpath':
            objpath = self.m_objpaths[pointIndex].replace('/', '|')
        else:
            objpath = self.m_objpaths[self.m_rtp[pointIndex]].replace('/', '|') + str(pointIndex)
        src = objpath.split('|')

        # dxAssembly
        if not cmds.objExists(src[1]):
            node = cmds.createNode('dxAssembly', n=src[1])
            cmds.setAttr('%s.fileName' % node, self.m_filename, type='string')
            # renderman attribute
            sgCommon.setInVisAttribute(node, 1)
            self.m_root = node

        # tree node
        for i in range(1, len(src)-1):
            path = string.join(src[:i+1], '|')
            if not cmds.objExists(path):
                if i == 1:
                    cmds.group(n=src[i], em=True)
                else:
                    pn = string.join(src[:i], '|')
                    cmds.group(n=src[i], p=pn, em=True)

        # dxComponent node
        if not cmds.objExists(objpath):
            parentNode = string.join(src[:-1], '|')

            rtp = self.m_rtp[pointIndex]
            arcfn = self.m_arcfiles[rtp]
            creator = None
            if self.m_createdMap.has_key(arcfn):
                creator = self.m_createdMap[arcfn]
            else:
                for creatorNode in cmds.ls(type = 'ZGpuMeshCreator'):
                    if cmds.getAttr('%s.file' % creatorNode).replace('_low', '') == arcfn:
                        creator = creatorNode
                        break

            # print arcfn, objpath

            cmds.createNode('dxComponent', n=src[-1], p=parentNode)
            attrMap = {'action':2, 'mode':1, 'display':3}
            for at in attrMap:
                cmds.setAttr('%s.%s' % (objpath, at), attrMap[at])
            cmds.setAttr('%s.abcFileName' % objpath, arcfn, type='string')
            # archive
            cpClass = sgComponent.Archive(objpath)
            cpClass.m_creator = creator
            cpClass.m_animation = False
            cpClass.doIt()

            self.AddExtraAttributes(objpath, pointIndex)

            if not self.m_createdMap.has_key(arcfn):
                self.m_createdMap[arcfn] = cpClass.m_creator


    def AddExtraAttributes(self, nodeName, pointIndex):
        if not self.m_extAttrs:
            return

        # extra attributes
        _typeMap = {'int': 'long'}
        attrData = dict()
        for ln in self.m_extAttrs:
            _data = dict()
            value = self.m_extAttrs[ln][pointIndex]
            vtype = _typeMap[type(value).__name__]
            if vtype != 'string':
                if value > -1:
                    _data['value'] = value
                    _data['type'] = vtype
            else:
                _data['value'] = value
                _data['type'] = 'string'
            if _data:
                attrData[ln] = _data
        if attrData:
            AddAttr(nodeName, attrData)


def importAssemblyFile(fileName):
    asbClass = AssemblyImport(fileName)
    asbClass.doIt()
    return asbClass.m_root



class PointsRdbExport:
    '''
    Alembic Points export for Destruction Key Animation data
    '''
    def __init__(self, filename, node, start, end):
        self.m_meshFile = os.path.splitext(filename)[0] + '_mesh.abc'
        self.m_pointFile = os.path.splitext(filename)[0] + '.asb'
        self.m_node = cmds.ls(node, l=True)[0]
        self.m_start = start
        self.m_end = end

        self.doIt()

    def doIt(self):
        self.RootNode = None
        self.ShapeNodes = None
        self.getObjects()
        if not self.RootNode:
            return

        startTime = time.time()
        self.clearFile()

        self.Selection = OpenMaya.MSelectionList()
        for n in self.ShapeNodes:
            self.Selection.add(n)
        self.Size = int(self.Selection.length())

        self.exportMesh()
        self.exportPoints()

        endTime = time.time()
        # Debug
        print '# Export Debug'
        print '    : meshfile -> %s' % self.m_meshFile
        print '    : pointfile -> %s' % self.m_pointFile
        print '  compute %.3f sec' % (endTime-startTime)


    def getObjects(self):
        if cmds.nodeType(self.m_node) != 'dxComponent':
            return
        action = cmds.getAttr('%s.action' % self.m_node) # 4 : RDB Export
        mode = cmds.getAttr('%s.mode' % self.m_node) # 0 : Mesh
        if action == 4 and mode == 0:
            child = cmds.listRelatives(self.m_node, f=True)[0]
            if child.split('|')[-1] == self.m_node.split('|')[-1] + 'Arc':
                self.RootNode = cmds.listRelatives(child, f=True)[0]
            else:
                self.RootNode = child
            self.RemoveName = cmds.listRelatives(self.RootNode, p=True, f=True)[0]
            self.ShapeNodes = cmds.ls(self.RootNode, dag=True, type='surfaceShape')

    def clearFile(self):
        if os.path.exists(self.m_meshFile):
            os.remove(self.m_meshFile)
        if os.path.exists(self.m_pointFile):
            os.remove(self.m_pointFile)


    def exportMesh(self):
        opts  = '-uv -wv -wuvs -ef -a MaterialSet -atp rman -df ogawa -ws -fr 1 1'
        opts += ' -rt %s' % self.RootNode
        opts += ' -file %s' % self.m_meshFile
        cmds.AbcExport(j=opts, v=True)

    def exportPoints(self):
        # time
        tunit = cmds.currentUnit(t=True, q=True)
        self.m_fps = sgCommon._timeUnitMap[tunit]

        timeSamp = TimeSampling(1.0/self.m_fps, self.m_start/self.m_fps)

        oarch = OArchive(str(self.m_pointFile), asOgawa=True)
        root = oarch.getTop()
        obj = OPoints(root, str(self.RootNode.split('|')[-1]), timeSamp)
        schema = obj.getSchema()
        arb = schema.getArbGeomParams()
        scaleParam = OFloatGeomParam(
            arb, 'scale', False, GeometryScope.kVaryingScope, 3, timeSamp
        )
        orientParam = OQuatfGeomParam(
            arb, 'orient', False, GeometryScope.kVaryingScope, 1, timeSamp
        )
        pivotParam = OFloatGeomParam(
            arb, 'pivot', False, GeometryScope.kVaryingScope, 3, timeSamp
        )
        pathParam = OStringGeomParam(
            arb, 'path', False, GeometryScope.kVaryingScope, 1
        )

        self.rest_pos, self.rest_scale, self.rest_orient, self.rest_pivot = self.getFrameData(1)
        # print rest_pos

        for f in xrange(self.m_start, self.m_end+1):
            self.setPointsFrame(
                schema, scaleParam, orientParam, pivotParam, pathParam, f
            )

        meshVal = StringArray(1)
        meshVal[0] = str(self.m_meshFile)
        meshSamp = OStringGeomParamSample()
        meshSamp.setScope(GeometryScope.kConstantScope)
        meshSamp.setVals(meshVal)
        meshParam = OStringGeomParam(
            arb, 'meshfile', False, GeometryScope.kConstantScope, 1
        )
        meshParam.set(meshSamp)


    def setPointsFrame(self, PointSchema, ScaleParam, OrientParam, PivotParam, PathParam, frame):
        positions = V3fArray(self.Size)
        ids       = IntArray(self.Size)
        scales    = FloatArray(self.Size * 3)
        orients   = QuatfArray(self.Size)
        pivots    = FloatArray(self.Size * 3)
        paths     = StringArray(self.Size)

        i = 0
        tr, sc, ro, pv = self.getFrameData(frame)
        itSelection = OpenMaya.MItSelectionList(self.Selection)
        while not itSelection.isDone():
            mdag = itSelection.getDagPath()
            name = mdag.fullPathName().replace(self.RemoveName, '')
            name = str(name.replace('|', '/'))
            paths[i] = name

            # payload position
            positions[i] = V3f(tr[i].x, tr[i].y, tr[i].z)
            # payload orient
            orients[i] = Quatf(ro[i].x, ro[i].y, ro[i].z, ro[i].w)
            # payload scale, pivot
            for x in xrange(3):
                scales[i*3 + x] = sc[i][x]
                # pivots[i*3 + x] = pv[i][x]
                pivots[i*3 + x] = self.rest_pos[i][x]
            # payload id
            ids[i] = i

            i += 1
            itSelection.next()

        schemaSamp = OPointsSchemaSample()
        schemaSamp.setPositions(positions)
        schemaSamp.setIds(ids)

        scalesSamp = OFloatGeomParamSample()
        scalesSamp.setScope(GeometryScope.kVertexScope)
        scalesSamp.setVals(scales)

        orientsSamp = OQuatfGeomParamSample()
        orientsSamp.setScope(GeometryScope.kVertexScope)
        orientsSamp.setVals(orients)

        pivotsSamp = OFloatGeomParamSample()
        pivotsSamp.setScope(GeometryScope.kVertexScope)
        pivotsSamp.setVals(pivots)

        pathsSamp = OStringGeomParamSample()
        pathsSamp.setScope(GeometryScope.kVertexScope)
        pathsSamp.setVals(paths)

        PointSchema.set(schemaSamp)
        ScaleParam.set(scalesSamp)
        OrientParam.set(orientsSamp)
        PivotParam.set(pivotsSamp)
        PathParam.set(pathsSamp)


    def getFrameData(self, frame):
        positions = list()
        scales = list()
        orients = list()
        pivots = list()

        i = 0
        itSelection = OpenMaya.MItSelectionList(self.Selection)
        while not itSelection.isDone():
            mdag = itSelection.getDagPath()
            mobj = mdag.transform()
            mfn  = OpenMaya.MFnDependencyNode(mobj)

            frameCtx = OpenMaya.MDGContext(
                OpenMaya.MTime(frame, OpenMaya.MTime.uiUnit())
            )

            mtxAttr = mfn.attribute('worldMatrix')
            mtxPlug = OpenMaya.MPlug(mobj, mtxAttr)
            mtxPlug = mtxPlug.elementByLogicalIndex(0)
            mtxObj = mtxPlug.asMObject(frameCtx)
            mtxData = OpenMaya.MFnMatrixData(mtxObj)

            transMtx = mtxData.transformation()
            # get translation
            tr = transMtx.translation(OpenMaya.MSpace.kWorld)
            # get scale
            sc = transMtx.scale(OpenMaya.MSpace.kWorld)
            # get rotation
            ro = transMtx.rotation(asQuaternion=True)

            pivotAttr = mfn.attribute('scalePivot')
            pivotPlug = OpenMaya.MPlug(mobj, pivotAttr)
            pv = list()
            for x in xrange(3):
                pval = pivotPlug.child(x).asDouble(frameCtx)
                pv.append(pval)

            positions.append(tr + OpenMaya.MVector(*pv) * mtxData.matrix())
            scales.append(sc)
            orients.append(ro)
            pivots.append(OpenMaya.MVector(*pv))

            i += 1
            itSelection.next()

        return positions, scales, orients, pivots
