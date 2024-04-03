#coding:utf-8
import maya.standalone
maya.standalone.initialize()

import maya.cmds as cmds
import maya.mel as mel
import os, sys, re, logging, string, datetime, getpass, json

if not cmds.pluginInfo("pxrUsd", q=True, l=True):
    cmds.loadPlugin('pxrUsd')

if not cmds.pluginInfo("DXUSD_Maya", q=True, l=True):
    cmds.loadPlugin('DXUSD_Maya')

import DXUSD.Utils as utl
import DXUSD_MAYA.MUtils as mutl
import DXRulebook.Interface as rb

snPath = sys.argv[1]
user = getpass.getuser()
try: user = sys.argv[2]
except: pass


def GetViz(node):
    '''
    :node - full path string
    '''
    viz = True
    source = node.split('|')
    for i in range(1, len(source)):
        path = string.join(source[:i+1], '|')
        if cmds.listConnections('%s.visibility' % path):
            vals = cmds.keyframe(path, at='visibility', q=True, vc=True)
            if vals and not 1.0 in vals:
                return False
        else:
            viz = cmds.getAttr('%s.visibility' % path)
            if not viz:
                return viz
        connects = cmds.listConnections('%s.drawOverride' % path, type='displayLayer')
        if connects:
            for c in connects:
                viz = cmds.getAttr('%s.visibility' % c)
                if not viz:
                    return viz
    return viz

def getRigReferenced():
    referList = []
    for i in cmds.ls(type='dxRig'):
        if cmds.referenceQuery(i, inr = True):
            referenceFile = cmds.referenceQuery(i, filename=True)
            referenceFile = referenceFile.split("{")[0]
        else:
            referenceFile = ""
        referList.append(
            [i, cmds.getAttr('%s.action' % i), GetViz(cmds.ls(i, l=True)[0]), referenceFile]
        )
    return referList

def getLayout():
    layout = []
    # pxrUsdReferenceAssembly
    for n in cmds.ls('|*', type='pxrUsdReferenceAssembly'):
        layout.append([n, 1, GetViz(cmds.ls(n, l=True)[0]), 'extra'])
    # pxrUsdProxyShape
    for n in cmds.ls('|*|*', type='pxrUsdProxyShape'):
        trans = cmds.listRelatives(n, p=True)[0]
        layout.append([trans, 1, GetViz(cmds.ls(trans, l=True)[0]), 'extra'])
    # dxBlock
    for n in cmds.ls('|*', type='dxBlock'):
        if cmds.getAttr('%s.type' % n) == 1 or cmds.getAttr('%s.type' % n) == 2:
            layout.append([n, cmds.getAttr('%s.action' % n), GetViz(cmds.ls(n, l=True)[0]), 'dxBlock'])
    return layout


def getCameras():
    excludeCamera = list()
    cams = []
    isRenderable = 0
    # dxCamera
    for node in cmds.ls(type="dxCamera"):
        cams.append([node, cmds.getAttr('%s.action' % node), GetViz(cmds.ls(node, l=True)[0])])
        for cShape in cmds.ls(node, dag=True, type='camera', l=True):
            if cmds.getAttr('%s.renderable' % cShape) == True:
                isRenderable += 1
            excludeCamera.append(cShape)

    if isRenderable == 0:
        return list()
    return cams


def getCrowdInfo():
    result = []
    if cmds.pluginInfo('MiarmyProForMaya2018', q=True, l=True) and cmds.ls(type = "McdGlobal"):
        result.append(["crowd", 1, 1])
    elif cmds.pluginInfo('glmCrowd', q=True, l=True) and cmds.ls(type='SimulationCacheProxy'):
        for n in cmds.ls(type='SimulationCacheProxy'):
            enable = cmds.getAttr('%s.enable' % n)
            crowdFields = str(cmds.getAttr('%s.crowdFields' % n)).split(';')
            if isinstance(crowdFields, list):
                for cf in crowdFields:
                    result.append(['%s:%s' % (n, cf), 1, enable, 'golaem'])
            else:
                result.append(['%s:%s' % (n, crowdFields), 1, enable, 'golaem'])
    return result


def getSimulation():
    simNodes = []
    for n in cmds.ls(type='dxBlock'):
        if cmds.getAttr("%s.type" % n) == 3: # and cmds.getAttr("%s.action" % n) == 1:
            nsLayer = cmds.getAttr("%s.nsLayer" % n)
            nodeName = n
            if not ":" in n:
                nodeName = "%s:%s" % (nsLayer, n)
            simNodes.append(
                [nodeName, cmds.getAttr("%s.action" % n), GetViz(cmds.ls(n, l=True)[0])]
            )
    return simNodes

def getZennNodes():
    zennNodes = []
    for set in cmds.ls("ZN_ExportSet", r=True):
        for node in cmds.sets(set, q=True):
            zennNodes.append([node, 1, 1])
    return zennNodes

def saveCallback():
    current_file = cmds.file(q=True, sn=True)

    # NEW
    try:
        attachdic = {}
        attachdic['geoCache'] = getRigReferenced()
        attachdic['layout'] = getLayout()
        attachdic['camera'] = getCameras()
        attachdic["sim"] = getSimulation()
        attachdic["zenn"] = getZennNodes()
        attachdic["crowd"] = getCrowdInfo()

        filename = os.path.basename(current_file)

        coder = rb.Coder()
        rbRet = coder.F.MAYA.Decode(filename, 'BASE')
        if rbRet.has_key('task'):
            attachdic["task"] = rbRet['task']
            writeSaveCallback(current_file, attachdic)
    except Exception as e:
        print e.message


def writeSaveCallback(filePath, dataDic):
    dataDic['artist'] = user
    dataDic['time'] = datetime.datetime.now().isoformat()
    dataDic['file'] = filePath
    fr = (cmds.playbackOptions(q=True, min = True), cmds.playbackOptions(q=True, max = True))
    dataDic['frameRange'] = fr
    dataDic['mayaVersion'] = "2018"
    if os.environ.has_key('REZ_USED_RESOLVE'):
        dataDic['rezResolve'] = list()
        tmp = os.environ['REZ_USED_RESOLVE'].split()
        for resolve in tmp:
            if not 'cent' in resolve:
                dataDic['rezResolve'].append(resolve)
    if os.environ.has_key('REZ_USED_REQUEST'):
        dataDic['rezRequest'] = list()
        tmp = os.environ['REZ_USED_REQUEST'].split()
        for request in tmp:
            if not 'cent' in request:
                dataDic['rezRequest'].append(request)

    with open(filePath.replace(".mb", ".json"), "w+") as f:
        json.dump(dataDic, f, indent=4)

def createLogger(logName, logPath):
    if logPath == '':
        logPath = '/tmp/log/'+logName+'.log'

    if os.path.exists(logPath):
        os.remove(logPath)
        # print '이전 Log 파일이 존재하여 삭제 하였습니다: %s' % logPath

    logDir = os.path.dirname(logPath)
    if not os.path.isdir(logDir):
        try: os.makedirs(logDir)
        except: pass

    newLogger = logging.getLogger(logName)
    newLogger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(fmt='%(asctime)-15s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    if os.path.isdir(logDir):
        fileHandler = logging.FileHandler(logPath)
        fileHandler.setFormatter(formatter)
        newLogger.addHandler(fileHandler)

    return newLogger

def pathToNative(path):
    return os.sep.join(re.split(r'\\|/', os.path.normpath(path)))

def pathFromNative(path):
    return '/'.join(re.split(r'\\|/', os.path.normpath(path)))

def checkGroom(refReplPath):
    groomPath = refReplPath.split('/rig/')[0]+'/groom/groom.usd'
    if os.path.isfile(groomPath):
        return True

    return False

def JoinNsNameAndNode(nsName, nodeName):
    return nodeName if not nsName else '%s:%s' % (nsName, nodeName)

def checkSimFrame(offset):
    minf = int(cmds.playbackOptions(q=True, min=True))
    initf = minf - offset

    dxRigList = cmds.ls(type='dxRig', long=True)
    for node in dxRigList:
        switch = False
        nsName, nodeName = mutl.GetNamespace(node)
        result = list()
        for nCrvShp in cmds.getAttr('%s.controllers' % node):
            name = JoinNsNameAndNode(nsName, nCrvShp)

            try:
                if cmds.keyframe(name, q=True):
                    firstFrame = sorted(cmds.keyframe(name, q=True))[0]
                    # print '%s: %s' % (name, firstFrame)
                    for f in [950, initf]:
                        if f >= firstFrame:
                            switch = True
                    if switch:
                        mayaLogger.info(u'[ mayapy ] %s: initialize frame 있음: ' % node + unicode(str(firstFrame)))
                        break
            except Exception as e:
                mayaLogger.warning(u'[ mayapy ] '+unicode(e))
                # mayaLogger.warning(u'[ mayapy ] 오브젝트 이름이 하나 이상 존재합니다. '+unicode(name))
                continue

        if not switch:
            try:    mayaLogger.warning(u'[ mayapy ] %s: initialize frame이 존재하지 않습니다: rig firstFrame %s' % (node, firstFrame))
            except: mayaLogger.warning(u'[ mayapy ] %s: initialize frame이 존재하지 않습니다.' % node)

def checkDxRig():
    dxRigList = cmds.ls(type='dxRig', long=True)
    if len(dxRigList) > 0:
        for dxRig in dxRigList:
            nsLayer = dxRig.split(':')[0]
            rigGrp = dxRig.split(':')[-1]
            # print 'dxRig:', dxRig, nsLayer, rigGrp
            if not '_rig' in nsLayer:
                coder = rb.Coder()
                try:
                    argv = coder.N.MAYA.Decode(rigGrp)
                    if argv.asset not in nsLayer:
                        mayaLogger.error(u'[ mayapy ] dxRig의 네임스페이스에 어셋명이 없습니다. 확인 바랍니다: '+unicode(dxRig))
                except:
                    mayaLogger.error(u'[ mayapy ] dxRig의 이름을 리네임 한 것 같습니다. 확인 바랍니다: '+unicode(dxRig))

                if cmds.listRelatives(dxRig, parent=True) == None:
                    if not cmds.referenceQuery(dxRig, isNodeReferenced=True):
                        mayaLogger.error(u'[ mayapy ] dxRig가 reference가 아닙니다: '+unicode(dxRig))
                else:
                    mayaLogger.error(u'[ mayapy ] 최상위에 존재하지 않는 dxRig가 있습니다.')
            else:
                mayaLogger.error(u'[ mayapy ] dxRig의 네임스페이스 정리가 필요합니다: '+unicode(dxRig))

def createDxCamera():
    movedCamList = []
    if len(cmds.ls(type='dxCamera')) < 1:
        dxCam = cmds.createNode('dxCamera', n='dxCamera')
        exclCamList = [ 'persp', 'top', 'front', 'side' ]
        exclCamList.append(dxCam)
        camList = cmds.ls(type='camera', l=True)
        for camShp in camList:
            passParent = False
            camTopNode = '|'+camShp.split('|')[1]
            for exclCam in exclCamList:
                if camTopNode == exclCam:
                    passParent = True
                    break

            if not passParent:
                cmds.parent(camTopNode, dxCam)
                movedCamList.append(camTopNode)
        mayaLogger.info(u'[ mayapy ] dxCamera가 생성되었습니다. 포함된 카메라: '+unicode(movedCamList))
    return movedCamList

def checkChildNamespace():
    childNs = False
    nsList = cmds.namespaceInfo(lon=True, r=True)
    for ns in nsList:
        if ns.count(':') > 0:
            if cmds.ls(ns + ':*', type='dxRig'):
                childNs = True
                mayaLogger.error(u'[ mayapy ] 여러 계층으로 구성된 네임스페이스가 있습니다: '+unicode(ns))
                # os._exit(1)

    return childNs

def checkRenderable():
    renderable = False
    camList = cmds.ls(type='camera')
    for cam in camList:
        renderable = cmds.getAttr(cam+'.renderable')
        if renderable:
            break

    if not renderable:
        mayaLogger.warning(u'[ mayapy ] 렌더 가능한 카메라가 존재하지 않습니다. camera option에서 renderable 체크 바랍니다.')

    return renderable



# main code ####################################################################
# ------------------------------------------------------------------------------
# _config 내에 rulebook이 존재하는지 확인
# ------------------------------------------------------------------------------
fileName = os.path.basename(snPath)
fileNameNoext = os.path.splitext(fileName)[0]

showName = fileName.split('_')[0]
showRulebookFile = os.path.join('/show', showName, '_config', 'DXRulebook.yaml')
if os.path.exists(showRulebookFile):
    os.environ['DXRULEBOOKFILE'] = showRulebookFile
    rb.Reload()


# ------------------------------------------------------------------------------
# 벤더 파일 디코딩
# ------------------------------------------------------------------------------
coder = rb.Coder()
tmpv = coder.F.MAYA.vendor.Decode(fileName)
tmpv.departs = 'ANI'


# ------------------------------------------------------------------------------
# works 저정 경로, 파일 설정
# ------------------------------------------------------------------------------
aniWorks = coder.D.ANI.WORKS.Encode(**tmpv)
aniFile = coder.F.MAYA.BASE.Encode(**tmpv)
snPathAs = os.path.join(aniWorks, aniFile)


# ------------------------------------------------------------------------------
# 기본변수 설정
# ------------------------------------------------------------------------------
showCode = tmpv.show
simFrameOffset = 51


# ------------------------------------------------------------------------------
# 로그 시작
# ------------------------------------------------------------------------------
logMayaPath = os.path.dirname(snPath)+'/logs/'+os.path.basename(snPath)+'.log'
mayaLogger = createLogger('VENDOR_CHECK', logMayaPath)


# ------------------------------------------------------------------------------
# 씬 파일 오픈
# ------------------------------------------------------------------------------
try:
    cmds.file(snPath, o=True, f=True, options='v=0;', ignoreVersion=True, loadReferenceDepth='none')
except:
    pass
mayaLogger.info(u'[ mayapy ] 파일을 엽니다. '+unicode(snPath))


# ------------------------------------------------------------------------------
# 프레임레이트 체크
# ------------------------------------------------------------------------------
mayaLogger.info(u'[ mayapy ] Scene FrameRange: %s ~ %s' % (cmds.playbackOptions(q=True, min = True), cmds.playbackOptions(q=True, max = True)))
fps = mel.eval('currentTimeUnitToFPS')
if 24.0 != fps:
    mayaLogger.warning(u'[ mayapy ] FPS가 %s 입니다. 해당 쇼의 FPS가 맞는지 확인 해 주세요. ' % fps)
else:
    mayaLogger.info(u'[ mayapy ] FPS: %s' % fps)


# ------------------------------------------------------------------------------
# 이미지플렌 경로 세팅
# ------------------------------------------------------------------------------
for imgpln in cmds.ls(type='imagePlane'):
    ipPath = cmds.getAttr(imgpln+'.imageName')
    ipReplPath = ipPath.replace('.exr', '.jpg') # 일부 exr을 쓰는 vendor도 있다

    imageFile = os.path.basename(ipReplPath)
    if not 'letterbox' in imageFile:
        try:
            coder = rb.Coder()
            argv = coder.F.IMAGEPLANE.IMAGES.Decode(imageFile)
            argv.show = showCode
            argv.pub = '_2d'

            imageDir = ''
            for res in ['lo', 'hi']:
                argv.res = res
                imageDir = coder.D.IMAGEPLANE.IMAGES.Encode(**argv)
                if os.path.exists(imageDir):
                    break
            ipReplPath = os.path.join(imageDir, imageFile)
        except:
            pass
    else:
        ipReplPath = os.path.join('/show/{SHOW}/works/MMV/asset/camera'.format(SHOW=showCode), imageFile)

    if os.path.isfile(ipReplPath):
        cmds.setAttr(imgpln+'.imageName', ipReplPath, type='string')
        mayaLogger.info(u'[ mayapy ] 이미지플랜 교체 완료. '+unicode(ipReplPath))
    else:
        if ipReplPath and 'letterbox' not in ipReplPath:
            mayaLogger.warning(u'[ mayapy ] 교체가 필요한 이미지플랜 파일이 존재하지 않습니다. '+unicode(ipReplPath))


# ------------------------------------------------------------------------------
# 레퍼런스파일 경로 세팅
# ------------------------------------------------------------------------------
groomExists = False
refFileList = cmds.file(q=True, reference=True)
for refFile in refFileList:
    isBranch = False
    refNode = cmds.referenceQuery(refFile, referenceNode=True)
    refFile = refFile.split('{')[0] # cmds.file 에서 레퍼런스 리스트를 생성 할 때 중복되는 내용에 붙이는 중괄호 제거
    refPypath = pathFromNative(refFile)

    if '/show/' in refPypath:
        refReplPath = '/show' + refPypath.split('/show')[-1]
    elif 'DEXTER_CDH' in refPypath:
        refReplPath = '/show/cdh2/_3d' + refPypath.split('/_3d')[-1]
    elif '/_3d/' in refPypath:
        refReplPath = '/show/_3d' + refPypath.split('/_3d')[-1]
    elif 'stdrepo' in refPypath:
        refReplPath = '/stdrepo' + refPypath.split('/stdrepo')[-1]
    else:
        refReplPath = refPypath

    # branch 경로를 뒤죽박죽 쓰는 경우가 있어서 branch 타입을 찾는 작업을 함
    tmp = os.path.dirname(refReplPath).split('/')
    if 'branch' in tmp:
        isBranch = True
        chk = ['', showCode, 'show', '_3d', 'asset', 'branch', 'scenes']
        for i in chk:
            if i in tmp:
                tmp.remove(i)

    if not os.path.isfile(refReplPath):
        if refPypath.count('rig'):
            sw = False
            try:
                if not os.path.isfile(refReplPath):
                    assetFile = os.path.basename(refPypath).split('{')[0]
                    if 'baseMan' in assetFile:
                        assetDir = '/assetlib/_3d/asset/baseMan/rig/scenes'
                    else:
                        coder = rb.Coder()
                        argv = coder.F.MAYA.rig.Decode(assetFile)
                        argv.show = showCode
                        argv.pub = '_3d'
                        if isBranch:
                            argv.branch = tmp[-1]
                        assetDir = coder.D.OTHER.TASKSCENE.Encode(**argv)

                    refReplPath = os.path.join(assetDir, assetFile)
            except:
                pass

        else:
            if  '/_3d/asset/' in refReplPath:
                refReplPath = refReplPath.replace('/scenes/', '/rig/scenes/')

        if refFile.count('lidar') > 0 and refFile.endswith('.obj'):
            refReplPath = refFile.split(':')[-1]
            if not os.path.isfile(refReplPath):
                try:
                    lidarFile = os.path.basename(refFile)
                    coder = rb.Coder()
                    argv = coder.F.OBJECT.GEOM.Decode(lidarFile)
                    argv.show = showCode
                    argv.pub = '_3d'
                    if isBranch:
                        argv.branch = tmp[-1]
                    assetDir = coder.D.Encode(**argv)
                    ver = utl.GetLastVersion(assetDir)
                    refReplPath = os.path.join(assetDir, ver, lidarFile)
                except:
                    pass

    mayaLogger.info(u'[ mayapy ] 레퍼런스를 교체합니다. '+unicode(refPypath))
    mayaLogger.info(u'[ mayapy ] -> '+unicode(refReplPath))
    if os.path.isfile(refReplPath):
        cmds.file(refReplPath, loadReference=refNode)
        mayaLogger.info(u'[ mayapy ] 레퍼런스 교체 완료.')
        refGroom = checkGroom(refReplPath)
        if refGroom:
            mayaLogger.info(u'[ mayapy ] -> groom 파일이 존재합니다.')

            # initialize frame이 있는지 확인합니다.
            checkSimFrame(simFrameOffset)

    else:
        if 'lidar' not in refReplPath and 'spheregrid' not in refReplPath:
            mayaLogger.warning(u'[ mayapy ] 교체에 필요한 레퍼런스 파일이 존재하지 않습니다. '+unicode(refFile)+u' -> '+unicode(refReplPath))


# ------------------------------------------------------------------------------
# 여러 계층으로 구성된 네임스페이스가 있는지 확인합니다.
# ------------------------------------------------------------------------------
checkChildNamespace()


# ------------------------------------------------------------------------------
# 렌더가 가능한 카메라가 있는지 확인합니다.
# ------------------------------------------------------------------------------
checkRenderable()


# ------------------------------------------------------------------------------
# dxRig가 최상위에 있는지 확인합니다.
# ------------------------------------------------------------------------------
checkDxRig()


# ------------------------------------------------------------------------------
# dxCamera 노드가 없으면 생성합니다.
# ------------------------------------------------------------------------------
createDxCamera()


# ------------------------------------------------------------------------------
# 씬파일 works에 저장
# ------------------------------------------------------------------------------
snPathAsDir = os.path.dirname(snPathAs)
if not os.path.isdir(snPathAsDir):
    try: os.makedirs(snPathAsDir)
    except: pass

cmds.file(rename=snPathAs)
cmds.file(s=True, f=True)
saveCallback()
mayaLogger.info(u'[ mayapy ] 파일이 저장되었습니다. '+unicode(snPathAs))
mayaLogger.info(u'[ mayapy ] < CHECKED >')


# ------------------------------------------------------------------------------
# 종료
# ------------------------------------------------------------------------------
os._exit(0)
