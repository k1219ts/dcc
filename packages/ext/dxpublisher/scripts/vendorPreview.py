#coding:utf-8
import maya.standalone
maya.standalone.initialize()

import maya.cmds as cmds
import maya.mel as mel
import os, sys, re, shutil, glob, subprocess, logging

if not cmds.pluginInfo("pxrUsd", q=True, l=True):
    cmds.loadPlugin('pxrUsd')

if not cmds.pluginInfo("DXUSD_Maya", q=True, l=True):
    cmds.loadPlugin('DXUSD_Maya')

import DXUSD.Utils as utl
import DXUSD_MAYA.MUtils as mutl
import DXRulebook.Interface as rb

snPath = sys.argv[1]
snPathAs = sys.argv[2]
prevDir = os.path.dirname(snPath)+'/preview'

def makeShotPreview(prevDir):
    cmds.setAttr('defaultRenderGlobals.currentRenderer', 'mayaSoftware', type='string')
    cmds.setAttr('defaultRenderGlobals.imageFormat', 8)
    cmds.setAttr('defaultRenderGlobals.animation', 0)
    cmds.setAttr('defaultRenderGlobals.enableDepthMaps', 0)
    cmds.setAttr('defaultRenderGlobals.jitterFinalColor', 0)
    cmds.setAttr('defaultRenderGlobals.clipFinalShadedColor', 0)
    cmds.setAttr('defaultRenderGlobals.recursionDepth', 1)
    cmds.setAttr('defaultRenderGlobals.leafPrimitives', 50)
    cmds.setAttr('defaultRenderGlobals.subdivisionPower', 0.01)
    cmds.setAttr('defaultRenderGlobals.enableStrokeRender', 0)

    cmds.setAttr('defaultResolution.width', 960)
    cmds.setAttr('defaultResolution.height', 540)
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
    cmds.setAttr('defaultRenderQuality.reflections', 1)
    cmds.setAttr('defaultRenderQuality.refractions', 6)
    cmds.setAttr('defaultRenderQuality.shadows', 2)
    cmds.setAttr('defaultRenderQuality.rayTraceBias', 0.0)
    cmds.setAttr('defaultRenderQuality.edgeAntiAliasing', 3)
    cmds.setAttr('defaultRenderQuality.renderSample', False)
    cmds.setAttr('defaultRenderQuality.useMultiPixelFilter', False)
    cmds.setAttr('defaultRenderQuality.pixelFilterType', 2)
    cmds.setAttr('defaultRenderQuality.pixelFilterWidthX', 2.20000004768)
    cmds.setAttr('defaultRenderQuality.pixelFilterWidthY', 2.20000004768)
    cmds.setAttr('defaultRenderQuality.plugInFilterWeight', 1.0)
    cmds.setAttr('defaultRenderQuality.shadingSamples', 1)
    cmds.setAttr('defaultRenderQuality.maxShadingSamples', 1)
    cmds.setAttr('defaultRenderQuality.visibilitySamples', 1)
    cmds.setAttr('defaultRenderQuality.maxVisibilitySamples', 4)
    cmds.setAttr('defaultRenderQuality.volumeSamples', 1)
    cmds.setAttr('defaultRenderQuality.particleSamples', 1)
    cmds.setAttr('defaultRenderQuality.enableRaytracing', False)
    cmds.setAttr('defaultRenderQuality.redThreshold', 0.40000000596)
    cmds.setAttr('defaultRenderQuality.greenThreshold', 0.300000011921)
    cmds.setAttr('defaultRenderQuality.blueThreshold', 0.600000023842)
    cmds.setAttr('defaultRenderQuality.coverageThreshold', 0.125)

    snFn = os.path.basename(snPathAs)
    snSeq = '_'.join(snFn.split('_')[:2])
    camShList = cmds.ls(snSeq+'_*main*', type='camera')
    minF = int(cmds.playbackOptions(min=True, q=True))
    maxF = int(cmds.playbackOptions(max=True, q=True))
    for f in range(minF, maxF+1):
        cmds.currentTime(int(f))
        for camSh in camShList:
            camTr = cmds.listRelatives(camSh, parent=True)[0]
            camTrFn = camTr.split('|')[-1].replace(':', '--')
            outPath = cmds.render(camSh)
            imgDir = prevDir+'/'+camTrFn
            try: os.makedirs(imgDir)
            except: pass
            imgPath = imgDir+'/'+camTrFn+'.'+str(f).zfill(4)+'.jpg'
            print outPath, '->', imgPath
            shutil.copy(outPath, imgPath)

    for camSh in camShList:
        camTr = cmds.listRelatives(camSh, parent=True)[0]
        camTrFn = camTr.split('|')[-1].replace(':', '--')
        imgDir = prevDir+'/'+camTrFn
        imgList = glob.glob(imgDir+'/'+camTrFn+'.*.jpg')
        totalF = maxF-minF+1
        if len(imgList) == totalF:
            cmd = '/backstage/dcc/DCC rez-env ffmpeg_toolkit -- ffmpeg_converter -s 1920x1080 -i  '+imgDir+' -o '+prevDir
            process = subprocess.Popen(cmd, shell=True)
            process.communicate()

def createLogger(logName, logPath):
    if logPath == '':
        logPath = '/tmp/log/'+logName+'.log'
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

def getDags(level=3):
    camExclList = ['persp', 'top', 'front', 'side', 'left', 'right', 'back', 'bottom']
    dagList = []
    for asm in cmds.ls(assemblies=True):
        if str(asm) in camExclList:
            continue
        dagList.append(asm)

    dagWalked = []
    def dagWalk(node, nodeList, level, depth=1):
        if not 'Shape' in node:
            nodeList.append(node)
        if level <= depth:
            return
        children = cmds.listRelatives(node, children=True, fullPath=True)
        if children != None:
            for child in children:
                dagWalk(child, nodeList, level, depth+1)

    for dag in dagList:
        dagWalk(dag, dagWalked, level)

    nodeDict = {}
    for dag in dagWalked:
        node = nodeDict
        dagTok = dag.split('|')
        for de in dagTok:
            if de:
                node = node.setdefault(de, dict())

    dagDict = { 'level': level }
    dagDict['nodes'] = nodeDict

    return dagDict

treeLines = []
def jsonKeyToTree(jn, depth=0):
    ch = u' | '
    for k in jn:
        if isinstance(jn[k], dict):
            treeLines.append(u' '+ch*(depth-1)+u' L '+str(depth)+u' '+ k)
            jsonKeyToTree(jn[k], depth+1)

logMayaPath = os.path.dirname(snPath)+'/logs/'+os.path.basename(snPath)+'.log'
mayaLogger = createLogger('VENDOR_CHECK', logMayaPath)

# cmds.file(snPathAs, o=True, f=True, options='v=0;', ignoreVersion=True)
cmds.file(snPathAs, o=True, f=True, options='v=0;', ignoreVersion=True, loadReferenceDepth='none')
refFileList = cmds.file(q=True, reference=True)
for refFile in refFileList:
    refNode = cmds.referenceQuery(refFile, referenceNode=True)
    try:  cmds.file(refFile, loadReference=refNode)
    except: mayaLogger.warning(u'레퍼런스 파일이 존재하지 않습니다. '+refFile)

jsonKeyToTree(getDags())
treeLines[0] = u'-'*80
treeLines = [u'[ mayapy ] DAG Tree']+ treeLines+[u'-'*80]
mayaLogger.info('\n'.join(treeLines))

makeShotPreview(prevDir)
mayaLogger.info(u'[ mayapy ] < PREVIEW >')
