import sys, os, site, re, glob, string, json

from pxr import Sdf, Usd, UsdGeom
import maya.cmds as cmds

import MsgSender
import dxsMsg
import Arguments
import dxsMayaUtils
import PathUtils
import Zenn
import os

import dxConfig
# tractor api setup
site.addsitedir( dxConfig.getConf("TRACTOR_API") )
import tractor.api.author as author
TRACTOR_IP = dxConfig.getConf("TRACTOR_CACHE_IP")# '10.0.0.25'
# TRACTOR_IP = '10.1.3.68'

DCC_PATH = '/backstage/bin/DCC'
TOOLKIT_PATH = os.getenv('REZ_MAYA_TOOLKIT_ROOT')
DXSUSD_DIR_PATH = os.path.join(TOOLKIT_PATH, 'scripts', 'dxsUsd')


def GetIterFrames(fr):
    duration = fr[1] - fr[0] + 1
    size = duration / 10
    if size < 5:
        size = 5
    if size > 50:
        size = 50
    result = list()
    for f in range(int(fr[0])-1, int(fr[1])+1, int(size)):
        start = f
        end   = f + size - 1
        if end > fr[1] + 1:
            end = fr[1] + 1
        if end - start > 2:
            result.append((start, end))
    if result[-1][-1] != fr[1]+1:
        result[-1] = (result[-1][0], fr[1]+1)
    return result


class SceneSpool:
    '''
    Args:
        fr (tuple): frameRange, not exportRange
        version (str): zenn output version
    '''
    def __init__(self, nsName, inputCache, zennFile, version, fr, step, user):
        self.nsName    = nsName
        self.inputCache= inputCache
        self.zennFile  = zennFile
        self.version   = version
        self.fr  = fr
        self.step= step
        self.user= user

        self.serviceKey = 'Cache'
        self.maxactive = 24
        self.projects  = 'export'
        self.tier = 'cache'
        self.tags = 'GPU'
        self.envkey = ''

        command = [DCC_PATH, 'rez-env']
        command += os.environ['REZ_RESOLVE'].split(' ')
        command += ['--', 'mayapy', os.path.join(DXSUSD_DIR_PATH, 'BatchZenn.py')]
        command+= ['--inputCache', inputCache]
        command+= ['--zennFile', zennFile]
        command+= ['--version', version]
        command+= ['--step', step]
        self.defaultCommand = command

        self.addInfo = {'comment': 'empty'}
        if os.environ['REZ_USED_RESOLVE']:
            self.addInfo['REZ_USED_RESOLVE'] = os.environ['REZ_USED_RESOLVE'].replace(' ', ',')

        # Member Variables
        self.shotName = ''
        self.title    = ''
        self.fileParser()

    def fileParser(self):
        if not '/show/' in self.inputCache:
            return
        splitPath = self.inputCache.split('/')
        index = splitPath.index('shot')
        seqName  = splitPath[index+1]
        shotName = splitPath[index+2]
        self.showDir = splitPath[splitPath.index("show") + 1]
        self.shotName = shotName
        self.title = self.inputCache.split('/%s/' % seqName)[-1]

        dbCommand = [DCC_PATH, 'rez-env', 'python-2', '--', 'python', os.path.join(DXSUSD_DIR_PATH, 'DBQuery.py')]
        dbCommand += ['--user', self.user]
        self.outDir = "/".join(splitPath[:index+3])
        self.outDir = os.path.join(self.outDir, "zenn", self.nsName, self.version)
        self.dbDefaultCommand = dbCommand

    def doIt(self):
        job = author.Job()
        job.title    = '(USD POST ZENN) %s' % self.title
        job.comment  = 'inputCache: %s' % self.inputCache
        job.metadata = 'zennAsset: %s, zennOutVersion: %s' % (self.zennFile, self.version)
        job.envkey   = [self.envkey]
        job.service  = self.serviceKey
        job.maxactive= self.maxactive
        job.tier = self.tier
        job.tags = [self.tags]
        job.projects = [self.projects]

        JobTask = author.Task(title='Job')
        JobTask.serialsubtasks = 1

        zennTask = author.Task(title='zenn')
        zennTask.serialsubtasks = 1
        JobTask.addChild(zennTask)

        self.FrameTask(zennTask)
        self.PayloadTask(zennTask)

        job.addChild(JobTask)
        # job.paused = True
        return job

    def dbTask(self, task, type, name, ver, addInfo = {}):
        dbCmd = self.dbDefaultCommand
        dbCmd += ["--showDir", self.showDir, "--shot", self.shotName, "--name", name, "--version", ver,
                  "--type", type, "--outDir", os.path.join(self.outDir)]
        if addInfo:
            for key in addInfo.keys():
                dbCmd += [key, addInfo[key]]
        task.addCommand(author.Command(argv=dbCmd, service=self.serviceKey))

    def PayloadTask(self, parent):
        jf = self.zennFile.replace('.mb', '.json')
        zennNodes = list()
        if os.path.exists(jf):
            with open(jf, 'r') as f:
                data = json.load(f)
                zennNodes = data['ZN_ExportSet']

        if zennNodes:
            ptask = author.Task(title='payload')
            for zn in zennNodes:
                task = author.Task(title=str(zn))
                cmd = list(self.defaultCommand)
                cmd+= ['--fr', self.fr[0], self.fr[1]]
                cmd+= ['--zennNode', zn]
                cmd+= ['--task', 'payload']
                task.addCommand(author.Command(argv=cmd, service=self.serviceKey))
                ptask.addChild(task)

            addInfo = {"zennNodes": zennNodes}
            addInfo.update(self.addInfo)
            self.dbTask(ptask, "zenn", self.nsName, self.version, addInfo=addInfo)
            parent.addChild(ptask)

        else:
            parent.title = 'zenn payload'
            cmd = list(self.defaultCommand)
            cmd+= ['--fr', self.fr[0], self.fr[1]]
            cmd+= ['--task', 'payload']

            parent.addCommand(author.Command(argv=cmd, service=self.serviceKey))
            self.dbTask(parent, "zenn", self.nsName, self.version, addInfo=self.addInfo)


    def FrameTask(self, parent):
        ptask = author.Task(title='geom')
        for start, end in GetIterFrames(self.fr):
            task = author.Task(title='frame %s-%s' % (start, end))
            cmd  = list(self.defaultCommand)
            cmd += ['--fr', int(start), int(end)]
            cmd += ['--step', self.step]
            cmd += ['--task', 'geom']
            task.addCommand(author.Command(argv=cmd, service=self.serviceKey))
            ptask.addChild(task)
        parent.addChild(ptask)



class MakeCacheMergeScene(Arguments.CommonArgs):
    '''
    Args:
        zennFile (str): zenn maya file
        inputCache (str): exported geom masterFile. merge to zenn inBodyMesh
    '''
    def __init__(self, inputCache, outDir=None, version=None, zennFile=None, zennNode=None, **kwargs):
        self.inputCache= inputCache
        self.outDir    = outDir
        self.version   = version
        self.zennFile  = zennFile
        self.zennNode  = zennNode
        Arguments.CommonArgs.__init__(self, **kwargs)

        # Member variables
        self.showDir = None
        self.showName= None
        self.seqName = None
        self.shotName= None
        self.dependency  = dict()
        self.rigAssetDir = None
        self.assetName   = None

        self.fileParser()

    def fileParser(self):
        if not '/show/' in self.inputCache:
            return
        splitPath = self.inputCache.split('/')
        index = splitPath.index('show')
        self.showDir = string.join(splitPath[:index+2], '/')
        self.showName= self.showDir.split('/')[-1].replace('_pub', '')
        index = splitPath.index('shot')
        self.seqName = splitPath[index+1]
        self.shotName= splitPath[index+2]
        task   = splitPath[index+3]
        version= splitPath[-2]
        self.dependency = {'task': [task], 'version': [version]}

        # Find Animation Cache and then, Get fr, fps, assetName, rigVersion
        aniCacheFile = self.inputCache
        if task == 'sim':
            rootLayer = Sdf.Layer.FindOrOpen(self.inputCache)
            customLayerData= rootLayer.customLayerData
            aniCacheFile   = customLayerData['simInputCacheFile']
            self.dependency['task'].append('ani')
            self.dependency['version'].append(os.path.basename(os.path.dirname(aniCacheFile)))

        stage = Usd.Stage.Open(aniCacheFile, load=Usd.Stage.LoadNone)
        customLayerData = stage.GetRootLayer().customLayerData
        if not self.fr[0] and not self.fr[1]:
            self.fr   = (customLayerData['start'], customLayerData['end'])
        self.fps  = stage.GetFramesPerSecond()
        self.expfr= (self.fr[0] - 1, self.fr[1] + 1)
        self.assetName = customLayerData['asset']
        if customLayerData.has_key('rigAssetDir'):
            self.rigAssetDir = customLayerData['rigAssetDir']

        dprim = stage.GetDefaultPrim()
        self.nsLayer= dprim.GetPath().name
        customData  = dprim.GetCustomData()
        self.rigVersion = customData['rig']


    def getZennAssetFile(self):
        if self.zennFile:
            return
        import batchCommon
        zennInfo = batchCommon.GetZennAssetInfo(self.showDir.replace('_pub', ''))
        assert zennInfo, "# msg : Not found 'AssetInfo.json' file"
        if not zennInfo.has_key(self.assetName):
            return
        zennAssetInfo = zennInfo[self.assetName]
        try:
            if zennAssetInfo.has_key('rigHair'):
                for ver in reversed(sorted(zennAssetInfo['rigHair'].keys())):
                    if self.shotName in zennAssetInfo['rigHair'][ver]['shotList']:
                        if self.rigVersion in zennAssetInfo['rigHair'][ver]['rigVersion']:
                            self.zennFile = zennAssetInfo['rigHair'][ver]['assetFile']
                            break
                        else:
                            print "[GetZennAsset] - Not found 'rigHair'"
            if not self.zennFile and zennAssetInfo.has_key('modelHair'):
                for ver in reversed(sorted(zennAssetInfo['modelHair'].keys())):
                    if zennAssetInfo['modelHair'][ver].has_key('shotList'):
                        if self.shotName in zennAssetInfo['modelHair'][ver]['shotList']:
                            if self.rigVersion in zennAssetInfo['modelHair'][ver]['rigVersion']:
                                self.zennFile = zennAssetInfo['modelHair'][ver]['assetFile']
                                break
                            else:
                                print "[GetZennAsset] - Not found assigned shotList in 'modelHair'"
            if not self.zennFile and zennAssetInfo.has_key('modelHair'):
                for ver in reversed(sorted(zennAssetInfo['modelHair'].keys())):
                    if not zennAssetInfo['modelHair'][ver].has_key('shotList'):
                        if self.rigVersion in zennAssetInfo['modelHair'][ver]['rigVersion']:
                            self.zennFile = zennAssetInfo['modelHair'][ver]['assetFile']
                            break
                        else:
                            print "[GetZennAsset] - Not found 'modelHair'"
        except:
            pass


    def getCustomData(self):
        result = dict()
        result['layer'] = {
            'zennAssetFile': self.zennFile,
            'zennInputCacheFile': self.inputCache
        }
        splitPath = os.path.dirname(self.inputCache).split('/')
        result['prim'] = {
            'zennAsset': os.path.basename(self.zennFile),
            'zennInputCache': string.join(splitPath[splitPath.index(self.shotName)+1:], '/')
        }
        result['dependency'] = self.dependency
        return result


    def sceneStart(self):
        self.getZennAssetFile()
        if not self.zennFile:
            print '# Debug : not found zennFile'
            return
        dxsMsg.Print('info', "[ZennAssetFile] - %s" % self.zennFile)

        customData = self.getCustomData()
        if self.rigAssetDir:
            customData['layer']['rigAssetDir'] = self.rigAssetDir

        comment = 'Generated with %s, %s' % (self.zennFile, self.inputCache)

        self.ZS = Zenn.ZennScene(self.zennFile, show=self.showName, shot=self.shotName, user=self.user)
        dxsMayaUtils.SetFPS(self.fps)

        args = {
            'nsLayer': self.nsLayer, 'fr': self.fr, 'step': self.step, 'fps': self.fps,
            'version': self.version, 'customData': customData, 'comment': comment
        }
        if self.outDir:
            args['outDir'] = self.outDir
        else:
            args['showDir'] = self.showDir
            args['seq'] = self.seqName
            args['shot']= self.shotName
        return args


    def doIt(self):
        args = self.sceneStart()
        if not args:
            return

        self.MergeCache()

        ZennExportClass = Zenn.ZennShotExport(**args)
        ZennExportClass.doIt(0)
        ZennExportClass.doIt(1)

    def proc(self, task='geom'):
        args = self.sceneStart()
        if not args:
            return

        if task == 'geom':
            self.MergeCache()

            ZennExportClass = Zenn.ZennShotExport(**args)
            ZennExportClass.expfr = (self.fr[0], self.fr[1])
            ZennExportClass.doIt(0)

        else:
            if self.zennNode:
                args['zennNodes'] = [self.zennNode]
            Zenn.ZennShotExport(**args).doIt(1)


    def MergeCache(self):
        import xbUtils
        xBlockNode = cmds.ls(type='xBlock', l=True)
        if not xBlockNode:
            xBlockImport = xbUtils.UsdImport(self.inputCache)
            rigGeomFile  = xBlockImport.getRigGeomFilename()
            xBlockNode   = xBlockImport.importGeom(rigGeomFile)
            for destMesh in self.ZS.zennBaseMeshes:
                dxsMayaUtils.ConnectBlendShape(destination=destMesh, sourceroot=xBlockNode).doIt()
        else:
            if len(xBlockNode) > 1:
                dxsMsg.Print('warning', 'xBlockNode has multiple')
            else:
                xBlockNode = xBlockNode[0]

        xbUtils.CacheMerge.UsdMerge(self.inputCache, [xBlockNode]).doIt()

        # re-initialize
        reInitNodes = list()    # ZN_Import
        for znode in self.ZS.zennImportNodeMap:
            for znip in self.ZS.zennImportNodeMap[znode]:   # ZN_Import
                curveGroup = cmds.listConnections('%s.inGuideCurves' % znip, s=True, d=False)
                if curveGroup:
                    curveGroup = cmds.ls(curveGroup, l=True)[0]
                    if xBlockNode in curveGroup:
                        reInitNodes.append(znip)

        if not reInitNodes:
            return

        restTime = 0
        for znode in self.ZS.zennImportNodeMap:
            for znip in self.ZS.zennImportNodeMap[znode]:
                if znip in reInitNodes:
                    restTime = cmds.getAttr('%s.restTime' % znip)
                    cmds.setAttr('%s.updateMesh' % znip, 1)
                    cmds.dgeval(znip)

        perFrameImportNodes = list()
        for znip in reInitNodes:
            if cmds.getAttr('%s.perFrameImport' % znip):
                perFrameImportNodes.append(znip)

        if not perFrameImportNodes:
            return

        cmds.currentTime(self.fr[0])
        cmds.currentTime(restTime)

        for znip in perFrameImportNodes:
            cmds.setAttr('%s.updateMesh' % znip, 1)
            cmds.dgeval(znip)


    def spool(self):
        self.getZennAssetFile()
        if not self.zennFile:
            print '# Debug : not found zennFile'
            return
        if self.fr[0] == self.fr[1]:
            return
        outDir = '{SHOW}/shot/{SEQ}/{SHOT}/zenn/{NAME}'.format(SHOW=self.showDir, SEQ=self.seqName, SHOT=self.shotName, NAME=self.nsLayer)
        version= PathUtils.GetVersion(outDir)
        job = SceneSpool(self.nsLayer, self.inputCache, self.zennFile, version, self.fr, self.step, self.user).doIt()
        job.priority = 100
        author.setEngineClientParam(hostname=TRACTOR_IP, port=80, user=self.user, debug=True)
        job.spool()
        author.closeEngineClient()


def GetXBlockMergeFile(node):
    source = node.split('|')
    for i in range(1, len(source)):
        npath = string.join(source[:i+1], '|')
        if cmds.nodeType(npath) == 'xBlock':
            return cmds.getAttr('%s.mergeFile' % npath)


def SceneExport(filename, outDir, version, fr, step, task='', user='', zennNode = ''):
    '''
    Args:
        outDir (str) : xxx/zenn/$NS (include nsLayer name)
    '''
    import batchCommon
    import Zenn

    # outdir parse
    showDir, seqName, shotName = batchCommon.PathParser(outDir)
    showName = showDir.split('/')[-1].replace('_pub', '')

    ZS = Zenn.ZennScene(filename, show=showName, shot=shotName, user=user)
    zennBaseMeshes = ZS.zennBaseMeshes

    # find xBlock
    inputCache = None
    inputCache = GetXBlockMergeFile(zennBaseMeshes[0])

    # edit 2019.03.25
    ns = os.path.basename(outDir)
    targetNodes  = list()
    targetMeshes = list()
    for m in zennBaseMeshes:
        if m.split('|')[-1].split(':')[0] == ns:
            targetMeshes.append(m)
            inputCache = GetXBlockMergeFile(m)
    if targetMeshes:
        for s in cmds.ls('ZN_ExportSet', r=True):
            for z in cmds.sets(s, q=True):
                meshes = Zenn.GetZennBaseMesh([z])
                for m in meshes:
                    if m in targetMeshes:
                        targetNodes.append(z)
    else:
        targetNodes = cmds.sets('ZN_ExportSet', q=True)

    # customData
    customData = {'layer': {}, 'prim': {}, 'dependency': {'task': [], 'version': []}}

    ZennExportClass = Zenn.ZennShotExport(zennNodes=targetNodes, fr=fr, step=step, outDir=outDir, version=version, customData=customData)

    if task == 'geom':
        ZennExportClass.expfr = (fr[0], fr[1])
        ZennExportClass.doIt(0)
    elif task == 'payload':
        if inputCache:
            customData = GetZennCustomData(filename, inputCache)
            ZennExportClass.customData = customData
        if zennNode:
            ZennExportClass.zennNodes = [zennNode]

        ZennExportClass.doIt(1)
    else:
        if inputCache:
            customData = GetZennCustomData(filename, inputCache)
            ZennExportClass.customData = customData
        ZennExportClass.doIt(0)
        ZennExportClass.doIt(1)


def GetZennCustomData(zennFile, inputCache):
    splitPath = os.path.dirname(inputCache).split('/')
    task = splitPath[splitPath.index('shot') + 3]
    iver = splitPath[-1]
    dependency = {'task': [task], 'version': [iver]}

    result = dict()
    result['layer'] = {
        'zennAssetFile': zennFile,
        'zennInputCacheFile': inputCache
    }
    result['prim'] = {
        'zennAsset': os.path.basename(zennFile),
        'zennInputCache': string.join(splitPath[splitPath.index('shot')+2:], '/')
    }

    if task == 'sim':
        rootLayer = Sdf.Layer.FindOrOpen(inputCache)
        customLayerData = rootLayer.customLayerData
        if customLayerData:
            inputCache = customLayerData['simInputCacheFile']
            dependency['task'].append('ani')
            dependency['version'].append(os.path.basename(os.path.dirname(inputCache)))
    result['dependency'] = dependency
    return result


if __name__ == '__main__':
    import batchCommon
    optparser = batchCommon.zennOptParserSetup()
    opts, args = optparser.parse_args(sys.argv)
    if not opts.inputCache and not opts.zennFile:
        os._exit(1)

    from pymel.all import *
    print "# already load pluginList :", cmds.pluginInfo(q = True, ls = True)
    plugins = ['backstageMenu', 'ZENNForMaya', 'pxrUsd']
    batchCommon.InitPlugins(plugins)

    '''
    opts memo.
    - only hair : inputCache
    - only payload : zennNode
    '''

    if opts.inputCache:
        MakeCacheMergeScene(opts.inputCache, outDir=opts.outdir, version=opts.version, zennFile=opts.zennFile, zennNode=opts.zennNode, fr=opts.frameRange, step=opts.step).proc(opts.task)
    else:
        SceneExport(opts.zennFile, opts.outdir, opts.version, opts.frameRange, opts.step, opts.task, opts.user, opts.zennNode)

    # quit
    os._exit(0)
