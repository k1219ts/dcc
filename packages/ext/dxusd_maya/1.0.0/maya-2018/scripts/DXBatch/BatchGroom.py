# -*- coding: utf-8 -*-
import sys, os, argparse, getpass, json

# Tractor
import TractorConfig

# Rulebook
import DXRulebook.Interface as rb
from pxr import Sdf, Usd
import DXUSD.Vars as var
import DXUSD.Utils as utl

def GetGroomAgents(usdpath, grp):
    res = list()
    lyr = utl.AsLayer(usdpath)
    if not lyr:
        return res

    with utl.OpenStage(lyr) as stg:
        dprim = stg.GetDefaultPrim()
        it = iter(Usd.PrimRange.AllPrims(dprim))
        it.next()
        grpprim = None
        for prim in it:
            if prim.GetName() != grp:
                continue
            grpprim = prim
            break
        else:
            return res

        for agent in grpprim.GetChildren():
            for stack in agent.GetPrimStack():
                if not stack.referenceList.prependedItems:
                    continue
                ref = stack.referenceList.prependedItems[0]
                ref = utl.GetAbsPath(stack.layer.realPath, ref.assetPath)

                # find groom usd
                # TODO: match rig usd's nslyr name and rigVer
                refArg = var.D.ASSET.Decode(utl.DirName(ref))
                refArg.task = var.T.GROOM
                groomPath = var.D.TASK.Encode(**refArg)
                groomFile = var.F.TASK.Encode(**refArg)
                groomFile = utl.SJoin(groomPath, groomFile)

                groomLyr = utl.AsLayer(groomFile)
                if not groomLyr:
                    continue

                prim = groomLyr.GetPrimAtPath('/%s'%groomLyr.defaultPrim)
                for vn in [var.T.VAR_RIGVER, var.T.VAR_GROOMVER]:
                    vsel = prim.variantSelections[vn]
                    prim = prim.path.AppendVariantSelection(vn, vsel)
                    prim = groomLyr.GetPrimAtPath(prim)

                groomPath = prim.referenceList.prependedItems[0]
                groomFile = utl.GetAbsPath(groomFile, groomPath.assetPath)

                groomArg = var.D.Decode(utl.DirName(groomFile))
                groomScn = utl.SJoin(
                    var.D.TASK.Encode(**groomArg),
                    'scenes',
                    '%s.hip'%groomArg.nslyr
                )

                info = [
                    agent.GetPath().pathString,
                    groomScn
                ]
                res.append(info)
                break

    return res


class GroomHoudiniSpool():
    def __init__(self, inputCache, groomFile, nsver,
                 frameRange, step, user, primPattern=None):
        self.inputCache = inputCache
        self.groomFile = groomFile
        self.nsver = nsver
        self.frameRange = frameRange
        self.step = step
        self.user = user
        self.primPattern = primPattern

        command = ['/backstage/dcc/DCC', 'rez-env']
        # command += os.environ['REZ_USED_REQUEST'].split()
        jsonFile = groomFile.replace('.hip', '.json')
        if os.path.exists(jsonFile):
            with open(jsonFile, 'r') as f:
                sceneData = json.load(f)
                bundleName = sceneData['rezRequest']
                command += bundleName
        else:
            command += ['houBundle-AST18.5']

        tmp = groomFile.split('/')
        show = tmp[tmp.index('show')+1]
        command += ['--show', show]
        command += ['--'] # up on rez command
        command += ['DXBatchMain']
        command += ['--inputCache', inputCache]
        command += ['--groomFile', groomFile]
        command += ['--nsver', nsver]
        command += ['--frameRange', str(frameRange[0]), str(frameRange[1])]
        command += ['--step', str(float(self.step))]
        if primPattern:
            command += ['--primPattern', primPattern]

        print('########################################################################')
        print(' '.join(command))
        self.groomCommonCmd = command

        self.decodeInputCache()

        self.jobTitle = '(HOU) - (USD POST GROOM) %s' % self.inputCache.split('/%s/' % self.rbRet.seq)[-1]

    def decodeInputCache(self):
        coder = rb.Coder()
        self.rbRet = coder.D.Decode(self.inputCache)

    def doIt(self):
        job, JobTask = TractorConfig.MakeTractorJob(self.jobTitle, comment='inputCache : %s' % self.inputCache,
                                     metadata='groomAsset : %s\ngroomNsVer : %s' % (self.groomFile, self.nsver))
        # post script
        jobMsgCmd = ['/backstage/dcc/DCC', 'rez-env', 'rocketchattoolkit', '--', 'TrBotMsg']
        # Error
        job.newPostscript(argv=jobMsgCmd + ['-b', 'BadBot'], when='error')
        # Done
        job.newPostscript(argv=jobMsgCmd + ['-b', 'GoodBot'], when='done')

        JobTask.serialsubtasks = 1

        groomTask = TractorConfig.MakeTask('groom')
        groomTask.serialsubtasks = 1
        JobTask.addChild(groomTask)

        self.NonDispatchGeomTask(groomTask)
        # self.GeomTask(groomTask)
        # self.CompTask(groomTask)

        return job

    def spool(self):
        job = self.doIt()
        TractorConfig.SpoolJob(job, self.user)

    def NonDispatchGeomTask(self, groomTask):
        geomTask = TractorConfig.MakeTask('geom')
        cmd = list(self.groomCommonCmd)
        geomTask.addCommand(TractorConfig.MakeCommand(cmd))
        groomTask.addChild(geomTask)

    def GeomTask(self, groomTask):
        geomTask = TractorConfig.MakeTask('geom')
        for start, end in TractorConfig.GetIterFrames(self.frameRange):
            task = TractorConfig.MakeTask('frame %s-%s' % (start, end))
            cmd = list(self.groomCommonCmd)
            cmd += ['--exportRange', str(int(start)), str(int(end))]
            cmd += ['--process', 'geom']
            task.addCommand(TractorConfig.MakeCommand(cmd))
            geomTask.addChild(task)
        groomTask.addChild(geomTask)

    def CompTask(self, groomTask):
        groomTask.title = 'groom Composition'
        cmd = list(self.groomCommonCmd)
        cmd += ['--frameRange', str(int(self.frameRange[0])), str(int(self.frameRange[1]))]
        cmd += ['--process', 'comp']

        groomTask.addCommand(TractorConfig.MakeCommand(cmd))

class GroomSpool():
    def __init__(self, inputCache, groomFile, nsver, frameRange, step, user):
        self.inputCache = inputCache
        self.groomFile = groomFile
        self.nsver = nsver
        self.frameRange = frameRange
        self.step = step
        self.user = user

        command = ['/backstage/dcc/DCC', 'rez-env']
        # command += os.environ['REZ_USED_REQUEST'].split()
        for package in os.environ['REZ_USED_RESOLVE'].split():
            if 'centos' not in package:
                command.append(package)

        # groom은 maya-2018로 캐시아웃
        if 'maya-2022' in command:
            command[command.index('maya-2022')] = 'maya-2018'
            command[command.index('mayausd')] = 'usd_maya'
            command.insert(command.index('usd_maya'), 'zelos')

        tmp = groomFile.split('/')
        show = tmp[tmp.index('show')+1]
        command += ['--show', show]
        command += ['--'] # up on rez command
        command += ['DXBatchGroom']
        if self.inputCache:
            command += ['--inputCache', inputCache]
        command += ['--groomFile', groomFile]
        command += ['--nsver', nsver]
        command += ['--step', str(float(self.step))]
        self.groomCommonCmd = command

        self.decodeInputCache()

        if self.inputCache:
            self.jobTitle = '(USD POST GROOM) %s' % self.inputCache.split('/%s/' % self.rbRet.seq)[-1]
        else:
            self.jobTitle = '(USD GROOM SIMULATION) %s' % self.groomFile.split('/%s/' % self.rbRet.seq)[-1]

    def decodeInputCache(self):
        coder = rb.Coder()
        self.rbRet = coder.D.Decode(self.inputCache)

    def doIt(self):
        job, JobTask = TractorConfig.MakeTractorJob(self.jobTitle, comment='inputCache : %s' % self.inputCache,
                                     metadata='groomAsset : %s\ngroomNsVer : %s' % (self.groomFile, self.nsver))
        # post script
        jobMsgCmd = ['/backstage/dcc/DCC', 'rez-env', 'rocketchattoolkit', '--', 'TrBotMsg']
        # Error
        job.newPostscript(argv=jobMsgCmd + ['-b', 'BadBot'], when='error')
        # Done
        job.newPostscript(argv=jobMsgCmd + ['-b', 'GoodBot'], when='done')

        JobTask.serialsubtasks = 1

        groomTask = TractorConfig.MakeTask('groom')
        groomTask.serialsubtasks = 1
        JobTask.addChild(groomTask)

        self.GeomTask(groomTask)
        self.CompTask(groomTask)

        return job

    def spool(self):
        job = self.doIt()
        TractorConfig.SpoolJob(job, self.user)

    def GeomTask(self, groomTask):
        geomTask = TractorConfig.MakeTask('geom')
        for start, end in TractorConfig.GetIterFrames(self.frameRange):
            task = TractorConfig.MakeTask('frame %s-%s' % (start, end))
            cmd = list(self.groomCommonCmd)
            cmd += ['--exportRange', str(int(start)), str(int(end))]
            cmd += ['--process', 'geom']
            task.addCommand(TractorConfig.MakeCommand(cmd))
            geomTask.addChild(task)
        groomTask.addChild(geomTask)

    def CompTask(self, groomTask):
        groomTask.title = 'groom Composition'
        cmd = list(self.groomCommonCmd)
        cmd += ['--frameRange', str(int(self.frameRange[0])), str(int(self.frameRange[1]))]
        cmd += ['--process', 'comp']

        groomTask.addCommand(TractorConfig.MakeCommand(cmd))


if __name__ == '__main__':
    import DXUSD_MAYA.MUtils as mutl
    parser = argparse.ArgumentParser(
        description='DXUSD_MAYA Groom Batch script.'
    )

    parser.add_argument('-i', '--inputCache', type=str, help='groom src cache filepath.')
    parser.add_argument('-g', '--groomFile', type=str, help='groom asset scene or groom simulation scene')
    parser.add_argument('-nv', '--nsver', type=str, help='groom nsver')
    # TimeRange argument
    parser.add_argument('-fr',  '--frameRange', type=int, nargs=2, default=[0, 0], help='frame range. (start, end)')
    parser.add_argument('-efr', '--exportRange', type=int, nargs=2, default=[0, 0], help='export frame range. (start, end)')
    # parser.add_argument('-fs',  '--frameSample', type=float, default=1.0, help='frame step size default = 1.0')
    parser.add_argument('-s',   '--step', type=float, default=1.0, help='frame step size default = 1.0')

    # Acting argument
    parser.add_argument('-p', '--process', type=str, choices=['both', 'geom', 'comp'], help='task export when choice process, [geom, comp]')
    parser.add_argument('-u', '--user', type=str, default=getpass.getuser(), help='user name')

    args, unknown = parser.parse_known_args(sys.argv)
    # if not args.inputCache or not os.path.exists(args.inputCache):
    #     assert False, 'not found inputCache'

    from pymel.all import *
    loadPluginList = ['backstageMenu', 'pxrUsd', 'DXUSD_Maya', 'ZENNForMaya']
    mutl.InitPlugins(loadPluginList)

    import DXUSD_MAYA.Groom as Groom

    if args.inputCache:
        # print('>>>', args)
        Groom.shotCacheMergeExport(args.inputCache, groomFile=args.groomFile, fr=args.frameRange, efr=args.exportRange, step=args.step,
                                   version=args.nsver, user=args.user, process=args.process)
    else:
        Groom.groomSimExport(groomFile=args.groomFile, fr=args.frameRange, efr=args.exportRange, step=args.step,
                             version=args.nsver, user=args.user, process=args.process)
