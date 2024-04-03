# -*- coding: utf-8 -*-
import os, argparse, datetime, sys, traceback, json

import DXUSD.Utils as utl
import DXUSD.Vars as var
import DXUSD_MAYA.Message as msg
import DXUSD_MAYA.MUtils as mutl
import DXRulebook.Interface as rb

import DXUSD_MAYA.Rig as Rig
import DXUSD_MAYA.Sim as Sim
import DXUSD_MAYA.Camera as Cam
import DXUSD_MAYA.Groom as Groom
import DXUSD_MAYA.Layout as Layout

from DXBatch import BatchGroom
from DXBatch import TractorConfig

import maya.cmds as cmds
import maya.api.OpenMaya as OpenMaya

def DBConnection(dbName, collName):
    import dxConfig
    from pymongo import MongoClient
    DB_IP = dxConfig.getConf('DB_IP')

    client = MongoClient(DB_IP)
    db = client[dbName]
    coll = db[collName]

    return coll

class SceneSpool():
    def __init__(self, args):
        self.args = args

        tmp = args.file.split('/')
        show = tmp[tmp.index('show')+1]

        command = ['/backstage/dcc/DCC', 'rez-env']
        #command += os.environ['REZ_USED_REQUEST'].split()

        # nxer ani 캐시아웃은 2018의 MASH 버그로 2022로 함 (2023.01.16 관태)
        packages = os.environ['REZ_USED_RESOLVE'].split()
        if  'die' in show and 'golaem_maya-8.1.3' not in packages:
            if 'maya-2022' not in packages:
                packages[packages.index('maya-2018')] = 'maya-2022'
                if 'zelos' in packages:
                    packages.remove('zelos')
            for rez in packages:
                if 'usd_maya-19' in rez:
                    packages[packages.index(rez)] = 'mayausd'
            os.environ['REZ_USED_RESOLVE'] = ' '.join(packages)

        for package in os.environ['REZ_USED_RESOLVE'].split():
            if 'centos' not in package:
                command.append(package)
        command += ['--show', show]
        command += ['--']
        command += ['DXBatchMain']
        command += ['--file', args.file]
        command += ['--outDir', args.outDir]
        command += ['--frameRange', str(args.frameRange[0]), str(args.frameRange[1])]
        command += ['--step', str(args.step)]
        command += ['--user', args.user]
        command += ['--host', 'tractor'] # because already spool command
        self.commonCommand = command

        # ???
        # self.addInfo = {'comment': 'empty'}
        # if os.environ['REZ_USED_RESOLVE']:
        #     self.addInfo['REZ_USED_RESOLVE'] = os.environ['REZ_USED_RESOLVE'].replace(' ', ',')

        self.rbRet = self.dbInformationParsing()
        self.action = 0

    def dbInformationParsing(self):
        decoder = rb.Coder('D')
        ret = decoder.Decode(self.args.outDir)
        if ret.has_key('show'):
            self.show = ret['show']

        if ret.has_key('shot'):
            self.shot = '%s_%s' % (ret['seq'], ret['shot'])

        return ret

    def doIt(self):
        title = '(USD) %s' % str(os.path.basename(self.args.file))
        comment = 'out directory : %s' % str(self.args.outDir)
        metadata = 'scene file : %s' % str(self.args.file)

        # 임시로 nxer ani 캐시아웃은 2018의 MASH 버그로 2022로 함 (2022.08.01 관태)
        packages = os.environ['REZ_USED_RESOLVE'].split()
        if 'maya-2022' in packages and 'golaem-8.1.3' not in packages:
            job, jobTask = TractorConfig.MakeTractorJob(title, comment=comment, metadata=metadata, service='MAYA2022')
        else:
            job, jobTask = TractorConfig.MakeTractorJob(title, comment=comment, metadata=metadata)
        # post script
        jobMsgCmd = ['/backstage/dcc/DCC', 'rez-env', 'rocketchattoolkit', '--', 'TrBotMsg']
        # Error
        job.newPostscript(argv=jobMsgCmd + ['-b', 'BadBot'], when='error')
        # Done
        job.newPostscript(argv=jobMsgCmd + ['-b', 'GoodBot'], when='done')

        jobTask.serialsubtasks = 1

        # coll = DBConnection(dbName, self.show)
        # dbItem = {"action": "export",
        #           "filepath": opts.srcfile,
        #           "user": opts.user,
        #           "time": datetime.datetime.today().isoformat(),
        #           "shot": shotName,
        #           "task": "ani",
        #           "enabled": False}
        # objId = coll.insert_one(dbItem).inserted_id
        #
        # command = ['/backstage/bin/DCC', 'rez-env', 'pylibs', '--', "/backstage/apps/inventory/enableDBRecord.py",
        #            dbName, showName, objId]
        # JobTask.addCommand(author.Command(argv=command, service=self.serviceKey))

        exportTask = TractorConfig.MakeTask('USD Export')
        jobTask.addChild(exportTask)

        if self.args.mesh:
            self.MeshTask(exportTask)
        if self.args.simMesh:
            self.SimTask(exportTask)
        if self.args.layout:
            self.LayoutTask(exportTask)
        if self.args.camera:
            self.CameraTask(exportTask)
        if self.args.crowd:
            self.CrowdTask(exportTask)
        if self.args.groomSim:
            self.GroomSimTask(exportTask)

        if not job or self.action <= 0:
            os._exit(0)
        TractorConfig.SpoolJob(job, args.user)

    # ---------------------------------------------------------------------------
    #
    #   MESH
    #
    # ---------------------------------------------------------------------------
    def MeshTask(self, parent):
        meshJob = TractorConfig.MakeTask('mesh')
        taskCmd = list(self.commonCommand)

        meshRet = self.rbRet
        meshRet['task'] = 'ani'
        meshRet['ext'] = 'usd'

        msg.debug(self.args.mesh)

        for meshInfo in self.args.mesh:
            version, nodeName = meshInfo.split('=')
            nodeRet = rb.Coder().N.USD.ani.Decode(nodeName, 'SHOT')
            nsLayer = nodeRet.nslyr

            if self.args.onlyGroom:
                meshRet['nslyr'] = nsLayer
                meshRet['nsver'] = version

                masterFileName = rb.Coder().F.USD.ani.MASTER.Encode(**meshRet)
                meshOutDir = rb.Coder().D.TASKNV.Encode(**meshRet)

                inputCache = utl.SJoin(meshOutDir, masterFileName)

                if os.path.exists(inputCache):
                    srclyr = utl.AsLayer(inputCache)
                    srcdata = srclyr.customLayerData
                    rigFile = srcdata.get('rigFile')
                    if not rigFile:
                        continue

                    outDir = self.args.outDir
                    configFile = Groom.GetConfigGroomFile(rigFile, outDir, nsLayer)
                    if configFile:
                        groomFile = configFile
                    else:
                        groomFile = Groom.GetGroomFile(rigFile, variant=srcdata.get('variant'))

                    if groomFile:
                        if groomFile.endswith('.mb'):
                            msg.debug('found Groom Maya File')

                            _tmp = meshRet['task']
                            meshRet['task'] = 'groom'
                            groomOutDir = rb.Coder().D.TASKN.Encode(**meshRet)
                            nsver = utl.GetNextVersion(groomOutDir)

                            # nsLayer, inputCache, groomFile, nsver, frameRange, step, user
                            BatchGroom.GroomSpool(inputCache, groomFile, nsver, (srcdata['start'], srcdata['end']),
                                                  self.args.step, self.args.user).spool()
                            print 'Batch Groom Spool'
                            meshRet['task'] = _tmp

                        elif groomFile.endswith('.hip'):
                            msg.debug('found Groom Hip File')

                            _tmp = meshRet['task']
                            meshRet['task'] = 'groom'
                            groomOutDir = rb.Coder().D.TASKN.Encode(**meshRet)
                            nsver = utl.GetNextVersion(groomOutDir)

                            # nsLayer, inputCache, groomFile, nsver, frameRange, step, user
                            BatchGroom.GroomHoudiniSpool(inputCache, groomFile, nsver, (srcdata['start'], srcdata['end']),
                                                  self.args.step, self.args.user).spool()
                            print 'Batch Groom Spool'
                            meshRet['task'] = _tmp
            else:
                subTask = TractorConfig.MakeTask('%s %s' % (nsLayer, version))

                cmd = taskCmd + ['--mesh', '%s=%s' % (version, nodeName)]
                if self.args.groom:
                    cmd += ['--groom']
                if self.args.rigUpdate:
                    cmd += ['--rigUpdate']

                print '\t\t', cmd

                subTask.addCommand(TractorConfig.MakeCommand(cmd + ['--process', 'geom']))

                # batchGroom은 maya-2018로 캐시아웃
                if self.args.groom:
                    if 'maya-2022' in cmd:
                        cmd[cmd.index('maya-2022')] = 'maya-2018'
                        cmd[cmd.index('mayausd')] = 'usd_maya'
                        cmd.insert(cmd.index('usd_maya'), 'zelos')

                parent.addCommand(TractorConfig.MakeCommand(cmd + ['--process', 'comp']))

                # self.dbTask(subTask, "ani", nsLayer, version, addInfo=self.addInfo)
                # meshJob.addChild(subTask)

                self.action += 1
                parent.parent.parent.comment += ' ani/{NS}/{VER}'.format(NS=nsLayer, VER=version)
                meshJob.addChild(subTask)

        parent.addChild(meshJob)

    # ---------------------------------------------------------------------------
    #
    #   MESH
    #
    # ---------------------------------------------------------------------------
    def SimTask(self, parent):
        simJob = TractorConfig.MakeTask('sim')
        taskCmd = list(self.commonCommand)

        msg.debug(self.args.simMesh)

        simMeshRet = self.rbRet
        simMeshRet['task'] = 'sim'
        simMeshRet['ext'] = 'usd'

        for simMeshInfo in self.args.simMesh:
            version, nodeName = simMeshInfo.split('=')
            nodeRet = rb.Coder().N.USD.ani.Decode(nodeName, 'SHOT')
            nsLayer = nodeRet.nslyr

            if self.args.onlyGroom:
                print '\t\t', simMeshRet
                simMeshRet['nslyr'] = nsLayer
                simMeshRet['nsver'] = version

                masterFileName = rb.Coder().F.USD.sim.MASTER.Encode(**simMeshRet)
                simMeshOutDir = rb.Coder().D.TASKNV.Encode(**simMeshRet)

                inputCache = utl.SJoin(simMeshOutDir, masterFileName)
                if os.path.exists(inputCache):
                    srclyr = utl.AsLayer(inputCache)
                    srcdata = srclyr.customLayerData
                    rigFile = srcdata.get('rigFile')
                    if not rigFile:
                        continue

                    outDir = self.args.outDir
                    configFile = Groom.GetConfigGroomFile(rigFile, outDir, nsLayer)
                    if configFile:
                        groomFile = configFile
                    else:
                        groomFile = Groom.GetGroomFile(rigFile, variant=srcdata.get('variant'))

                    if groomFile:
                        if groomFile.endswith('.mb'):
                            msg.debug('found Groom File')

                            simMeshRet['task'] = 'groom'
                            groomOutDir = rb.Coder().D.TASKN.Encode(**simMeshRet)
                            nsver = utl.GetNextVersion(groomOutDir)

                            # nsLayer, inputCache, groomFile, nsver, frameRange, step, user
                            BatchGroom.GroomSpool(inputCache, groomFile, nsver, (srcdata['start'], srcdata['end']),
                                                  self.args.step, self.args.user).spool()
                            print 'Batch Groom Spool'
                        elif groomFile.endswith('.hip'):
                            msg.debug('found Groom Hip File')

                            simMeshRet['task'] = 'groom'
                            groomOutDir = rb.Coder().D.TASKN.Encode(**simMeshRet)
                            nsver = utl.GetNextVersion(groomOutDir)

                            # nsLayer, inputCache, groomFile, nsver, frameRange, step, user
                            BatchGroom.GroomHoudiniSpool(inputCache, groomFile, nsver,
                                                         (srcdata['start'], srcdata['end']),
                                                         self.args.step, self.args.user).spool()
                            print 'Batch Groom Spool'
            else:
                subTask = TractorConfig.MakeTask('%s %s' % (nsLayer, version))
                cmd = taskCmd + ['--simMesh', '%s=%s' % (version, nodeName)]
                if 'maya-2022' in cmd:
                    cmd[cmd.index('maya-2022')] = 'maya-2018'
                    cmd[cmd.index('mayausd')] = 'usd_maya'
                    cmd.insert(cmd.index('usd_maya'), 'zelos')
                if self.args.groom:
                    cmd += ['--groom']

                subTask.addCommand(TractorConfig.MakeCommand(cmd + ['--process', 'geom']))
                parent.addCommand(TractorConfig.MakeCommand(cmd + ['--process', 'comp']))

                # self.dbTask(subTask, "ani", nsLayer, version, addInfo=self.addInfo)
                # meshJob.addChild(subTask)

                self.action += 1
                parent.parent.parent.comment += ' sim/{NS}/{VER}'.format(NS=nsLayer, VER=version)
                simJob.addChild(subTask)
        parent.addChild(simJob)

    # ---------------------------------------------------------------------------
    #
    #   ENVIRONMENT SET
    #
    # ---------------------------------------------------------------------------
    def LayoutTask(self, parent):
        layoutTask = TractorConfig.MakeTask('layout')

        msg.debug(self.args.layout)

        for layoutInfo in self.args.layout:
            ver, strnode = layoutInfo.split('=')
            ret = rb.Coder().N.USD.layout.Decode(strnode.split(',')[0])
            if ret.has_key('task'):
                node = ret['nslyr']
            else:
                node = 'extra'
            # ver, node = layoutInfo.split('=')
            subTask = TractorConfig.MakeTask('%s %s' % (node, ver))
            command = list(self.commonCommand)
            command += ['--layout', layoutInfo]

            subTask.addCommand(TractorConfig.MakeCommand(command + ['--process', 'geom']))
            parent.addCommand(TractorConfig.MakeCommand(command + ['--process', 'comp']))

            layoutTask.addChild(subTask)
            # addInfo = {"node": strnode}
            # addInfo.update(self.addInfo)
            # self.dbTask(task, "set", "set", ver, addInfo=addInfo)
            parent.parent.parent.comment += ' set/{VER}/{NODE}'.format(VER=ver, NODE=node)

        parent.addChild(layoutTask)
        self.action += 1

    # ---------------------------------------------------------------------------
    #
    #   ENVIRONMENT SET
    #
    # ---------------------------------------------------------------------------
    def CameraTask(self, parent):
        cameraTask = TractorConfig.MakeTask('camera')

        msg.debug(self.args.camera)

        for cameraInfo in self.args.camera:
            ver, node = cameraInfo.split('=')
            subTask = TractorConfig.MakeTask('%s %s' % ('cameras', ver))
            command = list(self.commonCommand)
            command += ['--camera', cameraInfo]
            subTask.addCommand(TractorConfig.MakeCommand(command + ['--process', 'geom']))
            parent.addCommand(TractorConfig.MakeCommand(command + ['--process', 'comp']))

            cameraTask.addChild(subTask)
            # addInfo = {"node": strnode}
            # addInfo.update(self.addInfo)
            # self.dbTask(task, "set", "set", ver, addInfo=addInfo)
            parent.parent.parent.comment += ' cam/{VER}'.format(VER=ver)

        self.action += 1
        parent.addChild(cameraTask)

    # ---------------------------------------------------------------------------
    #
    #   CROWD
    #
    # ---------------------------------------------------------------------------
    def CrowdTask(self, parent):
        crowdTask = TractorConfig.MakeTask('crowd')
        msg.debug(self.args.crowd)

        ver, null = self.args.crowd[0].split('=')

        if self.args.onlyGroom:
            groomAgents = []
            try:
                masterArgs = var.D.Decode(self.args.outDir)
                masterArgs.task = var.T.CROWD
                masterArgs.ver = ver
                masterPath = var.D.TASKV.Encode(**masterArgs)
                masterFile = var.F[var.T.CROWD].MASTER.Encode(**masterArgs)
                masterFile = utl.SJoin(masterPath, masterFile)
                masterLyr  = utl.AsLayer(masterFile)

                if not masterLyr:
                    raise valueerror('given file not exist (%s)'%masterFile)
            except Exception as e:
                msg.errmsg(e)
                msg.errmsg('failed to get crowd master file')
                return

            srcdata = masterLyr.customLayerData

            for grp in null.split(','):
                grp = grp.split(':')[0]
                groomAgents = BatchGroom.GetGroomAgents(masterFile, grp)
                for agent, groomScn in groomAgents:
                    if utl.NotExist(groomScn):
                        continue

                    if groomScn.endswith('.mb'):
                        # TODO: for maya hair publish
                        pass
                    elif groomScn.endswith('.hip'):
                        groomArg = masterArgs.Copy()
                        groomArg.task = var.T.GROOM
                        groomArg.nslyr = agent.split('/')[-1]
                        groomArg.pop('ver')

                        groomTaskPath = var.D.TASKN.Encode(**groomArg)
                        nsver = utl.GetNextVersion(groomTaskPath)

                        BatchGroom.GroomHoudiniSpool(
                            masterFile, groomScn, nsver,
                            (srcdata['start'], srcdata['end']),
                            self.args.step, self.args.user,
                            primPattern=agent
                        ).spool()
        else:
            command = list(self.commonCommand)
            command += ['--crowd', self.args.crowd[0]]
            subTask = TractorConfig.MakeTask('%s %s' % ('crowd', ver))
            subTask.addCommand(TractorConfig.MakeCommand(command + ['--process', 'geom']))
            parent.addCommand(TractorConfig.MakeCommand(command + ['--process', 'comp']))
            crowdTask.addChild(subTask)

            # ------------------------------------------------------------------
            # idx = command.index('--frameRange')
            # del command[idx:idx+2]
            #
            # for start, end in TractorConfig.GetIterFrames(args.frameRange):
            #     task = TractorConfig.MakeTask('frame %s-%s' % (start, end))
            #     cmd  = list(command)
            #     cmd += ['--exportRange', str(int(start)), str(int(end))]
            #     cmd += ['--crowd', self.args.crowd[0]]
            #     cmd += ['--progress', 'geom']
            #     task.addCommand(TractorConfig.MakeCommand(cmd))
            #     crowdTask.addChild(task)
            #
            # cmd = list(command)
            # cmd += ['--frameRange', str(args.frameRange[0]), str(args.frameRange[1])]
            # cmd += ['--progress', 'comp']
            # parent.addCommand(TractorConfig.MakeCommand(cmd))
            # ------------------------------------------------------------------

            parent.parent.parent.comment += ' crowd/{VER}'.format(VER=ver)

            self.action += 1
            parent.addChild(crowdTask)

    def GroomSimTask(self, parent):
        # HAIR SIMULATION
        print "HairSceneExport(args)"
        with open(self.args.file.replace('.mb', '.json'), 'r') as f:
            sceneData = json.load(f)
            start = sceneData['frameRange'][0]
            end = sceneData['frameRange'][1]
            print self.args, start, end

            for groomSimInfo in self.args.groomSim:
                ver, node = groomSimInfo.split('=')
                # subTask = TractorConfig.MakeTask('%s %s' % ('cameras', ver))
                # command = list(self.commonCommand)
                # command += ['--camera', cameraInfo]
                # subTask.addCommand(TractorConfig.MakeCommand(command + ['--process', 'geom']))
                # parent.addCommand(TractorConfig.MakeCommand(command + ['--process', 'comp']))
                #
                # cameraTask.addChild(subTask)
                # parent.parent.parent.comment += ' cam/{VER}'.format(VER=ver)

                BatchGroom.GroomSpool('', self.args.file, ver,
                                      frameRange=(start, end),
                                      step=args.step, user=args.user).spool()



def SceneExport(args):
    '''
    Export Cache in LocalPC
    :param args:
    :return:
    '''
    import maya.cmds as cmds

    plugins = ['pxrUsd', 'pxrUsdTranslators', 'AbcExport', 'DXUSD_Maya', 'backstageMenu']

    # unload plugins
    mutl.InitPlugins(plugins)

    # file load
    cmds.file(args.file, f=True, o=True)
    cmds.select(cl=True)

    coder = rb.Coder()

    ret = coder.D.Decode(args.outDir)

    showName = ret.show
    shotName = '{SEQ}_{SHOT}'.format(SEQ=ret.seq, SHOT=ret.shot)

    coll = DBConnection('WORK', showName)

    dbItem = {'action': 'export', 'filepath': args.file, 'user': args.user, 'time': datetime.datetime.now().isoformat(),
              'shot':shotName, 'task':'', 'enabled': False}

    # doIt Cache Export
    meshExportData = list() # because groom export using mesh cache file
    #---------------------------------------------------------------------------
    #
    # Mesh
    #
    #---------------------------------------------------------------------------
    if args.mesh:
        meshRet = dict(ret)
        meshRet['task'] = 'ani'
        meshRet['ext'] = 'usd'
        if args.onlyGroom:
            for meshInfo in args.mesh:
                version, nodeName = meshInfo.split('=')
                nodeRet = coder.N.USD.ani.Decode(nodeName, 'SHOT')

                nsLayer = nodeRet.nslyr
                meshRet['nslyr'] = nsLayer
                meshRet['nsver'] = version
                masterFileName = coder.F.USD.ani.MASTER.Encode(**meshRet)

                meshOutDir = coder.D.TASKNV.Encode(**meshRet)
                msg.debug('OUTDIR[ani]', ':', meshOutDir)
                msg.debug('FILENAME[ani]', ':', masterFileName)

                masterFilePath = utl.SJoin(meshOutDir, masterFileName)
                if os.path.exists(masterFilePath):
                    meshExportData.append(masterFilePath)
        else:
            for meshInfo in args.mesh:
                version, nodeName = meshInfo.split('=')
                masterFile = Rig.shotExport(node=nodeName, isRigUpdate=args.rigUpdate,
                                            show=ret.show, seq=ret.seq, shot=ret.shot, version=version, user=args.user,
                                            fr=args.frameRange, step=args.step, process=args.process)
                if masterFile and os.path.exists(masterFile) and args.process == 'comp':
                    meshExportData.append(masterFile)

    # ---------------------------------------------------------------------------
    #
    # Simulation Mesh
    #
    # ---------------------------------------------------------------------------
    if args.simMesh:
        simMeshRet = dict(ret)
        simMeshRet['task'] = 'sim'
        if args.onlyGroom:
            for simMeshInfo in args.simMesh:
                version, nodeName = simMeshInfo.split('=')
                nodeRet = coder.N.USD.sim.Decode(nodeName, 'SHOT')

                nsLayer = nodeRet.nslyr
                simMeshRet['nslyr'] = nsLayer
                simMeshRet['nsver'] = version
                masterFileName = coder.F.USD.sim.MASTER.Encode(**simMeshRet)

                simMeshOutDir = coder.D.TASKNV.Encode(**simMeshRet)
                msg.debug('OUTDIR[ani]', ':', simMeshOutDir)
                msg.debug('FILENAME[ani]', ':', masterFileName)

                masterFilePath = utl.SJoin(simMeshOutDir, masterFileName)
                if os.path.exists(masterFilePath):
                    meshExportData.append(masterFilePath)
        else:
            for simMeshInfo in args.simMesh:
                version, nodeName = simMeshInfo.split('=')
                nsLayer, nodeName = nodeName.split(':')
                masterFile = Sim.shotExport(node=nodeName, fr=args.frameRange, step=args.step,
                                            show=ret.show, seq=ret.seq, shot=ret.shot, version=version, user=args.user, process=args.process)

                if os.path.exists(masterFile) and args.process == 'comp':
                    meshExportData.append(masterFile)

    #---------------------------------------------------------------------------
    #
    #   ENVIRONMENT SET
    #
    #---------------------------------------------------------------------------
    if args.layout:
        for layoutInfo in args.layout:
            version, strnode = layoutInfo.split('=')
            Layout.shotExport(nodes=strnode.split(','), fr=args.frameRange, step=args.step,
                              show=ret.show, seq=ret.seq, shot=ret.shot, version=version, user=args.user, process=args.process)

    # ---------------------------------------------------------------------------
    #
    #   CAMERA
    #
    # ---------------------------------------------------------------------------
    if args.camera:
        for camInfo in args.camera:
            version, strnode = camInfo.split('=')
            Cam.cameraExport(dxNodes=strnode.split(','), fr=args.frameRange, step=args.step,
                             show=ret.show, seq=ret.seq, shot=ret.shot, version=version, user=args.user, process=args.process, insertDb=True)

    # ---------------------------------------------------------------------------
    #
    #   CROWD
    #
    # ---------------------------------------------------------------------------
    if args.crowd:
        isGolaem = False
        isMiarmy = False

        requests = os.environ['REZ_USED_REQUEST'].split()
        for r in requests:
            if r.startswith('golaem_maya'):
                isGolaem = True
                plugins.append('glmCrowd')
            if r.startswith('miarmy'):
                isMiarmy = True
                plugins.append('MiarmyProForMaya2018')
        mutl.InitPlugins(plugins)

        # crowd plugin loaded after import dxusd crowd
        import DXUSD_MAYA.Crowd as Crowd

        version, glmCaches = args.crowd[0].split('=')

        if isMiarmy:
            Crowd.shotExport_miarmy(show=ret.show, seq=ret.seq, shot=ret.shot, version=version,
                                    fr=args.frameRange, efr=args.exportRange, user=args.user, process=args.process)
        if isGolaem:
            Crowd.shotExport_golaem(show=ret.show, seq=ret.seq, shot=ret.shot, version=version,
                                    fr=args.frameRange, efr=args.exportRange, glmCaches=glmCaches, user=args.user, process=args.process)

    # ---------------------------------------------------------------------------
    #
    #   POST PROCESS
    #
    # ---------------------------------------------------------------------------
    if not meshExportData:
        return

    if not args.groom and not args.onlyGroom:
        return

    for inputCache in meshExportData:
        srclyr = utl.AsLayer(inputCache)
        srcdata = srclyr.customLayerData
        rigFile = srcdata.get('rigFile')
        if not rigFile:
            continue

        outDir = args.outDir
        configFile = Groom.GetConfigGroomFile(rigFile, outDir)
        if configFile:
            groomFile = configFile
        else:
            groomFile = Groom.GetGroomFile(rigFile, variant=srcdata.get('variant'))

        if groomFile:
            if args.host == 'local':
                groomArg = Groom.shotCacheMergeExport(inputCache, groomFile=groomFile, fr=args.frameRange,
                                                      step=args.step,
                                                      show=ret.show, seq=ret.seq, shot=ret.shot, user=args.user,
                                                      process='treat')
                if groomFile.endswith('.mb'):
                    Groom.shotCacheMergeExport(inputCache, groomFile=groomFile, fr=args.frameRange, step=args.step,
                                           show=ret.show, seq=ret.seq, shot=ret.shot, user=args.user, process='both', version=groomArg.nsver)
                elif groomFile.endswith('.hip'):
                    BatchGroom.GroomHoudiniSpool(inputCache, groomFile, groomArg.nsver,
                                                 frameRange=(srcdata['start'], srcdata['end']),
                                                 step=args.step, user=args.user).spool()
            else:
                groomArg = Groom.shotCacheMergeExport(inputCache, groomFile=groomFile, fr=args.frameRange, step=args.step,
                                           show=ret.show, seq=ret.seq, shot=ret.shot, user=args.user, process='treat')

                if groomFile.endswith('.mb'):
                    BatchGroom.GroomSpool(inputCache, groomFile, groomArg.nsver,
                                          frameRange=(srcdata['start'], srcdata['end']),
                                          step=args.step, user=args.user).spool()
                elif groomFile.endswith('.hip'):
                    # nsLayer, inputCache, groomFile, nsver, frameRange, step, user
                    BatchGroom.GroomHoudiniSpool(inputCache, groomFile, groomArg.nsver,
                                                 frameRange=(srcdata['start'], srcdata['end']),
                                                 step=args.step, user=args.user).spool()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='DXUSD_MAYA Batch script.'
    )

    parser.add_argument('-f', '--file', type=str, help='Maya filename.')
    parser.add_argument('-o', '--outDir', type=str, help='Cache out directory.')
    parser.add_argument('-u', '--user', type=str, help='User name')

    # TimeRange argument
    parser.add_argument('-fr',  '--frameRange', type=int, nargs=2, default=[0, 0], help='frame range, (start, end)')
    parser.add_argument('-efr', '--exportRange', type=int, nargs=2, default=[0, 0], help='export frame range, (start, end)')
    # parser.add_argument('-fs',  '--frameSample', type=float, default=1.0, help='frame step size default = 1.0')
    parser.add_argument('-s',   '--step', type=float, default=1.0, help='frame step size default = 1.0')

    # Acting argument
    parser.add_argument('-p', '--process', type=str, choices=['both', 'geom', 'comp'], help='task export when choice process, [geom, comp]')
    parser.add_argument('-hs', '--host', type=str, choices=['local', 'spool', 'tractor'], help='if host local, cache export. other option is "spool" spool')

    # task argument
    #   Rig Out
    parser.add_argument('-m', '--mesh', type=str, nargs='*',
                        help='export namspace:nodeName of dxRigNode \nex) --mesh v005=nsLayer1:node_rig_GRP v004=nsLayer2:node_rig_GRP')
    parser.add_argument('-ru', '--rigUpdate', action='store_true', default=False, help='if True, using rig latest version.')
    #   Sim Out
    parser.add_argument('-sm', '--simMesh', type=str, nargs='*',
                        help='export namspace:nodeName of dxRigNode \nex) --simMesh v005=nsLayer1:node_rig_GRP v004=nsLayer2:node_rig_GRP')
    #   Layout Out
    parser.add_argument('-l', '--layout', type=str, nargs='*',
                        help='export nodeName of dxBlock \nex) --layout v005=node_set v004=node1_set')
    #   Camera Out
    parser.add_argument('-c', '--camera', type=str, nargs='*',
                        help='export nodeName of dxCamera \nex) --camera v005=dxCamera1 v004=dxCamera2')
    #   Crowd Out
    parser.add_argument('-cw', '--crowd', type=str, nargs='*',
                        help='export nodeName of crowdField \nex) --crowd v005=cacheProxyShape1:crowdField1')
    #   Groom Out
    parser.add_argument('-g', '--groom', action='count', default=0, help='if groom, export groom after exported mesh')
    parser.add_argument('-og', '--onlyGroom', action='count', default=0, help='if groom, export groom after exported mesh')
    parser.add_argument('-gs', '--groomSim', type=str, nargs='*', help='export Groom of already simulation geoCache')

    args, unknown = parser.parse_known_args(sys.argv)

    print '# DEBUG :', args.file

    print '>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'
    print args

    if not args.file:
        sys.exit(1)

    if not args.outDir:
        flags = rb.Flags(pub='_3d')
        flags.D.SetDecode(os.path.dirname(args.file), 'ROOTS')
        flags.F.MAYA.SetDecode(os.path.basename(args.file), 'BASE')
        args.outDir = flags.D.SHOT

    if args.host == 'spool':
        job = SceneSpool(args).doIt()
    else:
        from pymel.all import *
        try:
            SceneExport(args)
            print "Local Export Exit"
        except Exception as e:
            errorStr = traceback.format_exc()
            msg.errmsg(errorStr)

            if cmds.about(batch=True):
                OpenMaya.MGlobal.displayError(errorStr)
                cmd = ['/backstage/dcc/DCC', 'rez-env', 'rocketchattoolkit', '--']
                cmd += ['BotMsg', '--artist',  args.user]
                cmd += ['--message', '\"%s\"' % errorStr]
                cmd += ['--bot', 'BadBot']
                # print 'Cmd:', ' '.join(cmd)
                os.system(' '.join(cmd))
                os._exit(1)
            else:
                cmds.error(errorStr)

    # quit
    sys.exit(0)
