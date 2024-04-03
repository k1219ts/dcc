import sys, os, site, datetime, json, glob

import dxConfig
from pymongo import MongoClient

# tractor api setup
dxConfig.getConf("TRACTOR_API")
site.addsitedir( dxConfig.getConf("TRACTOR_API") )
import tractor.api.author as author
TRACTOR_IP = dxConfig.getConf("TRACTOR_CACHE_IP")# '10.0.0.25'
# TRACTOR_IP = "10.0.0.71"

# BATCHSCENESCRIPT = '/backstage/bin/DCC rez-env {REZ_RESOLVED} -- mayapy {SCRIPTPATH}/apps/Maya/toolkits/dxsUsd/BatchScene.py'
DCC_PATH = '/backstage/bin/DCC'
TOOLKIT_PATH = os.getenv('REZ_MAYA_TOOLKIT_ROOT')
DXSUSD_DIR_PATH = os.path.join(TOOLKIT_PATH, 'scripts', 'dxsUsd')

DB_IP = dxConfig.getConf("DB_IP")
dbName = "WORK"

import BatchZenn

class SceneSpool:
    def __init__(self, opts):
        self.opts = opts
        self.serviceKey = 'USER||Cache'
        self.maxactive = 24
        self.projects  = 'export'
        self.tier  = 'cache'
        self.tags  = 'GPU'
        self.envkey= ''

        command = [DCC_PATH, 'rez-env']
        command += os.environ['REZ_RESOLVE'].split(' ')
        command += ['--', 'mayapy', os.path.join(DXSUSD_DIR_PATH, 'BatchScene.py')]
        command+= ['--srcfile', opts.srcfile]
        command+= ['--outdir', opts.outdir]
        command+= ['--fr', opts.frameRange[0], opts.frameRange[1]]
        command+= ['--step', opts.step]
        command+= ['--user', opts.user]
        command+= ['--host', 'tractor']
        self.defaultCommand = command

        self.addInfo = {'comment':'empty'}
        if os.environ['REZ_USED_RESOLVE']:
            self.addInfo['REZ_USED_RESOLVE'] = os.environ['REZ_USED_RESOLVE'].replace(' ', ',')

        self.dbInformationParser()

        self.action = 0

    def dbInformationParser(self):
        splitOutDir = opts.outdir.split("/")

        if "show" in splitOutDir:
            self.showDir = splitOutDir[splitOutDir.index("show") + 1]

        if "shot" in splitOutDir:
            self.shotName = splitOutDir[splitOutDir.index("shot") + 2]

    def doIt(self):
        job = author.Job()
        job.title    = '(USD) ' + str(os.path.basename(self.opts.srcfile))
        job.comment  = 'outdir: %s' % str(self.opts.outdir)
        job.metadata = 'scene: %s' % str(self.opts.srcfile)
        job.envkey   = [self.envkey]
        job.service  = self.serviceKey
        job.maxactive= self.maxactive
        job.tier = self.tier
        job.tags = [self.tags]
        job.projects = [self.projects]

        JobTask = author.Task(title='Job')
        JobTask.serialsubtasks = 1
        job.addChild(JobTask)

        showDir, seqName, shotName = batchCommon.PathParser(self.opts.outdir)
        showName = showDir.split('/')[-1]
        showName = showName.replace('_pub', '')

        client = MongoClient(DB_IP)
        db = client[dbName]
        coll = db[showName]

        dbItem = {"action": "export",
                  "filepath": opts.srcfile,
                  "user": opts.user,
                  "time": datetime.datetime.today().isoformat(),
                  "shot": shotName,
                  "task": "ani",
                  "enabled": False}
        objId = coll.insert_one(dbItem).inserted_id

        command = ["/backstage/apps/inventory/enableDBRecord.py", dbName, showName, objId]
        JobTask.addCommand(author.Command(argv=command, service=self.serviceKey))

        command = [DCC_PATH, "rez-env", "python-2", "--", "python", os.path.join(DXSUSD_DIR_PATH, "MsgSender.py"),
                   showName, str(job.title), opts.user]
        JobTask.addCommand(author.Command(argv=command, service=self.serviceKey))

        exportJob = author.Task(title='USD export')
        exportJob.serialsubtasks = 0
        JobTask.addChild(exportJob)

        if self.opts.mesh or self.opts.allexport:
            self.MeshTask(exportJob)
        if self.opts.simmesh or self.opts.allexport:
            self.SimTask(exportJob)
        if self.opts.hairSim or self.opts.allexport:
            self.HairTask(exportJob)
        if self.opts.crowd or self.opts.allexport:
            self.CrowdTask(exportJob)
        if self.opts.layout or self.opts.allexport:
            self.LayoutTask(exportJob)
        if self.opts.camera or self.opts.allexport:
            self.CameraTask(exportJob)

        # job.paused = True
        job = job if self.action > 0 else None
        return job

    def dbTask(self, task, type, name, ver, addInfo = {}):
        dbCmd = [DCC_PATH, 'rez-env', 'python-2', '--', 'python', os.path.join(DXSUSD_DIR_PATH, 'DBQuery.py')]
        dbCmd += ['--user', opts.user]
        outDir = self.opts.outdir
        if type == "cam" or type == "set" or type == 'crowd':
            outDir = os.path.join(self.opts.outdir, name, ver)
        else:
            outDir = os.path.join(self.opts.outdir, type, name, ver)
        dbCmd += ["--showDir", self.showDir, "--shot", self.shotName, "--name", name, "--version", ver,
                  "--type", type, "--outDir", outDir]
        if addInfo:
            for key in addInfo.keys():
                dbCmd += [key, addInfo[key]]
        task.addCommand(author.Command(argv=dbCmd, service=self.serviceKey))


    #---------------------------------------------------------------------------
    #
    #   MESH
    #
    #---------------------------------------------------------------------------
    def MeshTask(self, parent):
        meshJob = author.Task(title='mesh')
        taskCmd = list(self.defaultCommand)
        if self.opts.allexport:
            taskCmd += ['--mesh', '_all']
            meshJob.addCommand(author.Command(argv=taskCmd, service=self.serviceKey))
        else:
            self.opts.mesh = self.opts.mesh.replace('"', '')
            for vsrc in self.opts.mesh.split(';'):
                ver, strnode = vsrc.split('=')
                for n in strnode.split(','):
                    nsName, nodeName = n.split(':')
                    if self.opts.onlyzenn:
                        masterFile = '{DIR}/ani/{NS}/{VER}/{NS}.usd'.format(DIR=self.opts.outdir, NS=nsName, VER=ver)
                        BatchZenn.MakeCacheMergeScene(masterFile, fr=self.opts.frameRange, step=self.opts.step, user=self.opts.user).spool()
                    else:
                        subTask = author.Task(title='%s %s' % (nsName, ver))
                        cmd = taskCmd + ['--mesh', '%s=%s' % (ver, n)]
                        if self.opts.zenn:
                            cmd += ['--zenn']
                        if self.opts.rigUpdate:
                            cmd += ['--rigUpdate']
                        print cmd
                        subTask.addCommand(author.Command(argv=cmd, service=self.serviceKey))

                        self.dbTask(subTask, "ani", nsName, ver, addInfo=self.addInfo)
                        meshJob.addChild(subTask)

                        self.action += 1
                        parent.parent.parent.comment += ' ani/{NS}/{VER}'.format(NS=nsName, VER=ver)
        parent.addChild(meshJob)

    #---------------------------------------------------------------------------
    #
    #   SIMULATION MESH
    #
    #---------------------------------------------------------------------------
    def SimTask(self, parent):
        simtask = author.Task(title='sim')
        taskcmd = list(self.defaultCommand)
        if self.opts.allexport:
            taskcmd += ['--sim', '_all']
            simtask.addCommand(author.Command(argv=taskcmd, service=self.serviceKey))
        else:
            self.opts.simmesh = self.opts.simmesh.replace('"', '')
            for vsrc in self.opts.simmesh.split(';'):
                ver, strnode = vsrc.split('=')
                for n in strnode.split(','):
                    nsName, nodeName = n.split(':')
                    if opts.onlyzenn:
                        masterFile = '{DIR}/sim/{NS}/{VER}/{NS}.usd'.format(DIR=self.opts.outdir, NS=nsName, VER=ver)
                        BatchZenn.MakeCacheMergeScene(masterFile, fr=self.opts.frameRange, step=self.opts.step, user=self.opts.user).spool()
                    else:
                        subtask = author.Task(title='%s %s' % (nsName, ver))
                        taskcmd = list(self.defaultCommand) + ['--sim', '%s=%s' % (ver, n)]
                        if self.opts.zenn:
                            taskcmd += ['--zenn']
                        subtask.addCommand(author.Command(argv=taskcmd, service=self.serviceKey))

                        self.dbTask(subtask, "sim", nsName, ver, addInfo=self.addInfo)
                        simtask.addChild(subtask)

                        self.action += 1
                        parent.parent.parent.comment += ' sim/{NS}/{VER}'.format(NS=nsName, VER=ver)
        parent.addChild(simtask)

    #---------------------------------------------------------------------------
    #
    #   HAIR SIMULATION
    #
    #---------------------------------------------------------------------------
    def HairTask(self, parent):
        hairCommand = [DCC_PATH, 'rez-env']
        hairCommand += os.environ['REZ_RESOLVE'].split(' ')
        hairCommand += ['--', 'mayapy', os.path.join(DXSUSD_DIR_PATH, 'BatchZenn.py')]
        hairCommand += ['--zennFile', self.opts.srcfile]
        hairCommand += ['--step', self.opts.step]

        zennNodes = list()
        frameRange = self.opts.frameRange
        if not frameRange[0] and not frameRange[1]:
            mayaFile = batchCommon.GetMayaFilename(self.opts.srcfile)
            jsonFile = mayaFile.replace('.mb', '.json')
            if not os.path.exists(jsonFile):
                assert False, '# msg : not found json file.'
            with open(jsonFile, 'r') as f:
                sceneGraph = json.load(f)
                if sceneGraph.has_key('frameRange'):
                    frameRange = sceneGraph['frameRange']
                if sceneGraph.has_key('zenn'):
                    for z in sceneGraph['zenn']:
                        zennNodes.append(z[0])

        self.opts.hairSim = self.opts.hairSim.replace('"', '')
        for vsrc in self.opts.hairSim.split(';'):
            ver, strnode = vsrc.split('=')
            for n in strnode.split(','):
                nsName, nodeName = n.split(':')

                if self.opts.serial:
                    ztask = author.Task(title='zenn export - %s' % nsName)
                    command = list(hairCommand)
                    command += ['--outdir', os.path.join(self.opts.outdir, 'zenn', nsName)]
                    command += ['--version', ver]
                    command += ['--fr', int(frameRange[0]), int(frameRange[1])]
                    command += ['--task', 'none']
                    ztask.addCommand(author.Command(argv=command, service=self.serviceKey))

                    self.dbTask(ztask, "zenn", nsName, ver, addInfo=self.addInfo)
                    parent.addChild(ztask)
                else:
                    ztask = author.Task(title='zenn')
                    ztask.serialsubtasks = 1
                    parent.addChild(ztask)

                    gtask = author.Task(title='geom')
                    ztask.addChild(gtask)

                    if zennNodes:
                        ptask = author.Task(title='payload - %s' % nsName)
                        ztask.addChild(ptask)
                        for zn in zennNodes:
                            task = author.Task(title=str(zn))
                            command = list(hairCommand)
                            command+= ['--outdir', os.path.join(self.opts.outdir, 'zenn', nsName)]
                            command+= ['--version', ver]
                            command+= ['--fr', int(frameRange[0]), int(frameRange[1])]
                            command+= ['--zennNode', zn]
                            command+= ['--task', 'payload']
                            task.addCommand(author.Command(argv=command, service=self.serviceKey))
                            ptask.addChild(task)

                        self.dbTask(ptask, "zenn", nsName, ver, addInfo=self.addInfo)
                    else:
                        ztask.title = 'zenn payload - %s' % nsName
                        command = list(hairCommand)
                        command+= ['--outdir', os.path.join(self.opts.outdir, 'zenn', nsName)]
                        command+= ['--version', ver]
                        command+= ['--fr', int(frameRange[0]), int(frameRange[1])]
                        command+= ['--task', 'payload']
                        ztask.addCommand(author.Command(argv=command, service=self.serviceKey))

                        self.dbTask(ztask, "zenn", nsName, ver, addInfo=self.addInfo)

                    for start, end in BatchZenn.GetIterFrames(frameRange):
                        ftask = author.Task(title='frame %s-%s' % (start, end))
                        command = list(hairCommand)
                        command+= ['--outdir', os.path.join(self.opts.outdir, 'zenn', nsName)]
                        command+= ['--version', ver]
                        command+= ['--fr', int(start), int(end)]
                        command+= ['--task', 'geom']
                        ftask.addCommand(author.Command(argv=command, service=self.serviceKey))
                        gtask.addChild(ftask)

                self.action += 1
                parent.parent.parent.comment += ' zenn/{NS}/{VER}'.format(NS=nsName, VER=ver)



    #---------------------------------------------------------------------------
    #
    #   CROWD
    #
    #---------------------------------------------------------------------------
    def CrowdTask(self, parent):
        crowdCommand = [DCC_PATH, 'rez-env']
        crowdCommand += os.environ['REZ_RESOLVE'].split(' ')
        crowdCommand += ['--', 'mayapy']

        isSimulation = 0
        frameRange   = self.opts.frameRange
        if not frameRange[0] and not frameRange[1]:
            mayaFile = batchCommon.GetMayaFilename(self.opts.srcfile)
            jsonFile = mayaFile.replace('.mb', '.json')
            if not os.path.exists(jsonFile):
                assert False, '# msg : not found json file.'
            with open(jsonFile, 'r') as f:
                sceneGraph = json.load(f)
                crowdInfo  = sceneGraph['crowd'][0]
                isSimulation= crowdInfo[1]
                if sceneGraph.has_key('frameRange'):
                    frameRange = sceneGraph['frameRange']
                    frameRange = (int(frameRange[0]), int(frameRange[1]))
                # miarmy version
                if sceneGraph.has_key('miarmyVersion'):
                    crowdCommand += ['-mv', sceneGraph['miarmyVersion']]

        crowdCommand += [os.path.join(DXSUSD_DIR_PATH, 'BatchCrowd.py')]

        self.opts.crowd = self.opts.crowd.replace('"', '')
        for vsrc in self.opts.crowd.split(';'):
            ver, strnode = vsrc.split('=')
            outdir = os.path.join(self.opts.outdir, 'crowd', ver)

            # Only Bake
            if self.opts.crowdbake:
                ptask = author.Task(title='crowd payload %s-%s' % (frameRange[0], frameRange[1]))
                parent.addChild(ptask)

                skelFiles = glob.glob(outdir + '/crowd.skel.[0-9]*.usd')
                skelFiles.sort()

                for sf in skelFiles:
                    command = list(crowdCommand)
                    command+= ['--srcfile', sf, '--onlybake']
                    ftask = author.Task(title=sf.split('.')[-2])
                    ftask.addCommand(author.Command(argv=command, service=self.serviceKey))
                    ptask.addChild(ftask)

                self.action += 1
                parent.parent.parent.comment += ' crowd/{VER}-SkinBake'.format(VER=ver)

            # Miarmy MayaScene
            else:
                ptask = author.Task(title='crowd payload %s-%s' % (int(frameRange[0]), int(frameRange[1])))
                command = list(crowdCommand)
                command+= ['--srcfile', self.opts.srcfile]
                command+= ['--outdir', outdir, '--fr', frameRange[0], frameRange[1], '--task', 'payload']
                ptask.addCommand(author.Command(argv=command, service=self.serviceKey))

                self.dbTask(ptask, "crowd", "crowd", ver, addInfo=self.addInfo)
                parent.addChild(ptask)

                geomparent = ptask

                # meshdrive
                if isSimulation:
                    ptask.serialsubtasks = 0
                    mdtask  = author.Task(title='meshdrive')
                    command = list(crowdCommand)
                    command+= ['--srcfile', self.opts.srcfile]
                    command+= ['--meshdrive']
                    mdtask.addCommand(author.Command(argv=command, service=self.serviceKey))
                    ptask.addChild(mdtask)
                    geotask = author.Task(title='geom')
                    ptask.addChild(geotask)
                    geomparent = geotask

                # geometry frame
                for start, end in BatchZenn.GetIterFrames(frameRange):
                    ftask = author.Task(title='frame %s-%s' % (start, end))
                    command = list(crowdCommand)
                    command+= ['--srcfile', self.opts.srcfile]
                    command+= ['--outdir', outdir, '--expfr', start, end, '--task', 'geom']
                    ftask.addCommand(author.Command(argv=command, service=self.serviceKey))
                    geomparent.addChild(ftask)

                self.action += 1
                parent.parent.parent.comment += ' crowd/{VER}'.format(VER=ver)

    #---------------------------------------------------------------------------
    #
    #   ENVIRONMENT SET
    #
    #---------------------------------------------------------------------------
    def LayoutTask(self, parent):
        task = author.Task(title='set')
        self.opts.layout = self.opts.layout.replace('"', '')
        command = list(self.defaultCommand)
        if self.opts.allexport:
            command += ['--layout', '_all']
        else:
            command += ['--layout', '%s' % self.opts.layout]
        task.addCommand(author.Command(argv=command, service=self.serviceKey))

        for vsrc in opts.layout.split(';'):
            ver, strnode = vsrc.split('=')
            addInfo = {"node": strnode}
            addInfo.update(self.addInfo)
            self.dbTask(task, "set", "set", ver, addInfo=addInfo)

        parent.addChild(task)
        self.action += 1

    #---------------------------------------------------------------------------
    #
    #   CAMERA
    #
    #---------------------------------------------------------------------------
    def CameraTask(self, parent):
        self.opts.camera = self.opts.camera.replace('"', '')
        ver = self.opts.camera.split(';')[0].split('=')[0]
        camJob = author.Task(title='camera %s' % ver)
        taskCmd= list(self.defaultCommand)
        if self.opts.allexport:
            taskCmd += ['--camera', '_all']
        else:
            taskCmd += ['--camera', '%s' % self.opts.camera]
        camJob.addCommand(author.Command(argv=taskCmd, service=self.serviceKey))

        self.dbTask(camJob, "cam", "cam", ver, addInfo=self.addInfo)
        parent.addChild(camJob)

        self.action += 1
        parent.parent.parent.comment += ' cam/{VER}'.format(VER=ver)




def SceneExport(opts, local=True):
    import batchCommon
    import Rig
    import Sim
    import Zenn
    import Crowd
    import Camera
    import EnvSet

    showDir, seqName, shotName = batchCommon.PathParser(opts.outdir)
    showName = showDir.split('/')[-1]
    showName = showName.replace('_pub', '')
    # print showDir, seqName, shotName

    client = MongoClient(DB_IP)
    db = client[dbName]
    coll = db[showName]

    dbItem = {"action": "export",
              "filepath": opts.srcfile,
              "user": opts.user,
              "time": datetime.datetime.today().isoformat(),
              "shot": shotName,
              "task": "ani",
              "enabled": False}
    objId = coll.insert_one(dbItem).inserted_id

    if opts.allexport:
        opts.mesh    = '_all'
        opts.simmesh = '_all'
        opts.layout  = '_all'
        opts.camera  = '_all'

    meshExportData = list()
    #---------------------------------------------------------------------------
    #
    #   MESH
    #
    #---------------------------------------------------------------------------
    if opts.mesh:
        meshOutDir = os.path.join(opts.outdir, 'ani')
        if opts.onlyzenn:
            if opts.mesh == '_all':
                pass
            else:
                for vsrc in opts.mesh.split(';'):
                    ver, strnode = vsrc.split('=')
                    for n in strnode.split(','):
                        nsName, nodeName = n.split(':')
                        masterPayloadFile = '{DIR}/{NS}/{VER}/{NS}.payload.usd'.format(DIR=meshOutDir, NS=nsName, VER=ver)
                        if os.path.exists(masterPayloadFile):
                            meshExportData.append(masterPayloadFile)
        else:
            if opts.mesh == '_all':
                exportNodes = Rig.GetRigNodes()
                if exportNodes:
                    for n in exportNodes:
                        masterPayloadFile = Rig.RigShotExport(node=n, rigUpdate=opts.rigUpdate, outDir=meshOutDir, fr=opts.frameRange, step=opts.step, user=opts.user).doIt()
                        meshExportData.append(masterPayloadFile)
            else:
                for vsrc in opts.mesh.split(';'):
                    ver, strnode = vsrc.split('=')
                    for n in strnode.split(','):
                        masterPayloadFile = Rig.RigShotExport(node=n, rigUpdate=opts.rigUpdate, outDir=meshOutDir, fr=opts.frameRange, step=opts.step, version=ver, user=opts.user).doIt()
                        meshExportData.append(masterPayloadFile)

    #---------------------------------------------------------------------------
    #
    #   SIMULATION MESH
    #
    #---------------------------------------------------------------------------
    if opts.simmesh:
        simOutDir = os.path.join(opts.outdir, 'sim')
        if opts.onlyzenn:
            if opts.simmesh == '_all':
                pass
            else:
                for vsrc in opts.simmesh.split(';'):
                    ver, strnode = vsrc.split('=')
                    for n in strnode.split(','):
                        nsName, nodeName = n.split(':')
                        masterPayloadFile = '{DIR}/{NS}/{VER}/{NS}.payload.usd'.format(DIR=simOutDir, NS=nsName, VER=ver)
                        if os.path.exists(masterPayloadFile):
                            meshExportData.append(masterPayloadFile)
        else:
            if opts.simmesh == '_all':
                exportNodes = Sim.GetSimNodes()
                if exportNodes:
                    for n, lyr in exportNodes:
                        masterPayloadFile = Sim.SimExport(node=n, outDir=simOutDir, fr=opts.frameRange, step=opts.step, user=opts.user).doIt()
                        meshExportData.append(masterPayloadFile)
            else:
                for vsrc in opts.simmesh.split(';'):
                    ver, strnode = vsrc.split('=')
                    for n in strnode.split(','):
                        masterPayloadFile = Sim.SimExport(node=n, outDir=simOutDir, fr=opts.frameRange, step=opts.step, version=ver, user=opts.user).doIt()
                        meshExportData.append(masterPayloadFile)

    #---------------------------------------------------------------------------
    #
    #   ENVIRONMENT SET
    #
    #---------------------------------------------------------------------------
    if opts.layout:
        setOutDir = os.path.join(opts.outdir, 'set')
        if opts.layout == '_all':
            exportNodes = EnvSet.GetSetNodes()
            for n in exportNodes:
                EnvSet.SetShotExport(node=n, outDir=setOutDir, fr=opts.frameRange, step=opts.step, user=opts.user).doIt()
        else:
            for vsrc in opts.layout.split(';'):
                ver, strnode = vsrc.split('=')
                for n in strnode.split(','):
                    EnvSet.SetShotExport(node=n, outDir=setOutDir, fr=opts.frameRange, step=opts.step, version=ver, user=opts.user).doIt()

    #---------------------------------------------------------------------------
    #
    #   CAMERA
    #
    #---------------------------------------------------------------------------
    if opts.camera:
        camOutDir = os.path.join(opts.outdir, 'cam')
        if opts.camera == '_all':
            exportNodes = Camera.GetCameraNodes()
            for n in exportNodes:
                Camera.CameraShotExport(node=n, outDir=camOutDir, fr=opts.frameRange, step=opts.step, user=opts.user).doIt()
        else:
            for vsrc in opts.camera.split(';'):
                ver, strnode = vsrc.split('=')
                for rn in strnode.split(','):
                    exportNodes = Camera.GetCameraNodes(rn)
                    for n in exportNodes:
                        Camera.CameraShotExport(node=n, outDir=camOutDir, fr=opts.frameRange, step=opts.step, version=ver, user=opts.user).doIt()

    #---------------------------------------------------------------------------
    #
    #   CROWD
    #
    #---------------------------------------------------------------------------
    if opts.crowd:
        isSimulation = 0
        frameRange = opts.frameRange
        miarmyVersion = "6.5.21"
        if not frameRange[0] and not frameRange[1]:
            mayaFile = batchCommon.GetMayaFilename(opts.srcfile)
            jsonFile = mayaFile.replace(".mb", ".json")
            if not os.path.exists(jsonFile):
                assert False, "# msg : not found json file"
            with open(jsonFile, "r") as f:
                sceneGraph = json.load(f)
                crowdInfo = sceneGraph['crowd'][0]
                isSimulation = crowdInfo[1]
                if sceneGraph.has_key('frameRange'):
                    frameRange = sceneGraph['frameRange']
                    frameRange = (int(frameRange[0]), int(frameRange[1]))
                # miarmy version
                if sceneGraph.has_key('miarmyVersion'):
                    miarmyVersion = sceneGraph['miarmyVersion']

        for vsrc in opts.crowd.split(";"):
            ver, strnode = vsrc.split('=')
            outdir = os.path.join(opts.outdir, "crowd", ver)

            if opts.crowdbake:
                Crowd.UsdSkelBakeOnly(outdir).doIt()
            else:
                # if isSimulation:
                #     import Miarmy
                #     Miarmy.MiarmyBatchMeshDriveExport(opts.srcfile)
                Crowd.CrowdShotExport(task="geom", sceneFile=opts.srcfile, fr=opts.frameRange, outDir=outdir, version=ver).doIt()
                Crowd.CrowdShotExport(task="payload", sceneFile=opts.srcfile, fr=opts.frameRange, outDir=outdir, version=ver).doIt()

    #---------------------------------------------------------------------------
    #
    #   POST PROCESS
    #
    #---------------------------------------------------------------------------
    if not meshExportData:
        return
    if not opts.zenn and not opts.onlyzenn:
        return
    # print '[Debug] PostProcess'
    # print meshExportData
    import BatchZenn
    for payloadFile in meshExportData:
        filename = payloadFile.replace('.payload.usd', '.usd')
        if local:
            BatchZenn.MakeCacheMergeScene(filename, fr=opts.frameRange, step=opts.step, user=opts.user).doIt()
        else:
            BatchZenn.MakeCacheMergeScene(filename, fr=opts.frameRange, step=opts.step, user=opts.user).spool()

    command = "/backstage/apps/inventory/enableDBRecord.py {0} {1} {2}".format(dbName, showName, objId)
    os.system(command)

#---------------------------------------------------------------------------
#
#   HAIR SIMULATION
#
#---------------------------------------------------------------------------
def HairSceneExport(opts):
    hairSimOutDir = os.path.join(opts.outdir, 'zenn')
    for vsrc in opts.hairSim.split(';'):
        ver, strnode = vsrc.split('=')
        for n in strnode.split(','):
            nsName, nodeName = n.split(':')
            outDir = os.path.join(hairSimOutDir, nsName)
            BatchZenn.SceneExport(opts.srcfile, outDir, ver, opts.frameRange, opts.step, task='')



#---------------------------------------------------------------------------
#
#   MAIN
#
#---------------------------------------------------------------------------
if __name__ == '__main__':
    import batchCommon
    optparser = batchCommon.sceneOptParserSetup()
    opts, args= optparser.parse_args(sys.argv)
    if not opts.srcfile or not opts.outdir:
        os._exit(0)

    if opts.host == 'spool':
        job = SceneSpool(opts).doIt()
        if not job:
            os._exit(0)
        job.priority = 100
        author.setEngineClientParam(hostname=TRACTOR_IP, port=80, user=opts.user, debug=True)
        job.spool()
        print job.asTcl()
        author.closeEngineClient()
    else:
        from pymel.all import *
        plugins = [
            'backstageMenu', 'pxrUsd', 'pxrUsdTranslators', 'AbcExport'
        ]
        batchCommon.InitPlugins(plugins)
        # HAIR SIMULATION
        if opts.hairSim:
            HairSceneExport(opts)

        # elif opts.crowd:
        #     sys.stderr.write("# ERROR : Not support process. using 'BatchCrowd.py'")

        else:
            # open file
            cmds.file(opts.srcfile, force=True, open=True)
            cmds.select(cl=True)
            local = True if opts.host == 'local' else False
            SceneExport(opts, local)

    # quit
    os._exit(0)
