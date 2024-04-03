from Katana import NodegraphAPI, Nodes3DAPI

import os, sys
import configobj
import string
import datetime
import getpass
import shutil
import site

import dxConfig
site.addsitedir(dxConfig.getConf('TRACTOR_API'))

import tractor.api.author as author
from TractorEngine import TractorEngine


def GetOutputDefineNode(renderNode):
    rootProducer = Nodes3DAPI.GetGeometryProducer(renderNode)
    producer = rootProducer.getProducerByPath('/root')
    node = NodegraphAPI.GetNode(producer.getAttribute('OutputDefine').getValue())
    return node


class JobMain:
    def __init__(self, katanaFile, sceneFrameRange, renderSettings, onlyDenoise=False, localQueue=False, makeMov=False):
        self.katanaFile = katanaFile
        self.renderFile = self.katanaFile
        self.sceneFrameRange = sceneFrameRange
        self.renderSettings  = renderSettings
        self.makeMov = makeMov
        self.nodeSettings= list()
        self.onlyDenoise = onlyDenoise
        self.localQueue  = localQueue

        self.getTractorConfig()             # self.tr_ engine, port, tier, priority, maxactive, tags
        self.getGraphStateVariablesName()   # result : self.GSV
        self.initRenderSettings()

        # REZ OPTIONS
        self.m_prman_ver = None
        self.m_rezopts = os.getenv('REZ_USED_RESOLVE').split()
        for p in self.m_rezopts:
            if 'renderman-' in p:
                self.m_prman_ver = p.split('-')[-1]

    def initRenderSettings(self):
        def maxDigit(rs):
            specified = rs.customFarmSettings['Specified.frames']
            if specified:
                src = list()
                for i in specified.split(','):
                    src.append(int(i.split('-')[-1]))
                src.sort()
                return len(str(src[-1]))
            if rs.frameRange:
                return len(str(int(rs.frameRange[1])))
            else:
                return len(str(self.sceneFrameRange['end']))

        for rs in self.renderSettings:
            if rs.outputs:
                digit = maxDigit(rs)
                if digit > 4:
                    pad = '%0' + str(digit) + 'd'
                    new_outputs = list()
                    for out in rs.outputs:
                        location = out['outputLocation']
                        out['outputLocation'] = location.replace('%04d', pad)
                        new_outputs.append(out)
                    rs.outputs = new_outputs
                    self.nodeSettings.append(rs)
                else:
                    self.nodeSettings.append(rs)


    def getTractorConfig(self):
        self.tr_engine = dxConfig.getConf('TRACTOR_IP')
        self.tr_port   = dxConfig.getConf('TRACTOR_PORT')

        engineStr = self.renderSettings[-1].customFarmSettings['Tractor.engine']
        if engineStr:
            splitStr = engineStr.split(':')
            self.tr_engine = splitStr[0]
            if len(splitStr) > 1:
                self.tr_port = int(splitStr[1])

        self.tr_tier     = self.renderSettings[-1].customFarmSettings['Tractor.tier']
        self.tr_priority = int(self.renderSettings[-1].customFarmSettings['Tractor.priority'])
        self.tr_projects = self.renderSettings[-1].customFarmSettings['Tractor.projects'].split(',')
        self.tr_maxactive= int(self.renderSettings[-1].customFarmSettings['Tractor.maxactive'])
        # limit tag name
        tags = self.renderSettings[-1].customFarmSettings['Tractor.tags']
        if tags:
            tags = tags.split(',')
            if 'all' in tags:
                tags.remove('all')
            if not '3d' in tags:
                tags.append('3d')
        else:
            tags = ['3d']
        # if 'all' in tags:
        #     tags.remove('all')
        self.tr_tags = tags

        self.srvkey = 'KatanaRender'

    def getGraphStateVariablesName(self):
        result = list()
        variablesGroup = NodegraphAPI.GetRootNode().getParameter('variables')
        for vg in variablesGroup.getChildren():
            name = vg.getName()
            name = name.replace('Variant', '')
            value= vg.getChild('value').getValue(0)
            if value:
                result += [name, value]
        self.GSV = string.join(result, '_')

    def createTemp(self):
        now  = datetime.datetime.now()
        stamp= now.strftime('%m%d%y_%H%M_%S')
        base, ext = os.path.splitext(os.path.basename(self.katanaFile))
        dirpath = os.path.dirname(self.katanaFile)
        dirpath+= '/Renderfarm_Submissions'
        suffix  = getpass.getuser()
        if self.GSV:
            suffix += '-' + self.GSV
            dirpath+= '/' + self.GSV
        suffix += '-' + stamp
        tempfile= '{DIR}/{BASE}--{SUFFIX}{EXT}'.format(
            DIR=dirpath, BASE=base, SUFFIX=suffix, EXT=ext
        )
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        shutil.copy2(self.katanaFile, tempfile)
        return tempfile

    def GetSlots(self, rs):
        atleast = 1; atmost = 1
        slots = rs.customFarmSettings['Tractor.slots']
        if slots:
            splitStr = slots.split('/')
            if len(splitStr) > 1:
                atleast = int(splitStr[0])
                atmost  = int(splitStr[1])
            else:
                atleast = int(splitStr[0])
                atmost  = int(splitStr[0])
        return atleast, atmost

    def GetFrameSamples(self, rs):
        frames = list()
        sequential = True
        # specified frames
        specifiedframes = rs.customFarmSettings['Specified.frames']
        if specifiedframes:
            sequential = False
            for i in specifiedframes.split(','):
                t = i.split('-')
                if len(t) == 2:
                    for f in range(int(t[0]), int(t[1])+1):
                        frames.append(f)
                else:
                    frames.append(int(t[0]))
            return frames, sequential

        by = int(rs.customFarmSettings['Specified.byframe'])
        fr = rs.frameRange
        if not fr:
            fr = (self.sceneFrameRange['start'], self.sceneFrameRange['end'])
        for f in range(int(fr[0]), int(fr[1])+1, by):
            frames.append(f)
        return frames, sequential


    #---------------------------------------------------------------------------
    def doIt(self):
        title = '(KAT)'
        if self.GSV:
            title += ' GSV:' + self.GSV
        else:
            title += ' ' + os.path.splitext(os.path.basename(self.katanaFile))[0]
        if self.titleSuffix:
            title += ' ---- %s' % self.titleSuffix

        if self.onlyDenoise:
            title += ' (DenoiseOnly)'
        else:
            self.renderFile = self.createTemp()
        print '[INFO TractorSpool RenderFile] :', self.renderFile
        job = author.Job(
            title=title, tier=self.tr_tier,
            priority=self.tr_priority, projects=self.tr_projects,
            maxactive=self.tr_maxactive, tags=self.tr_tags,
            comment='RenderFile: %s' % self.renderFile,
            metadata='', service=self.srvkey
        )

        JobTask = author.Task(title='job')
        JobTask.serialsubtasks = 1
        job.addChild(JobTask)
        # job.paused = True

        # Main Render
        if not self.onlyDenoise:
            self.render_jobscript(JobTask)

        # Denoise Render
        self.denoise_jobscript(JobTask)

        if self.localQueue:
            job.projects = []
            return job.asTcl()
        else:
            engine = TractorEngine(
                hostname=self.tr_engine, port=self.tr_port, user=getpass.getuser(), debug=True
            )
            engine.spool(job)


    def render_jobscript(self, parent):
        renderTask = author.Task(title='Render')
        parent.addChild(renderTask)

        frameTaskMap = dict()
        for rs in self.nodeSettings:
            frames, sequential = self.GetFrameSamples(rs)
            for f in frames:
                # FrameTask
                if frameTaskMap.has_key(f):
                    frameTask = frameTaskMap[f]
                else:
                    frameTask = author.Task(title='%04d' % f)
                    frameTaskMap[f] = frameTask
                    renderTask.addChild(frameTask)
                self.frameRender(rs, f, frameTask)

            if self.makeMov:
                for output in rs.outputs:
                    if output['name'] == 'primary':
                        atleast, atmost = self.GetSlots(rs)
                        outputLocation = output['outputLocation']
                        outputMov = '_'.join(outputLocation.rsplit('/', 1))

                        dirname = os.path.dirname(outputMov)
                        filename = os.path.basename(outputMov)
                        filename = '%s.mov' % filename.split('.')[0]
                        outputMov = os.path.join(dirname, filename)
                        command = ['/backstage/bin/DCC', 'rez-env', 'rv', '--', 'rvio', output['outputLocation'], '-o', outputMov, '-out709']
                        renderTask.addCommand(author.Command(service=self.srvkey, argv=command, atleast=atleast, atmost=atmost))

    def frameRender(self, rs, frame, parent):
        title = '%s : %04d' % (rs.nodeName, frame)
        if len(self.nodeSettings) > 1:
            renderTask = author.Task(title=title)
            parent.addChild(renderTask)
        else:
            renderTask = parent
            renderTask.title = title

        command = ['DCC', 'rez-env'] + self.m_rezopts + ['--', 'katana', '--batch']
        command+= ['--katana-file=%s' % self.renderFile]
        command+= ['--t=%s' % frame]
        command+= ['--render-node=%s' % rs.nodeName]
        atleast, atmost = self.GetSlots(rs)
        renderTask.addCommand(
            author.Command(
                service=self.srvkey, argv=command, atleast=atleast, atmost=atmost
            )
        )


    def denoise_jobscript(self, parent):
        self.denoiseSettings = list()
        for rs in self.nodeSettings:
            nodeName   = rs.nodeName
            renderNode = NodegraphAPI.GetNode(nodeName)
            # Get OutputDefine Node
            defineNode = GetOutputDefineNode(renderNode)
            if defineNode.getParameter('user.denoise'):
                # New Style
                enableParm = defineNode.getParameter('user.denoise.enable')
                if enableParm:
                    if enableParm.getValue(0):
                        mode = defineNode.getParameter('user.denoise.options.mode').getValue(0)
                        dfilter = defineNode.getParameter('user.denoise.options.filter').getValue(0)
                        strength= defineNode.getParameter('user.denoise.options.strength').getValue(0)
                        aov     = defineNode.getParameter('user.denoise.options.aov').getValue(0)
                        version = defineNode.getParameter('user.denoise.options.version').getValue(0)
                        data = {
                            'nodeName': nodeName, 'frameSamples': self.GetFrameSamples(rs),
                            'denoise': mode, 'filter': dfilter, 'strength': strength,
                            'primary': '', 'aovs': [], 'version': version
                        }
                        for output in rs.outputs:
                            if output['name'] == 'primary':
                                data['primary'] = output['outputLocation']
                            if aov:
                                if output['name'] == 'lpes' or output['name'] == 'lgt':
                                    data['aovs'].append(output['outputLocation'])
                        self.denoiseSettings.append(data)

        if not self.denoiseSettings:
            return

        denoiseTask = author.Task(title='Denoise')
        parent.addChild(denoiseTask)

        frameTaskMap = dict()
        for ds in self.denoiseSettings:
            frames, sequential = ds['frameSamples']
            if sequential and ds['denoise'] == 2:
                del frames[1]; del frames[-2]
            for f in frames:
                # FrameTask
                if frameTaskMap.has_key(f):
                    frameTask = frameTaskMap[f]
                else:
                    frameTask = author.Task(title='%04d' % f)
                    frameTaskMap[f] = frameTask
                    denoiseTask.addChild(frameTask)
                self.frameDenoise(ds, f, frameTask)

    def frameDenoise(self, ds, frame, parent):
        title = '%s : %04d' % (ds['nodeName'], frame)
        if len(self.denoiseSettings) > 1:
            denoiseTask = author.Task(title=title)
            parent.addChild(denoiseTask)
        else:
            denoiseTask = parent
            denoiseTask.title = title

        # primary
        if ds['aovs']:
            primaryTask = author.Task(title='primary %04d' % frame)
            denoiseTask.addChild(primaryTask)
        else:
            primaryTask = denoiseTask
            primaryTask.title = 'primary %04d' % frame
        self.denoiseCommand(ds['primary'], frame, ds, primaryTask)

        # aovs
        for fn in ds['aovs']:
            aov = fn.split('.')[-3]
            aovTask = author.Task(title='%s %04d' % (aov, frame))
            denoiseTask.addChild(aovTask)
            self.denoiseCommand(fn, frame, ds, aovTask)

    def denoiseCommand(self, filename, frame, ds, parent):
        if ds.has_key('version') and ds['version']:
            renderman_pkgname = 'renderman-%s' % ds['version']
        else:
            renderman_pkgname = 'renderman'
            if self.m_prman_ver:
                renderman_pkgname += '-%s' % self.m_prman_ver

        cmd  = ['DCC', 'rez-env', renderman_pkgname, '--', 'denoiser']
        cmd += ['-f', ds['filter']]
        cmd += ['--strength', '%.02f' % float(ds['strength'])]
        cmd += ['--frame', frame]
        if ds['denoise'] != 1:
            cmd += ['--crossframe']

        cmd += [filename]
        parent.addCommand(
            author.Command(service='PixarRender', argv=cmd, tags=['denoise'])
        )
