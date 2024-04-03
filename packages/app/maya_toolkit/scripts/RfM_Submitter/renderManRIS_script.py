#encoding=utf-8
#!/usr/bin/env python

"""
RenderMan RIS render job script

TODO:

LAST RELEASE:
- 2017.08.26 : remove prmanDeepRender task. it's seperate beauty and deep. using deepRender.so ri-filter
- 2017.09.25 : option re-build
- 2017.11.10 : render_mayafile path change to scenes directory
- 2017.12.01 : stereo camera control
- 2017.12.03 : prman command add loglevel not apply
- 2017.12.10 : if denoise, suffix add '_variance'

adding by joonkyun
- 2018.03.13 : denoise task add highRes(image resolusion over 4k) flag
"""


import config
from configMaya import *
from TractorEngine import *

import precompTask
import copy
import json


def SetMetaData():
    """
    Set OpenEXR METADATA
    :return:
    """
    if cmds.getAttr('rmanFinalOutputGlobals0.rman__riopt__Display_type') != 'openexr':
        return
    rg = 'rmanFinalOutputGlobals0'

    # render
    attr_name = 'rman__riopt__Display_exrheader_render'
    if not cmds.attributeQuery(attr_name, n=rg, ex=True):
        mel.eval('rmanAddAttr "%s" "%s" ""' % (rg, attr_name))
    cmds.setAttr('%s.%s' % (rg, attr_name),
                 '[mel "GetRenderData [GetVar CAMERA]"]', type='string')

    # focalLength
    attr_name = 'rman__riopt__Display_exrheader_focal'
    if not cmds.attributeQuery(attr_name, n=rg, ex=True):
        mel.eval('rmanAddAttr "%s" "%s" ""' % (rg, attr_name))
    cmds.setAttr('%s.%s' % (rg, attr_name),
                 '[mel "GetFocal [GetVar CAMERA]"]', type='string')

    # horizontalFilmAperture
    attr_name = 'rman__riopt__Display_exrheader_haperture'
    if not cmds.attributeQuery(attr_name, n=rg, ex=True):
        mel.eval('rmanAddAttr "%s" "%s" ""' % (rg, attr_name))
    cmds.setAttr('%s.%s' % (rg, attr_name),
                 '[expr [mel "GetHAperture [GetVar CAMERA]"]/0.039370]', type='string')

    # verticalFilmAperture
    attr_name = 'rman__riopt__Display_exrheader_vaperture'
    if not cmds.attributeQuery(attr_name, n=rg, ex=True):
        mel.eval('rmanAddAttr "%s" "%s" ""' % (rg, attr_name))
    cmds.setAttr('%s.%s' % (rg, attr_name),
                 '[expr [mel "GetVAperture [GetVar CAMERA]"]/0.039370]', type='string')


class JobMain:
    """
    RenderManForMaya renderManRIS jobscript
    options : dictionary type
        - init options
        m_mayafile : current maya file
        m_mayaproj : maya project path
        m_rmsprod  : renderman production root path
        m_version  : out version
        m_outdir   : output image path. ex> ../lighting/dev/images/v002
        m_mayaext  : maya file extension
        m_mayabasename :

        m_engine   : tractor-engine ip <str>
        m_port     : tractor-engine port <int>
        m_renderer : renderManRIS
        m_priority : 100
        m_envkey   : rfm2-21.4-maya-2017
        m_maxactive: 0
        m_user     : job spool user

        if stereo cam
            m_stereo : {'right':True, 'left':False}

        m_range    : frame range <str>, ex> 1-100,105,120
        m_by       : by frame <int>

        m_width, m_height : render resolution <int>

        m_ribgenLimit: ribgen dispatch count. default = 1
        m_ribgenOnly: 0

        m_shutterAngle : <int>
        m_motionBlur   : on/off <int>
        m_cameraBlur   : on/off <int>
        m_tracedBlur   : on/off <int>
        m_minsamples   : <int>
        m_maxsamples   : <int>
        m_checkPoint   : <int>
        m_incremental  : on/off <int>

        m_denoise : <int>
        m_denoiseFilter :
        m_denoiseaov    : on/off <int>
        m_denoiseStrength : <float>
    """
    def __init__(self, options):
        self.options = OptionsSetup(options)

    def doIt(self):
        """
        Main Process
        :return:
        """
        # renderSetup info
        self.rs = RenderSetupInfo()

        self.rs_denoiseLayers = self.rs.getDenoiseInfo(self.options['m_denoise'],
                                                       self.options['m_denoiseFilter'])

        # create maya tmp
        self.createMayaTempFile()

        # job
        job = self.external_jobscript()

        engine = TractorEngine(
            hostname=self.options['m_engine'],
            user=self.options['m_user']
        )
        if self.options['m_cloudJob']:
            job.paused = True

        job.tier = 'LT'
        job.projects = []
        job.tags = ['ALL']
        if self.options['m_cloudJob']:
            job.tags = []

        project = 'lt_other'
        src = self.options['m_mayafile'].split('/')
        if 'show' in src:
            file_project = src[src.index('show')+1]
            if file_project in engine.cfg_projects:
                project = str('lt_%s' % file_project)
            if 'asset' in src:
                project = 'lt_asset'
        job.projects = [project]

        # f = open(self.options['m_jobfile'], 'w')
        # f.write( job.asTcl() )
        # f.close()
        # print '# write jobscript : %s' % self.options['m_jobfile']

        retMsg = engine.spool(job)

        print self.options['m_cloudJob']
        if self.options['m_cloudJob']:
            self.cloudJobSetup(retMsg)

    def getPrefix(self):
        '''
        $SCENENAME_$LAYER_$CAMERA_$VERSION
        '''
        layer_prefix  = ''
        file_prefix = self.options['m_mayabasename']

        # renderLayer
        if self.rs._doLayer:
            layer_prefix = '<RenderLayer>'
            file_prefix += '_<RenderLayer>'

        # camera
        ifCamEver = False
        for i in self.rs._info:
            if len(self.rs._info[i]['camera']) > 1:
                ifCamEver = True
        if ifCamEver:
            file_prefix += '_<Camera>'

        # version
        file_prefix += '_' + self.options['m_version']

        # prefix _variance
        if cmds.ls(type='dxAOV'):
            if self.options['m_denoise'] or self.rs_denoiseLayers:
                file_prefix += '_variance'

        prefix = os.path.join(layer_prefix, file_prefix)
        return prefix.replace(os.path.sep, '/')

    # Create Render File
    def createMayaTempFile(self):
        now = datetime.datetime.now()
        stamp = now.strftime('%Y-%m-%d-%H%M-%S')
        render_mayafile = '{PROJ}/scenes/{BASE}_{VERSION}-{USER}-{STAMP}{EXT}'.format(
            PROJ=self.options['m_mayaproj'],
            BASE=self.options['m_mayabasename'],
            VERSION=self.options['m_version'],
            USER=self.options['m_user'],
            STAMP=stamp,
            EXT=self.options['m_mayaext']
        )
        self.options['m_rendermaya'] = render_mayafile
        self.options['m_jobfile'] = self.options['m_rendermaya'].replace(self.options['m_mayaext'], '.alf')

        # backstageLight node check
        if not cmds.ls(type=['stupidAOV', 'stupidRender', 'dxAOV']):
            cmds.createNode('dxAOV')

        # openexr autocrop setting
        autocrop_attr = 'rmanFinalOutputGlobals0.rman__riopt__Display_autocrop'
        current_autocrop = cmds.getAttr(autocrop_attr)
        if self.options['m_denoise'] or self.rs_denoiseLayers:
            cmds.setAttr(autocrop_attr, 'false', type='string')
        # openexr storage setting
        cmds.setAttr('rmanFinalOutputGlobals0.rman__riopt__Display_storage', 'scanline', type='string')

        # openexr meta-data
        SetMetaData()

        # prefixSetup
        cmds.setAttr('defaultRenderGlobals.imageFilePrefix', self.getPrefix(), type='string')

        # Stereo Camera
        if self.options.has_key('m_stereo'):
            rs_info = copy.deepcopy(self.rs._info)
            for s in self.options['m_stereo']:
                if not self.options['m_stereo'][s]:
                    for layer in rs_info:
                        for camShape in rs_info[layer]['camera']:
                            if camShape.find(s) > -1:
                                cmds.setAttr('%s.renderable' % camShape, 0)
                                self.rs._info[layer]['camera'].remove(camShape)

        # save and copy
        cmds.file(save=True)
        shutil.copy2(self.options['m_mayafile'], self.options['m_rendermaya'])

        # reset
        cmds.setAttr(autocrop_attr, current_autocrop, type='string')
        cmds.setAttr('defaultRenderGlobals.imageFilePrefix', '', type='string')
        if self.options.has_key('m_stereo'):
            for layer in rs_info:
                for camShape in rs_info[layer]['camera']:
                    cmds.setAttr('%s.renderable' % camShape, 1)


    def postScript( self, Parent=None ):
        """
        PostScript for cleanup
        :param tempFiles:
        :param Parent:
        :return:
        """
        # cleanup
        dirpath = os.path.join( self.options['m_mayaproj'],
                                'tmp', 'renderman',
                                os.path.splitext(os.path.basename(self.options['m_rendermaya']))[0] )
        dirpath = dirpath.replace( os.path.sep, '/' )
        Parent.addCleanup(
            author.Command(argv=['/bin/rm', '-rf', '%%D(%s)' % dirpath])
        )


    def ribgenRender(self, Start=1, End=1, By=1, Layer=None, Parent=None):
        """
        RibGen Taks
        :param Start:
        :param End:
        :param By:
        :param Layer:
        :param Parent:
        :return:
        """
        command  = ['Render', '-r', 'rib']
        if Layer:
            command += ['-rl', Layer]

        command += ['-fnc', 3, '-of', 'OpenEXR', '-pad', 4, '-iip']
        command += ['-setAttr', 'motionBlur', self.options['m_motionBlur']]
        command += ['-setAttr', 'cameraBlur', self.options['m_cameraBlur']]
        command += ['-setAttr', 'shutterAngle', self.options['m_shutterAngle']]
        command += ['-setAttr', 'trace:samplemotion', self.options['m_tracedBlur']]

        # override
        incremental = self.options['m_incremental']
        minsamples  = self.options['m_minsamples']
        maxsamples  = self.options['m_maxsamples']
        if Layer and self.rs._info[Layer].has_key('sampling'):
            if self.rs._info[Layer]['sampling'].has_key('incremental'):
                incremental = self.rs._info[Layer]['sampling']['incremental']
            if self.rs._info[Layer]['sampling'].has_key('minsamples'):
                minsamples  = self.rs._info[Layer]['sampling']['minsamples']
            if self.rs._info[Layer]['sampling'].has_key('maxsamples'):
                maxsamples  = self.rs._info[Layer]['sampling']['maxsamples']

        command += ['-setAttr', 'Hider:incremental', incremental]
        command += ['-setAttr', 'Hider:minsamples', minsamples]
        command += ['-setAttr', 'Hider:maxsamples', maxsamples]

        denoise = self.options['m_denoise']
        if Layer and self.rs._info[Layer].has_key('denoise'):
            denoise = self.rs._info[Layer]['denoise'][0]

        command += ['-setAttr', 'denoise', denoise]

        command += ['-s', Start, '-e', End, '-b', By]
        if self.options.has_key('m_width') and self.options.has_key('m_height'):
            command += ['-res', self.options['m_width'], self.options['m_height']]
        command += ['-rd', '%%D(%s)' % self.options['m_outdir']]
        command += ['-proj', '%%D(%s)' % self.options['m_mayaproj']]
        command += ['%%D(%s)' % self.options['m_rendermaya']]

        ribgenKey = 'RfMRibGen'
        RenderTask = author.Task( title='%s-%s' % (Start, End) )
        RenderTask.addCommand(
            author.Command(service=ribgenKey,
                           envkey=[self.options['m_envkey']],
                           tags=['rib', 'lt_ribgen'],
                           argv=command, atleast=1)
        )
        Parent.addChild( RenderTask )



    def prmanRender( self, FilePath='', FileName='', Frame=None, Parent=None,  ):
        """
        PRMan main Task
        :param FilePath:
        :param FileName:
        :param Frame:
        :param Parent:
        :return:
        """
        if Frame:
            Title = '%s %04d' % (FileName, Frame)
            ribfile = string.join(
                    [FilePath, '%04d' % Frame, '%s.%04d.rib' % (FileName, Frame)], '/' )
        else:
            Title = FileName
            ribfile = string.join(
                    [FilePath, 'job', '%s.rib' % FileName], '/' )

        # command = [ 'prman', '-t:0', '-Progress', '-loglevel', 2 ]
        command = ['prman', '-t:0', '-Progress']
        if self.options['m_incremental']:
            # command += [ '-recover', '%r', '-checkpoint', '%sm' % self.options['m_checkPoint'] ]
            # command += ['-recover', '1']
            # command += ['-checkpoint', '10m,3h']
            command += ['-recover', '%s' % self.options['m_recovery']]
            command += ['-checkpoint', '{0}m'.format(self.options['m_checkPoint'])]
        command += [ '-cwd', '%%D(%s)' % self.options['m_rmsprod'] ]
        command += [ '%%D(%s)' % ribfile ]

        RenderTask = author.Task( title=str(Title) )
        RenderTask.addCommand(
            author.Command(service='PixarRender',
                           envkey=[self.options['m_envkey']],
                           tags=['prman', 'lt_prman'],
                           argv=command, atleast=1)
        )

        Parent.addChild( RenderTask )


    def prmanTask( self, Iter=None, Parent=None ):
        # ribfile path
        basename = os.path.splitext(os.path.basename(self.options['m_rendermaya']))[0]
        ribpath = string.join([self.options['m_mayaproj'], 'tmp', 'renderman', basename, 'rib'], '/')

        RootTask = author.Task( title='PRMan %s' % self.options['m_range'] )
        RootTask.serialsubtasks = 1

        # Preflight
        PrefRootTask = author.Task( title='Preflight' )
        PrefRootTask.serialsubtasks = 0
        # for renderLayer
        if self.rs._doLayer:
            for layer in self.rs._info:
                filename = 'job_%s' % self.rs.getRibname(layer)
                filename = filename.replace( ':', '_' )
                self.prmanRender( FilePath=ribpath, FileName=filename, Parent=PrefRootTask )
        else:
            self.prmanRender( FilePath=ribpath, FileName='job', Parent=PrefRootTask )

        RootTask.addChild( PrefRootTask )

        # Frame
        FrameRootTask = author.Task( title='Frame %s' % self.options['m_range'] )
        FrameRootTask.serialsubtasks = 0
        for start, end in Iter:
            for i in range( start, end+1, self.options['m_by'] ):
                FrameTask = author.Task( title='%04d' % i )
                # for renderLayer
                if self.rs._doLayer:
                    for layer in self.rs._info:
                        LayerTask = author.Task( title=str(layer) )
                        for cam in self.rs._info[layer]['camera']:
                            filename = '%s_Final_%s' % (cam, self.rs.getRibname(layer))
                            filename = filename.replace( ':', '_' )
                            filename = filename.replace( '|', '_' )
                            self.prmanRender( ribpath, filename, i, LayerTask )

                            if self.options['m_cloudJob']:
                                self.cloudFileCopy(LayerTask, i, config.GetCloudCopyIP(self.options['m_engine']), self.rs.getRibname(layer))
                                # print os.path.join(self.options['m_outdir'], self.rs.getRibname(layer), "*.%s.exr" % i)
                        FrameTask.addChild( LayerTask )
                else:
                    for cam in self.rs._info['defaultRenderLayer']['camera']:
                        filename = '%s_Final' % cam
                        filename = filename.replace( ':', '_' )
                        filename = filename.replace( '|', '_' )
                        self.prmanRender( ribpath, filename, i, FrameTask )

                        if self.options['m_cloudJob']:
                            self.cloudFileCopy(FrameTask, i, config.GetCloudCopyIP(self.options['m_engine']))
                FrameRootTask.addChild( FrameTask )

        RootTask.addChild( FrameRootTask )

        Parent.addChild( RootTask )

    def resCheck(self):
        width = self.options['m_width']
        height = self.options['m_height']
        if width*height > pow(4096, 2):
            return 1
        else:
            return 0



    def denoiseTask(self, Parent=None):
        """
        Denoise Spool Task
        :param Parent:
        :return:
        """
        if self.options['m_denoise'] or self.rs_denoiseLayers:
            task = author.Task(title='DeNoise Spool')
            command  = [DeNoiseCmd]
            command += ['-p', 'jobscript']
            command += ['-i', self.options['m_outdir']]
            command += ['-f', self.options['m_range']]
            command += ['-d', self.options['m_denoise']]
            if self.rs_denoiseLayers:
                command += ['-l', string.join(self.rs_denoiseLayers, ',')]
            if self.options['m_denoiseaov']:
                command += ['-a']
            command += ['--denoiseFilter', self.options['m_denoiseFilter']]
            command += ['--strength', self.options['m_denoiseStrength']]
            # command += ['--engine', self.options['m_engine']]
            command += ['--hiRes', self.resCheck()]
            command += ['--envkey', self.options['m_envkey']]
            command += ['--user', getpass.getuser()]
            command += ['--title', '%s_%s' % (self.options['m_mayabasename'], self.options['m_version'])]

            if self.options['m_cloudJob']:
                command += ['--denoiseip', config.GetCloudDenoiseIP(self.options['m_engine'])]
                command += ['--copyip', config.GetCloudCopyIP(self.options['m_engine'])]

            # IF PRECOMP OPTION THEN PASS PRECOMP OPTION TO DENOISE SCRIPT
            if self.options.has_key('m_precompFile') and self.options['m_precompFile']:
                command += ['--precomp', self.options['m_precompFile']]

            task.addCommand(
                author.Command(service='PixarRender', argv=command, atleast=1)
            )

            Parent.addChild(task)

            return True



    def precompTask(self, Parent=None):
        """
        PreComp Spool Task
        """

        if self.options.has_key('m_precompFile') and self.options['m_precompFile']:
            task = author.Task(title='PreComp Spool')
            Parent.addChild(task)


    def external_jobscript( self ):
        """
        JobScript Write
        :return:
        """
        title = '(RIS) {BASE}_{VERSION}'.format(
            BASE=self.options['m_mayabasename'], VERSION=self.options['m_version']
        )
        serviceKey = "PixarRender"
        if self.options['m_cloudJob']:
            serviceKey = "DexterRender"
        job = author.Job(
            title=str(title),
            comment=str('maya:%s' % self.options['m_rendermaya']),
            metadata=str('outdir:%s' % self.options['m_outdir']),
            envkey=[self.options['m_envkey']],
            service=serviceKey,
            priority=self.options['m_priority'],
            maxactive=self.options['m_maxactive']
        )

        # directory mapping
        job.newDirMap(src='S:/', dst='/show/', zone='NFS')
        job.newDirMap(src='N:/', dst='/netapp/', zone='NFS')

        JobTask = author.Task(title='Job')
        JobTask.serialsubtasks = 1

        iterate = IterateFrame(self.options['m_range'], self.options['m_ribgenLimit'])

        # ribgen
        RibTask = author.Task(title='RibGen %s' % self.options['m_range'])
        if self.rs._doLayer:
            for layer in self.rs._info:
                RibLayerTask = author.Task(title=str(layer))
                for start, end in iterate:
                    self.ribgenRender(
                        Start=start, End=end, By=self.options['m_by'],
                        Layer=layer, Parent=RibLayerTask
                    )
                RibTask.addChild(RibLayerTask)
        else:
            for start, end in iterate:
                self.ribgenRender(
                    Start=start, End=end, By=self.options['m_by'],
                    Parent=RibTask
                )
        JobTask.addChild(RibTask)

        # prman
        self.prmanTask(Iter=iterate, Parent=JobTask)

        # denoise
        denoise = self.denoiseTask(Parent=JobTask)

        # IF NO DENOISE AND PRECOMP NK FILE
        if not denoise:
            if self.options.has_key('m_precompFile') and self.options['m_precompFile']:
                precompTask.precompTask(self.options['m_precompFile'],
                                        self.options['m_outdir'],
                                        Parent=JobTask)
            ################ adding data to db function name
            #self.toDbTask(JobTask)

        # clean up
        self.postScript(Parent=JobTask)

        job.addChild(JobTask)

        return job


    # # ex) adding data to db funciton(temp) : make module file
    # def toDbTask(self, parent):
    #     dbtask = author.Task(title='toDb')
    #     command = ['python','toDb.py']
    #     dbtask.addCommand(author.Command(service='PixarRender', argv=command))
    #     parent.addChild(dbtask)

    # CLOUD SETUP
    def cloudJobSetup(self, spoolMsg):
        # {u'msg': u'job script accepted, jid: 16', u'jid': 16, u'rc': 0}
        engine = TractorEngine(
            hostname="10.0.0.35",
            user=self.options['m_user']
        )

        # title = '(RIS) {BASE}_{VERSION}'.format(
        #     BASE=self.options['m_mayabasename'], VERSION=self.options['m_version']
        # )
        project = 'lt_other'
        file_project = 'other'
        src = self.options['m_mayafile'].split('/')
        if 'show' in src:
            file_project = src[src.index('show') + 1]
            if file_project in engine.cfg_projects:
                project = str('lt_%s' % file_project)

        title = '(CHECK CLOUD) CLOUD[{IP}]--{SHOW}--{BASE}_{VERSION}'.format(IP=self.options['m_engine'],
                                                                             SHOW=file_project,
                                                                             BASE=self.options['m_mayabasename'],
                                                                             VERSION=self.options['m_version'])
        job = author.Job(
            title=str(title),
            comment=str('maya:%s' % self.options['m_rendermaya']),
            metadata=str('outdir:%s' % self.options['m_outdir']),
            envkey=[self.options['m_envkey']],
            service='rsync',
            priority=1000,
            maxactive=self.options['m_maxactive']
        )

        # directory mapping
        job.newDirMap(src='S:/', dst='/show/', zone='NFS')
        job.newDirMap(src='N:/', dst='/netapp/', zone='NFS')

        JobTask = author.Task(title='Job')
        JobTask.serialsubtasks = 1

        job.tier = 'LT'
        job.projects = []
        job.tags = ['ALL']
        job.projects = [project]

        cacheFilePath = self.makeFileList(self.options['m_rendermaya'])

        RibTask = author.Task(title='(CHECK CLOUD) CLOUD[{IP}]--{SHOW}--{BASE}'.format(IP=self.options['m_engine'],
                                                                                       SHOW=file_project,
                                                                                       BASE=self.options['m_mayabasename']))

        ribgenKey = 'RfMRibGen'
        #
        command = ["python", "/dexter/Cache_DATA/supervisor/daeseok/4_Cloud/Oracle/oracleJobCheckin.py"]
        command += [cacheFilePath]
        command += ["175.126.207.106"]
        command += [str(spoolMsg['jid'])]
        RibTask.addCommand(
            author.Command(service="rsync",
                           envkey=[self.options['m_envkey']],
                           tags=[],
                           argv=command, atleast=1)
        )

        JobTask.addChild(RibTask)

        job.addChild(JobTask)

        # f = open(self.options['m_jobfile'], 'w')
        # f.write( job.asTcl() )
        # f.close()
        # print '# write jobscript : %s' % self.options['m_jobfile']
        # job.paused = True

        retMsg = engine.spool(job)

    def makeFileList(self, filename):
        filePathList = []
        dirPathList = []

        def insertFile(filePath):
            filePath = filePath.replace("/netapp/dexter/show", "/show")
            if not os.path.exists(filePath):
                return

            if "/assetlib/3D" in filePath:
                texDirPath = os.path.join(os.path.dirname(os.path.dirname(filePath)), "texture", "tex")
                insertDir(texDirPath)
                return

            if not filePath in filePathList:
                filePathList.append(filePath)

        def insertDir(filePath):
            filePath = filePath.replace("/netapp/dexter/show", "/show")
            if not os.path.exists(filePath):
                return
            if not filePath in dirPathList:
                dirPathList.append(filePath)

        # first Dome Light hdri
        for node in cmds.ls(type="PxrDomeLight"):
            insertFile(cmds.getAttr("%s.lightColorMap" % node))

        # assembly(.asb)
        for node in cmds.ls(type="ZAssemblyArchive"):
            asbPath = cmds.getAttr("%s.asbFilePath" % node)
            insertFile(asbPath)

            jsonPath = asbPath.replace('.asb', '.json')
            if not os.path.exists(jsonPath):
                splitJsonPath = jsonPath.split(".")[0]
                jsonDirname = os.path.dirname(splitJsonPath)
                jsonFilename = os.path.basename(splitJsonPath)
                splitFileName = jsonFilename.split("_")
                jsonFileName = "%s_assembly_%s.json" % (splitFileName[0], splitFileName[-1])
                jsonPath = os.path.join(jsonDirname, jsonFileName)

            if not os.path.exists(jsonPath):
                cmds.warning("not found asb to json : %s" % jsonPath)
                continue

            f = open(jsonPath, "r")
            data = json.load(f)
            for filePath in data["InstanceSetup"]["abcfiles"]:
                insertFile(filePath)

        for node in cmds.ls(type="dxAssembly"):
            insertFile(cmds.getAttr("%s.fileName" % node))

        # dxComponent
        for node in cmds.ls(type='dxComponent'):
            abcFilePath = cmds.getAttr("%s.abcFileName" % node)
            path = cmds.getAttr("%s.renderFile" % node)
            worldFilePath = cmds.getAttr("%s.worldFileName" % node)
            jsonfilePath = cmds.getAttr("%s.jsonFile" % node)
            if path:
                insertFile(path)
            if abcFilePath:
                insertFile(abcFilePath)
            if worldFilePath:
                insertFile(worldFilePath)
            if jsonfilePath:
                insertFile(jsonfilePath)

        for node in cmds.ls(type="ZGpuMeshCreator"):
            path = cmds.getAttr("%s.file" % node)
            if path:
                insertFile(path)

        # Tane Source
        for node in cmds.ls(type="TN_AbcProxyMPxSurfaceShape"):
            renderFile = cmds.getAttr("%s.filepath" % node)
            proxyFile = cmds.getAttr("%s.proxypath" % node)
            if renderFile:
                insertFile(renderFile)
            if proxyFile:
                insertFile(proxyFile)

        # Tane Cache
        for node in cmds.ls(type="TN_ImportCacheMPxNode"):
            cachePath = cmds.getAttr("%s.cachePath" % node)
            # if not cachePath:
            #    cachePath = cmds.getAttr("%s.zenvCachePath" % node)
            insertFile(cachePath)

        # Camera
        for node in cmds.ls(type='dxCamera'):
            insertDir(os.path.dirname(cmds.getAttr("%s.fileName" % node)))

        for node in cmds.ls(type="ZVOceanLoader"):
            if cmds.listConnections("%s.resultVector" % node):
                insertDir(os.path.dirname(cmds.getAttr("%s.jsonFile" % node)))

        # ZENN Archive
        for node in cmds.ls(type="zennArchive"):
            insertDir(cmds.getAttr("%s.cachePath" % node))

        # second crowd data
        for node in cmds.ls(type="dxArchive"):
            path = cmds.getAttr("%s.cachefile" % node)
            insertDir(os.path.dirname(os.path.dirname(os.path.dirname(path))))

        for node in cmds.ls(type="ZAbcPtcViewer"):
            insertFile(cmds.getAttr("%s.file" % node))

        for node in cmds.ls(type='DxTexture'):
            if cmds.getAttr("%s.txmode" % node) == 0:
                insertFile(cmds.getAttr("%s.filename" % node))

        for node in cmds.ls(type="PxrBoraOcean"):
            if cmds.listConnections("%s.outputRGB" % node):
                insertDir(os.path.dirname(cmds.getAttr("%s.inputFile" % node)))

        # scene file insert
        insertFile(filename)

        basicCachePath = filename.split('.')[0] + "_cacheList"
        with open(basicCachePath, "w") as f:
            for filePath in filePathList:
                print filePath
                f.write(filePath + "\n")
            for filePath in dirPathList:
                f.write(filePath + "\n")

        return basicCachePath

    def cloudFileCopy(self, task, frame, targetIP, rsLayer = "", denoise=False):
        return

        filepath = self.options['m_outdir']
        if rsLayer:
            filepath = os.path.join(filepath, rsLayer)
        filepath += "/"
        targetDir = "%s:%s" % (targetIP, os.path.dirname(filepath))

        if not os.path.exists(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))

        command = ["rsync", "-avzh", "--progress", "--no-o", "--no-g"]
        imgFileName = ".%s.exr" % frame
        if denoise:
            imgFileName = ".filtered" + imgFileName
        command += ['--include=*%s' % imgFileName, '--exclude=*']
        command += [filepath, targetDir]
        task.addCommand(
            author.Command(service='Rsync',
                           envkey=[self.options['m_envkey']],
                           tags=['prman', 'lt_prman'],
                           argv=command, atleast=1)
        )