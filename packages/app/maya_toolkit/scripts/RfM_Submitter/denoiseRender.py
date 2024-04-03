#!/bin/python2.7

"""
Denoise Process

LAST RELEASE:
- 2017.11.10 : spool engine config by DenoiseEngine
               project different setup by engine
- 2017.12.01 : denoise command strength bugfix
               FilteredImageRemove process delete
               JobMain.searchFiles process change
               args engine remove

by joonkyun
- 2018.03.13 : adding getServiceKey function(for over 4k image denoise)
"""

from config import *
import optparse
from TractorEngine import *
import pprint
import precompTask



def FrameDenoiseProcess(options):
    cmd = 'denoise'
    cmd += ' -f %s' % options.denoiseFilter

    if options.denoiseFilter.find('volume') == -1:
        cmd += ' --override filterbanks.*.strength %s' % options.strength
        cmd += ' --'

    if options.aovname:
        cmd += ' %s_variance.%04d.exr' % (options.fileGroup, options.frame)
        cmd += ' %s.%s.%04d.exr' % (options.fileGroup, options.aovname, options.frame)
    else:
        cmd += ' %s.%04d.exr' % (options.fileGroup, options.frame)

    sys.stderr.write('# Command : %s\n' % cmd)
    os.system(cmd)


def CrossFrameDenoiseProcess(options):
    frame = options.frame
    if options.frameOption == 'start':
        skipOption = '-L true'
        frameOption = '%04d,%04d,%04d' % (frame, frame + 1, frame + 2)
    elif options.frameOption == 'end':
        skipOption = '-F true'
        frameOption = '%04d,%04d,%04d' % (frame - 2, frame - 1, frame)
    else:
        skipOption = '-F true -L true'
        frameOption = '%04d,%04d,%04d' % (frame - 1, frame, frame + 1)

    cmd = 'denoise'
    cmd += ' --crossframe -v variance %s' % skipOption
    cmd += ' -f %s' % options.denoiseFilter
    if options.denoiseFilter.find('volume') == -1:
        cmd += ' --override filterbanks.*.strength %s' % options.strength
        cmd += ' --'

    if options.aovname:
        cmd += ' %s_variance.{%s}.exr' % (options.fileGroup, frameOption)
        cmd += ' %s.%s.{%s}.exr' % (options.fileGroup, options.aovname, frameOption)
    else:
        cmd += ' %s.{%s}.exr' % (options.fileGroup, frameOption)

    sys.stderr.write('# Command : %s\n' % cmd)
    os.system(cmd)





def iterateFrame(frameRange):
    result = list()
    for i in frameRange.split(','):
        source = i.split('-')
        if len(source) > 1:
            start_frame = int(source[0])
            end_frame = int(source[1])
            for f in range(start_frame, end_frame+1):
                result.append(f)
        else:
            result.append(int(i))
    return result


class JobMain:
    def __init__(self, options):
        self.m_options = options

        self.m_layer = list(); self.m_filter = dict()
        self.m_layerOption = dict()
        if self.m_options.layer:
            for i in self.m_options.layer.split(','):
                src = i.split(':')  # denoise:layername:filtername
                self.m_layer.append(src[1])
                self.m_layerOption[src[1]] = (src[0], src[2])
                self.m_filter[src[1]] = src[2]


    def doIt(self):
        self.m_files, self.m_fileGroup, self.m_aovGroup = JobMain.searchFiles(self.m_options.imagepath,
                                                                              self.m_layer)
        self.masterImageRename()
        self.denoise_jobscript()


    @staticmethod
    def searchFiles(imagepath, layers):
        m_files = list()
        m_fileGroup = list()
        m_aovGroup  = list()

        if layers:
            for layer in layers:
                imgpath = os.path.join(imagepath, layer)
                files, fileGroup, aovGroup = JobMain.fileWalk(imgpath)
                m_files += files
                m_fileGroup += fileGroup
                m_aovGroup  += aovGroup
        else:
            files, fileGroup, aovGroup = JobMain.fileWalk(imagepath)
            m_files += files
            m_fileGroup += fileGroup
            m_aovGroup  += aovGroup

        m_files.sort()
        m_fileGroup = list(set(m_fileGroup))
        m_fileGroup.sort()
        m_aovGroup = list(set(m_aovGroup))
        m_aovGroup.sort()
        # debug
        print '# File Group'
        pprint.pprint(m_fileGroup)
        print '# AOV Group'
        pprint.pprint(m_aovGroup)
        return m_files, m_fileGroup, m_aovGroup

    @staticmethod
    def fileWalk(imagepath):
        m_files = list(); m_fileGroup = list(); m_aovGroup = list()
        for (root, dirs, files) in os.walk(imagepath):
            for f in files:
                if os.path.splitext(f)[-1] == '.exr' and f.find('_filtered') == -1:
                    filename = f.replace('_variance', '')
                    src = filename.split('.')
                    if len(src) == 3:
                        m_files.append(os.path.join(root, f))
                        m_fileGroup.append(os.path.join(root, src[0]))
                    elif len(src) == 4:
                        if f.find('.lpes.') > -1 or f.find('.lgt.') > -1:
                            m_aovGroup.append(os.path.join(root, src[0] + '.' + src[1]))
        return m_files, m_fileGroup, m_aovGroup


    def masterImageRename(self):
        for f in self.m_files:
            if f.find('variance') == -1:
                dirpath = os.path.dirname(f)
                basename = os.path.basename(f)
                src = basename.split('.')
                src[0] += '_variance'
                variance_file = string.join(src, '.')
                print '# rename : ', f
                print '\t-> ', variance_file
                os.rename(f, os.path.join(dirpath, variance_file))


    def denoise_jobscript(self):
        src = self.m_options.imagepath.split('/')
        jobtitle = '(DeNoise) '
        if self.m_options.title:
            jobtitle += ' ' + self.m_options.title
        else:
            if 'shot' in src:
                jobtitle += src[src.index('shot')+2] + ' '
            jobtitle += os.path.basename(self.m_options.imagepath)

        service = self.getServiceKey()
        if self.m_options.denoiseip:
            service = "DexterRender"
        job = author.Job(
            title=str(jobtitle),
            comment='',
            metadata='',
            envkey=[str(self.m_options.envkey)],
            service=service, # self.getServiceKey(),
            maxactive=0
        )

        if self.m_options.denoise == 1:
            Title = 'Frame Denoise'
        else:
            Title = 'Cross-frame Denoise'
        Title += ' %s' % self.m_options.frameRange

        JobTask = author.Task(title=Title)
        JobTask.serialsubtasks = 0

        iterate = iterateFrame(self.m_options.frameRange)
        if self.m_options.denoise == 1:
            self.frameDenoiseTask(iterate, JobTask)
        else:
            self.crossframeDenoiseTask(iterate, JobTask)

        # IF PRECOMP OPTION,
        if self.m_options.precomp:
        # if hasattr(self.m_options, 'precomp'):
            pcTask = precompTask.precompTask(self.m_options.precomp,
                                             self.m_options.imagepath,
                                             Parent=job)
            pcTask.addChild(JobTask)

        ################ adding data to db function name
        # self.addToDB(JobTask)

        job.addChild(JobTask)

        job.priority = 100
        self.enginehost = GetProcessEngine('DenoiseEngine')
        if self.m_options.denoiseip:
            self.enginehost = self.m_options.denoiseip
        else:
            if self.m_options.highResolution:
                self.enginehost = GetProcessEngine('TractorEngine')
                job.priority = 100
            else:
                self.enginehost = GetProcessEngine('DenoiseEngine')
                job.priority = 1000
        engine = TractorEngine(
            hostname=self.enginehost,
            user=self.m_options.user
        )

        job.tier = 'LT'
        job.tags = ['ALL']

        project = self.getProject()
        job.projects = [project]

        # job.paused = True
        engine.spool(job)
        # print job.asTcl()

    # def cloudFileCopy(self, task, frame, targetIP, rsLayer = "", denoise=False):
    #     filepath = self.m_options['m_outdir']
    #     if rsLayer:
    #         filepath = os.path.join(filepath, rsLayer)
    #     targetDir = "%s:%s" % (targetIP, os.path.dirname(filepath))
    #     command = ["rsync", "-avzh", "--progress", "-r"]
    #     imgFileName = ".%s.exr" % frame
    #     if denoise:
    #         imgFileName = ".filtered" + imgFileName
    #     command += ['--include=*%s' % imgFileName, '--exclude=*']
    #     command += [filepath, targetDir]
    #     task.addCommand(
    #         author.Command(service='PixarRender',
    #                        envkey=[self.options['m_envkey']],
    #                        tags=['prman', 'lt_prman'],
    #                        argv=command, atleast=1)
    #     )

    def getProject(self):
        if self.enginehost == '10.0.0.30' and not self.m_options.highResolution:
            project = 'lt_other'
            src = self.m_options.imagepath.split('/')
            if 'show' in src:
                file_project = src[src.index('show')+1]
                if file_project in engine.cfg_projects:
                    project = str('lt_%s' % file_project)
            return project
        elif self.enginehost == '10.0.0.35' or self.m_options.highResolution:
            return 'denoise'
        else:
            return 'lt_other'


    def frameDenoiseTask(self, Iter=None, Parent=None):
        for frame in Iter:
            FrameTask = author.Task(title='%04d' % frame)

            layers = list()
            for f in self.m_fileGroup:
                tmp = f.split(self.m_options.imagepath)[-1]
                src = tmp.split('/')
                if len(src) > 3:
                    layers.append(src[1])
            layers = list(set(layers))

            if layers:
                self.LayerTask(layers, frame, None, FrameTask)
            else:
                self.Task(frame, None, FrameTask)

            Parent.addChild(FrameTask)


    def crossframeDenoiseTask(self, Iter=None, Parent=None):
        for frame in Iter:
            FrameTask = None
            frameOpt  = None
            if frame == Iter[0]:
                Title = '%04d %04d' % (frame, frame+1)
                FrameTask = author.Task(title=Title)
                frameOpt = 'start'
            elif frame == Iter[1]:
                pass
            elif frame == Iter[-1]:
                Title = '%04d %04d' % (frame-1, frame)
                FrameTask = author.Task(title=Title)
                frameOpt = 'end'
            elif frame == Iter[-2]:
                pass
            else:
                FrameTask = author.Task(title='%04d' % frame)

            if FrameTask:
                layers = list()
                for f in self.m_fileGroup:
                    tmp = f.split(self.m_options.imagepath)[-1]
                    src = tmp.split('/')
                    if len(src) > 3:
                        layers.append(src[1])
                layers = list(set(layers))

                if layers:
                    self.LayerTask(layers, frame, frameOpt, FrameTask)
                else:
                    self.Task(frame, frameOpt, FrameTask)

                Parent.addChild(FrameTask)


    def Task(self, Frame=None, FrameOption=None, Parent=None):
        for f in self.m_fileGroup:
            # beauty
            fgroup = f
            if f.find('_variance') == -1:
                fgroup += '_variance'
            self.CommandTask(fgroup, Frame, FrameOption, None,
                             self.m_options.denoise, self.m_options.denoiseFilter, Parent)
            # aov
            if self.m_aovGroup and self.m_options.aov:
                for a in self.m_aovGroup:
                    if a.find(f) > -1:
                        aovName = a.split('.')[-1]
                        self.CommandTask(f, Frame, FrameOption, aovName,
                                         self.m_options.denoise, self.m_options.denoiseFilter, Parent)

            if self.m_options.copyip:
                self.cloudTask(fgroup, Frame, Parent)

    def LayerTask(self, Layers=list(), Frame=None, FrameOption=None, Parent=None):
        for lyr in Layers:
            LayerTask = author.Task(title=str(lyr))
            for f in self.m_fileGroup:
                if f.find(lyr) > -1:
                    # denoise, denoiseFilter
                    denoise    = self.m_options.denoise
                    filtername = self.m_options.denoiseFilter
                    if self.m_layerOption.has_key(lyr):
                        denoise    = self.m_layerOption[lyr][0]
                        filtername = self.m_layerOption[lyr][1]
                    # beauty
                    fgroup = f
                    if f.find('_variance') == -1:
                        fgroup += '_variance'
                    self.CommandTask(fgroup, Frame, FrameOption, None,
                                     denoise, filtername, LayerTask)
                    # aov
                    print self.m_aovGroup, self.m_options.aov
                    if self.m_aovGroup and self.m_options.aov:
                        for a in self.m_aovGroup:
                            if a.find(f) > -1:
                                aovName = a.split('.')[-1]
                                self.CommandTask(f, Frame, FrameOption, aovName,
                                                 denoise, filtername, LayerTask)

                    if self.m_options.copyip:
                        self.cloudTask(fgroup, Frame, LayerTask)
            Parent.addChild(LayerTask)


    def getServiceKey(self):
        if self.m_options.highResolution:
            return 'HiResDenoise'
        else:
            return 'DenoiseRender'


    def CommandTask(self,
                    FileGroup=None, Frame=None, FrameOption=None,
                    AOVName=None, Denoise=1, Filter=None, Parent=None):
        Title = str(os.path.basename(FileGroup))
        if AOVName:
            Title += '.%s' % AOVName
        task = author.Task(title=Title)
        command  = [PYTHONCMD, DeNoiseCmd, '-p', 'denoise']
        command += ['--fg', FileGroup, '--frame', Frame]
        if FrameOption:
            command += ['--fo', FrameOption]
        command += ['--denoise', Denoise]
        command += ['--strength', self.m_options.strength]
        if AOVName:
            command += ['--aovname', AOVName]
        if Filter:
            command += ['--denoiseFilter', Filter]
        task.addCommand(author.Command(service=self.getServiceKey(),
                                       argv=command, tags=['denoise']))

        Parent.addChild(task)

    # def addToDB(self, parent):
    #     dbtask = author.Task(title='toDb')
    #     command = ['python', 'toDb.py']
    #     dbtask.addCommand(author.Command(service='PixarRender', argv=command))
    #     parent.addChild(dbtask)

    def cloudTask(self, FileGroup = None, Frame = None, task = None):
        return
        srcDir = os.path.dirname(FileGroup)
        targetDir = "%s:%s" % (self.m_options.copyip, srcDir)
        command = ["rsync", "-avzh", "--progress", "--no-o", "--no-g"]
        imgFileName = "*_filtered.%s.exr" % Frame
        command += ['--include=%s' % imgFileName, '--exclude=*']
        command += [srcDir + "/", targetDir]
        task.addCommand(author.Command(service="Rsync", #self.getServiceKey(),
                                       argv=command))


if __name__ == '__main__':
    optparser = optparse.OptionParser()

    optparser.add_option(
        '-p', '--process', dest='process', type='string',
        help='ex> jobscript, denosie'
    )

    # JobScript Options
    optparser.add_option(
        '-i', '--imagepath', dest='imagepath', type='string',
        help='JobScript Options : render image path'
    )
    optparser.add_option(
        '-l', '--layer', dest='layer', type='string',
        help='JobScript Options : renderlayer name. ex> layerA,layerB'
    )
    optparser.add_option(
        '-f', '--fr', dest='frameRange', type='string',
        help='JobScript Options : frame range. ex> 1-24,100,102'
    )
    optparser.add_option(
        '-d', '--denoise', dest='denoise', type='int', default=1,
        help='JobScript Options : 1=Frame, 2=Cross-Frame'
    )
    optparser.add_option(
        '-a', '--aov', dest='aov', action='store_true',
        help='JobScript Options : enable denoise for aov files'
    )

    optparser.add_option(
        '--engine', dest='engine', type='string', default='10.0.0.30',
        help='JobScript Options : Tractor Engine IP, NOT WORK'
    )
    optparser.add_option(
        '--envkey', dest='envkey', type='string', default='rfm2-21.5-maya-2017',
        help='JobScript Options : Tractor Environment Key'
    )
    optparser.add_option(
        '--user', dest='user', type='string', default=getpass.getuser(),
        help='JobScript Options : Tractor username'
    )
    optparser.add_option(
        '--title', dest='title', type='string',
        help='spool title'
    )
    optparser.add_option(
        '--hiRes', dest='highResolution', type='int', default=0,
        help='JobScript Options : service key change by image resolution'
    )

    # cloud option
    optparser.add_option(
        '--denoiseip', dest='denoiseip', type='string', default='', action="store",
        help='JobScript Options : cloud denoise inline ip'
    )
    optparser.add_option(
        '--copyip', dest='copyip', type='string', default='', action="store",
        help='JobScript Options : cloud file copy ip'
    )

    # denoise command Options
    optparser.add_option(
        '--denoiseFilter', dest='denoiseFilter', type='string',
        default='default.filter.json',
        help='Denoise Options : denoise filter'
    )
    optparser.add_option(
        '--strength', dest='strength', type='float', default=0.2,
        help='Denoise Options : filter strength'
    )
    optparser.add_option(
        '--fg', dest='fileGroup', type='string',
        help='Denoise Options : file group name'
    )
    optparser.add_option(
        '--frame', dest='frame', type='int',
        help='Denoise Options : frame number'
    )
    optparser.add_option(
        '--fo', dest='frameOption', type='string',
        help='Denoise Options : frame option. ex> start, end'
    )
    optparser.add_option(
        '--aovname', dest='aovname', type='string',
        help='Denoise Options : aov name. ex> lgt, lpes'
    )
    # PRECOMP OPTION
    optparser.add_option(
        '--precomp', dest='precomp', type='string',
        help='--precomp nuke file path'
    )

    (opts, args) = optparser.parse_args(sys.argv)
    if opts.process == 'jobscript' and opts.imagepath:
        if os.path.isdir(opts.imagepath):
            jobClass = JobMain(opts)
            jobClass.doIt()

    elif opts.process == 'denoise':
        if opts.denoise == 1:
            FrameDenoiseProcess(opts)
        elif opts.denoise == 2:
            CrossFrameDenoiseProcess(opts)
