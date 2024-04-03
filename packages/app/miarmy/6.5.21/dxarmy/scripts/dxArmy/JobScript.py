'''
LAST RELEASE
- 2017.12.13 : start
'''

import maya.cmds as cmds
import maya.mel as mel

from config import *
from TractorEngine import *


def IterateFrame(start, end, chunk):
    result = list()
    for f in range(start, end+1, chunk):
        s = f
        e = f + chunk - 1
        if e > end:
            e = end
        result.append((s, e))
    return result


def GetMiarmyEnvKey():
    maya_ver = cmds.about(v=True)
    plugname = 'MiarmyProForMaya%s' % maya_ver
    if not cmds.pluginInfo(plugname, q=True, l=True):
        cmds.loadPlugin(plugname)
    miarmy_ver = cmds.pluginInfo(plugname, q=True, v=True)
    return 'miarmy-%s-%s' % (maya_ver, miarmy_ver)

__DefaultOptions = {
    'm_engine': '10.0.0.25',
    'm_port': 80,
    'm_user': getpass.getuser(),
    'm_envkey': GetMiarmyEnvKey(),
    'm_priority': 100
}

def OptionsSetup(options):
    result = dict()
    for d in (__DefaultOptions, options):
        for key, value in d.items():
            result[key] = value
    return result


class JobMain:
    '''
    Miarmy RIB Export Jobscript
    options : dictionary types
        m_engine : tractor-engine ip <str>, 10.0.0.25
        m_port : tractor-engine port <int>, 80
        m_user   : job spool user <str>, desktop login username
        m_envkey : tractor environment key <str>, miarmy-$mayaversion-$miarmyversion
        m_priority : tractor priority <int>, 100
        m_tier : tractor tier
        m_projects : tractor projects
        m_tags : tractor tags

        m_chunk  : dispatch size <int>

        m_mayafile : current maya file <str>
        m_outdir   : output folder <str>
        m_start    : start frame <int>
        m_end      : end frame <int>
    '''
    def __init__(self, options):
        self.options = OptionsSetup(options)


    def doIt(self):
        job = self.jobscript()

        engine = TractorEngine(self.options)

        if self.options.has_key('m_tier') and self.options['m_tier']:
            job.tier = self.options['m_tier']
        if self.options.has_key('m_projects') and self.options['m_projects']:
            job.projects = [self.options['m_projects']]
        if self.options.has_key('m_tags') and self.options['m_tags']:
            job.tags = [self.options['m_tags']]

        engine.spool(job)


    def jobscript(self):
        '''
        JobScript Write
        '''
        basename = os.path.splitext(os.path.basename(self.options['m_mayafile']))[0]
        title = '(MiArmy-RibExport) %s' % basename
        meta  = 'scene:%s,outdir:%s' % (self.options['m_mayafile'], self.options['m_outdir'])
        job = author.Job(
            title=str(title),
            comment=str(''),
            metadata=str(meta),
            envkey=[str(self.options['m_envkey'])],
            service='Miarmy',
            priority=self.options['m_priority']
        )

        JobTask = author.Task(title='Job')
        JobTask.serialsubtasks = 0

        # points export
        self.exportTask(
            'AbcPoints %s-%s' % (self.options['m_start'], self.options['m_end']),
            3, self.options['m_start'], self.options['m_end'],
            JobTask
        )

        if self.options['m_chunk'] == 0:
            self.exportTask(
                'Rib %s-%s' % (self.options['m_start'], self.options['m_end']),
                0, self.options['m_start'], self.options['m_end'],
                JobTask
            )
        else:
            self.exportTask(
                'PrimAsset', 1, self.options['m_start'], self.options['m_end'],
                JobTask
            )

            iterate = IterateFrame(
                self.options['m_start'], self.options['m_end'],
                self.options['m_chunk']
            )
            ribTask = author.Task(title='Frame')
            ribTask.serialsubtasks = 0
            for start, end in iterate:
                self.exportTask(
                    '%s-%s' % (start, end), 2, start, end, ribTask
                )
            JobTask.addChild(ribTask)

        job.addChild(JobTask)
        return job


    def exportTask(self, name, mode, start, end, parent):
        task = author.Task(title=str(name))

        command  = ['maya', '-batch', '-file', self.options['m_mayafile']]
        command += ['-command', 'DxArmyExportRib %s %s %s \"%s\"' % (mode, start, end, self.options['m_outdir'])]
        task.addCommand(
            author.Command(argv=command, service='Miarmy', tags=['maya', 'miarmy'])
        )
        parent.addChild(task)


#-------------------------------------------------------------------------------
def ExportRibSpool(options):
    jobClass = JobMain(options)
    jobClass.doIt()


def SampleSpool():
    filename = cmds.file(q=True, sn=True)
    basename = os.path.splitext(os.path.basename(filename))[0]
    outdir = os.path.join(filename.split('scenes')[0], 'data', basename)
    options = {
        'm_engine': '10.1.3.59',
        'm_user': getpass.getuser(),
        'm_envkey': 'miarmy-5.1.11',
        'm_priority': 100,
        'm_chunk': 10,
        'm_mayafile': cmds.file(q=True, sn=True),
        'm_outdir': outdir,
        'm_start': int(cmds.playbackOptions(q=True, min=True)),
        'm_end': int(cmds.playbackOptions(q=True, max=True))
    }
    return options
