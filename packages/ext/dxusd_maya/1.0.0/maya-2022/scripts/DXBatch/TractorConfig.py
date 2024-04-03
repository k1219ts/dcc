import sys

# Tractor
import dxConfig
TRACTOR_IP = dxConfig.getConf("TRACTOR_CACHE_IP")
# TRACTOR_IP = '10.0.0.25'
PORT = 80

import tractor.api.author as author

SERVICE_KEY = "Cache"
MAX_ACTIVE = 20
PROJECTS = ["export"]
TIER = "cache"
TAGS = ["GPU"]
ENVIROMNET_KEY = ""

def GetIterFrames(frameRange):
    '''
    distribute cache export
    :param frameRange:
    :return:
    '''

    result = list()
    if frameRange[0] == frameRange[1]:
        result.append((frameRange[0], frameRange[1]))
        return result

    duration = frameRange[1] - frameRange[0] + 1
    size = duration / 10
    if size < 5:
        size = 5
    elif size > 50:
        size = 50

    for frame in range(int(frameRange[0]) - 1, int(frameRange[1]) + 1, int(size)):
        start = frame
        end = frame + size - 1
        if end > frameRange[1] + 1:
            end = frameRange[1] + 1
        if end - start > 2:
            result.append((start, end))

    if result[-1][-1] != frameRange[1] + 1:
        result[-1] = (result[-1][0], frameRange[1] + 1)

    return result

def MakeTractorJob(title, comment="", metadata="", service=SERVICE_KEY):
    job = author.Job()
    job.title = title
    job.comment = comment
    job.metadata = metadata
    job.service = service
    job.maxactive = MAX_ACTIVE
    job.tier = TIER
    job.tags = TAGS
    job.projects = PROJECTS

    rootTask = MakeTask("Job Task")

    job.addChild(rootTask)
    return job, rootTask

def MakeTask(title):
    task = author.Task(title=title)
    return task

def MakeCommand(cmd):
    return author.Command(argv=cmd, service=SERVICE_KEY)

def SpoolJob(job, user):
    job.priority = 100
    # job.paused = True
    author.setEngineClientParam(hostname=TRACTOR_IP, port=PORT, user=user, debug=True)
    job.spool()
    print(job.asTcl())
    author.closeEngineClient()
