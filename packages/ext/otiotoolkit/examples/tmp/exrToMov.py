import subprocess
import os
# import dxConfig
# TRACTOR_IP = '10.0.0.35' # dxConfig.getConf("TRACTOR_CACHE_IP")
# PORT = 80
#
# import tractor.api.author as author
#
# SERVICE_KEY = "Cache"
# MAX_ACTIVE = 20
# PROJECTS = ["export"]
# TIER = "cache"
# TAGS = ["GPU"]
# ENVIROMNET_KEY = ""
#
# def MakeTractorJob(title, comment="", metadata=""):
#     job = author.Job()
#     job.title = title
#     job.comment = comment
#     job.metadata = metadata
#     job.service = SERVICE_KEY
#     job.maxactive = MAX_ACTIVE
#     job.tier = TIER
#     job.tags = TAGS
#     job.projects = PROJECTS
#
#     rootTask = MakeTask("Job Task")
#
#     job.addChild(rootTask)
#     return job, rootTask
#
# def MakeTask(title):
#     task = author.Task(title=title)
#     return task
#
# def MakeCommand(cmd):
#     return author.Command(argv=cmd, service=SERVICE_KEY)
#
# def SpoolJob(job, user):
#     job.priority = 100
#     author.setEngineClientParam(hostname=TRACTOR_IP, port=PORT, user=user, debug=True)
#     job.spool()
#     print job.asTcl()
#     author.closeEngineClient()

exrRootDir = '/assetlib/2D/Reference_Comp/EMD/src'
ocioConfig = os.path.join(os.getenv('REZ_OCIO_CONFIGS_ROOT'), 'config.ocio')
print ocioConfig

# ACES-2065-1 -> aces REC.709
cmdRule = 'oiiotool {INPUT_RULE} --colorconfig {OCIO_CONFIG} --colorconvert "ACES - ACES2065-1" "Output - Rec.709" -o {OUTPUT_RULE}'

for exrDir in os.listdir(exrRootDir):
    if not exrDir.startswith('.'):
        exrFullPathDir = os.path.join(exrRootDir, exrDir)
        if os.path.isdir(exrFullPathDir):
            fileName = os.listdir(exrFullPathDir)[0]
            splitFileName = fileName.split('.')
            splitFileName[-2] = '#'
            inputRule = os.path.join(exrFullPathDir, '.'.join(splitFileName))
            print inputRule
            print

            fileName = os.listdir(exrFullPathDir)[0]
            splitFileName = fileName.split('.')
            splitFileName[-2] = '#'
            splitFileName[-1] = 'jpg'
            outputRule = os.path.join(exrFullPathDir, 'jpg', '.'.join(splitFileName))
            print outputRule
            print

            outputDir = os.path.dirname(outputRule)
            if not os.path.exists(outputDir):
                os.makedirs(outputDir)

            cmd = cmdRule.format(INPUT_RULE=inputRule, OCIO_CONFIG=ocioConfig, OUTPUT_RULE=outputRule)
            print cmd
            print
            p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            while p.poll() is None:
                output = p.stdout.readline()
                if output:
                    print output.strip()

            # Make MOV
            movCmd = 'ffmpeg_converter -i %s -o %s -c proresLT' % (outputDir, exrRootDir)
            print movCmd
            p = subprocess.Popen(movCmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            while p.poll() is None:
                output = p.stdout.readline()
                if output:
                    print output.strip()

            rmDirCmd = 'rm -rf %s' % outputDir
            print rmDirCmd
            os.system(rmDirCmd)