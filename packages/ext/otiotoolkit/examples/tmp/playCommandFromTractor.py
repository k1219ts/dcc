import os
import subprocess
import getpass

# # Tractor
TRACTOR_IP = '10.0.0.106'
PORT = 80

import tractor.api.author as author

SERVICE_KEY = "nuke"
MAX_ACTIVE = 0
PROJECTS = ["comp"]
TIER = "comp"
TAGS = ["2d"]
ENVIROMNET_KEY = ""
# targetDir = '/show/yys/screening/_closed/_final/_final/20200929_257'
targetDir = '/show/yys/stuff/ftp/_257/to_dexter/20201109/comp/exr'


title = 'EXR MOV'
job = author.Job()
job.title = title
job.comment = 'exr directory : %s' % targetDir
job.service = SERVICE_KEY
job.maxactive = MAX_ACTIVE
job.tier = TIER
job.tags = TAGS
job.projects = PROJECTS

rootTask = author.Task(title=title)
cmd = '/backstage/bin/DCC rez-env pylibs -- python /backstage/apps/Maya/toolkits/dxsUsd/MsgSender.py yys "EXR MOV OK" younae.hong'
rootTask.addCommand(author.Command(argv=cmd, service=SERVICE_KEY))
job.addChild(rootTask)

for dirName in os.listdir(targetDir):
    exrDir = os.path.join(targetDir, dirName)
    if os.path.isdir(exrDir):
        task = author.Task(title=exrDir)
        cmd = '/backstage/bin/DCC rez-env ffmpeg nuke-10 -- nukeX -i -t /backstage/dcc/packages/ext/otiotoolkit/examples/tmp/exrToMovUsingNuke.py --exrDir %s' % exrDir
        task.addCommand(author.Command(argv=cmd, service=SERVICE_KEY))

        print cmd
        rootTask.addChild(task)

job.priority = 100
author.setEngineClientParam(hostname=TRACTOR_IP, port=PORT, user=getpass.getuser(), debug=True)
job.spool()
print job.asTcl()
author.closeEngineClient()
