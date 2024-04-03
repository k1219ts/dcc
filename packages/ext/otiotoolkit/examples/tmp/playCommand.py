import os
import subprocess

targetDir = '/show/yys/stuff/ftp/_257/to_dexter/20200922/comp/exr/OK'

title = 'EXR MOV'

for dirName in os.listdir(targetDir):
    exrDir = os.path.join(targetDir, dirName)
    if os.path.isdir(exrDir):
        cmd = '/backstage/dcc/DCC rez-env ffmpeg nuke-10 -- nukeX -i -t /backstage/dcc/packages/ext/otiotoolkit/examples/tmp/exrToMovUsingNuke.py --exrDir %s' % exrDir
        print cmd
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        while p.poll() is None:
            output = p.stdout.readline()
            if output:
                print output.strip()

