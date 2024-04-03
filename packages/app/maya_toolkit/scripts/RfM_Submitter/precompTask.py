import os
from config import *
import optparse
from TractorEngine import *
import pprint


def getNukeVersion(nukefile):
    fileVersion = open(nukefile).readline().split('/')[3]
    # Nuke10.0v4
    return fileVersion


def precompTask(nkfile, lgtPath, Parent=None, ):
    """
    PreComp Spool Task
    """
    # IF FILE IS NOT EXISTS THEN RETURN
    if not (os.path.exists(nkfile)):
        return

    task = author.Task(title='PreComp Spool')
    nukeVer = getNukeVersion(nkfile)
    nukeexec = nukeVer.split('v')[0]
    prcScript = '/netapp/backstage/pub/apps/maya2/global/RfM_Submitter/precompRender.py'
    """
    /usr/local/Nuke10.0v4/Nuke10.0 
    -t /netapp/backstage/pub/apps/maya2/global/RfM_Submitter/precompRender.py 
    -n /show/god/asset/global/user/taehyung.lee/compbot/MAT_0840_comp_v003.nk 
    -i /show/god/shot/MAT/MAT_0840/lighting/pub/images/MAT_0840_rnd_v99_w99 
    -v
    """
    command = ['/usr/local/%s/%s' % (nukeVer, nukeexec)]
    command += ['-t', prcScript]
    command += ['-n', nkfile]
    command += ['-i', lgtPath]
    command += ['-v']
    task.addCommand(
        author.Command(service='PixarRender', argv=command, atleast=1)
    )
    Parent.addChild(task)
    return task