#encoding=utf-8
#!/usr/bin/env python

import os
import getpass
import sys
import dxConfig
sys.path.append(dxConfig.getConf('TRACTOR_API')) # '/netapp/backstage/pub/apps/tractor/linux/Tractor-2.2/lib/python2.7/site-packages')

currentDir = os.path.dirname(__file__)

import tractor.api.author as author

inputFile = sys.argv[1]
reduceList = [70, 50, 15, 5, 2] # here

job = author.Job()
job.title       = '(HOU) Reduce--%s' % (os.path.splitext(os.path.basename(inputFile))[0])
job.service     = 'houdini'
job.maxactive   = 5
job.tier        = 'as_other'
job.projects    = ['convert']
# job.tags        = ['py']

# directory mapping
job.newDirMap( src='S:/', dst='/show/', zone='NFS' )
job.newDirMap( src='N:/', dst='/netapp/', zone='NFS' )
job.newDirMap( src='R:/', dst='/dexter/', zone='NFS' )

for reduceValue in reduceList:
    JobTask = author.Task( title='Reduce Processing' )
    command = "DCC rez-env houdini-17.5.229 pylibs -- hython %s/reduceModel.py -input %s -reduce %d" % (currentDir, inputFile, reduceValue)
    JobTask.addCommand(
	            author.Command( argv=command, service='houdini')
	            )

    job.addChild( JobTask )
job.priority = 100

author.setEngineClientParam( hostname="10.0.0.35",
                             port=80,
                             user=getpass.getuser(),
                             debug=True )
job.spool()
author.closeEngineClient()

job.asTcl()
