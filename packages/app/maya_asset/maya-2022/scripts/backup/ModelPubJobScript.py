#encoding=utf-8
#!/usr/bin/env python

#-------------------------------------------------------------------------------
#
#   RenderMan TD
#
#       Sanghun Kim, rman.td@gmail.com
#
#       2015.06.03 $1
#-------------------------------------------------------------------------------

import os, sys
import getpass
import subprocess
import site

#if sys.platform.find('win') > -1:
#    TractorRoot = 'N:/backstage/pub/apps/tractor/win64/Tractor-2.0'
#    site.addsitedir( '%s/lib/python2.7/Lib/site-packages' % TractorRoot )
#else:
#    TractorRoot = '/netapp/backstage/pub/apps/tractor/linux/Tractor-2.0'
#    site.addsitedir( '%s/lib/python2.7/site-packages' % TractorRoot )
#ScriptRoot  = '/netapp/backstage/pub/apps/maya/2014/team/modeling/linux/scripts'

import dxConfig

site.addsitedir( dxConfig.getConf('TRACTOR_API') )

ScriptRoot = os.path.dirname( os.path.abspath(__file__) )
'''
testPath
# ScriptRoot = '/dexter/Cache_DATA/RND/daeseok/maya2/versions/2017/team/linux/scripts'
'''

import tractor.api.author as author

import maya.cmds as cmds
import maya.mel as mel

#options = {
#       'm_basename': 'filename',
#       'm_envkey': 'rms-19.0-maya-2014',
#       'm_svckey': 'Cent7',
#       'm_maxactive': 1,
#       'm_mayafile': 'filename.mb'
#               }

class JobMain:
    def __init__( self, options ):
        self.options = options

    def modelPub_jobscript( self ):
        job = author.Job()
        job.title       = '(ModelPub) ' + str( self.options['m_basename'] )
        job.comment     = ''
        job.metadata    = ''
        job.envkey      = [ 'rfm2-21.4-maya-2017' ]
        job.service     = 'Cent7'
        job.maxactive   = 1
        job.tier        = 'user'
        job.projects    = ['lgt']
        job.tags        = ['user']

        # directory mapping
        job.newDirMap( src='X:/', dst='/show/', zone='NFS' )
        job.newDirMap( src='N:/', dst='/netapp/', zone='NFS' )
        job.newDirMap( src='R:/', dst='/dexter/', zone='NFS' )

        JobTask = author.Task( title='Job' )
        JobTask.serialsubtasks = 1

        ScriptTask = author.Task( title='batchScript' )
        command = [ 'mayapy', '%%D(%s/ModelPub.py)' % ScriptRoot,
                                '%%D(%s)' % self.options['m_mayafile'],
                                '%%D(%s)' % self.options['m_exportfile'],
                                self.options['m_maya'],
                                self.options['m_abc'],
                                self.options['m_tex'],
                                self.options['m_expGroup'] ]
        ScriptTask.addCommand(
                        author.Command( argv=command, service='RfMRender', tags=['py'] )
                        )
        JobTask.addChild( ScriptTask )

        job.addChild( JobTask )

        job.priority = 1000
        author.setEngineClientParam( hostname=dxConfig.getConf('TRACTOR_CACHE_IP'),
                                     port=dxConfig.getConf('TRACTOR_PORT'),
                                     user=getpass.getuser(), debug=True )
#        author.setEngineClientParam( hostname='10.0.0.30', port=80, user=getpass.getuser(), debug=True )
        job.spool()
        author.closeEngineClient()

        return job.asTcl()

    def doIt( self ):
        # job script
        tclscript = self.modelPub_jobscript()
        #print tclscript
