#encoding=utf-8
#--------------------------------------------------------------------------------
#
#    Dexter CG-Supervisor
#
#        Sanghun Kim, rman.td@gmail.com
#
#    rman.td 2016.06.29 $1
#
#-------------------------------------------------------------------------------

import os, sys
import re
import string
import shutil


def versionCheckCopy( fileNames ):
    current_dir = os.path.dirname( fileNames[0] )
    current_version_files = list()
    for f in os.listdir( current_dir ):
        if os.path.isfile( os.path.join(current_dir,f) ):
            current_version_files.append( f )

    verCheck = re.compile(r'v(\d+)').findall( os.path.basename(current_dir) )

    print verCheck, current_dir, current_version_files
    if verCheck:
        ifever = 0
        print '# Result : Previous version file copy'

        rootPath = os.path.dirname( current_dir )
        version_dirs = list()
        for f in os.listdir( rootPath ):
            if os.path.isdir( os.path.join(rootPath, f) ) and re.compile(r'v(\d+)').findall(f):
                version_dirs.append( f )
        version_dirs.sort()

        previous_ver = version_dirs[ version_dirs.index('v%03d' % int(verCheck[-1]))-1 ]
        previous_dir = os.path.join( rootPath, previous_ver )
        for f in os.listdir( previous_dir ):
            filename = os.path.join( previous_dir, f )

            if os.path.isfile( filename ):
                if not f in current_version_files:
                    ifever += 1
                    print '# Debug : %s -> %s' % ( filename, current_dir )
                    shutil.copy2( filename, current_dir )
        if ifever:
            return True

