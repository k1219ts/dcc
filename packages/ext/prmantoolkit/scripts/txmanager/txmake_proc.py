#encoding=utf-8
from __future__ import print_function
#--------------------------------------------------------------------------------
#
#    Dexter CG-Supervisor
#
#        Sanghun Kim, rman.td@gmail.com
#
#    rman.td 2016.04.02 $2
#
#-------------------------------------------------------------------------------

import os, sys
import re
import glob
import time
import shutil
import subprocess
import platform

# import #mari

class TxMake:
    def __init__( self, files ):
        self.files = files
        self.threads = 8
        self.Events = None

    def process( self, source, target ):
        cmd  = 'txmake -t:8'
        cmd += ' -smode periodic -tmode periodic'
        cmd += ' -pattern diagonal'
        cmd += ' -resize up-'
        cmd += ' -format pixar'
        cmd += ' -newer'
        cmd += ' -verbose'
        cmd += ' %s %s' % (source, target)
        #print source + '\n'
        pipe = subprocess.Popen( cmd, shell=True )
        return pipe

    def getFileStruct( self ):
        result = list()
        step = len(self.files) / self.threads
        for i in range(step):
            result.append( list() )
            for x in range(self.threads):
                index = i*self.threads + x
                result[i].append( self.files[index] )
        if step*self.threads != len(self.files):
            result.append( list() )
            for i in range(step*self.threads,len(self.files)):
                result[step].append( self.files[i] )
        return result

    def getOutFile( self, inputfile ):
        dirname  = os.path.dirname( inputfile )
        basename = os.path.basename( inputfile )
        basename = os.path.splitext(basename)[0]

        # original : texture/pub/tex/v01/filename.tex
        # original : texture/pub/v01/filename.jpg
        # new : texture/tex/v001/filename.tex
        # new : texture/images/v001/filename.tex

        vcheck = re.compile(r'v(\d+)').findall( inputfile )
        pubPath = os.path.dirname(os.path.dirname(dirname))
        versionName = dirname.split('/')[-1]
        outdir = os.path.join( pubPath, 'tex', versionName)
        print(os.path.join( pubPath, 'tex', versionName))
        if not os.path.exists( outdir ):
            os.makedirs( outdir )
        return os.path.join( outdir, '%s.tex' % basename )

    def cversionCheckCopy( self, outfiles ):
        current_dir = os.path.dirname( outfiles[0] )
        current_version_files = list()
        for i in os.listdir(current_dir):
            if os.path.isfile(os.path.join(current_dir,i)) and os.path.splitext(i)[-1] == '.tex':
                current_version_files.append( i )
        vcheck = re.compile(r'v(\d+)').findall( current_dir )
        if vcheck:
            previous_dir = os.path.join( os.path.dirname(current_dir), 'v%02d' % (int(vcheck[0])-1) )
            #print 'previous_version_dir : %s' % previous_dir
            if os.path.exists( previous_dir ):
                if self.Events:
                    self.Events.progress_text.setText( 'previous version file copying ...' )
                    self.Events.pbar.setValue( 95 )
                    #mari.app.processEvents()
                ifever = 0
                for i in os.listdir(previous_dir):
                    if os.path.isfile(os.path.join(previous_dir,i)) and os.path.splitext(i)[-1] == '.tex':
                        if not i in current_version_files:
                            ifever += 1
                            shutil.copy2( os.path.join(previous_dir,i), current_dir )
                if ifever:
                    return True

    def convert( self ):
        result = list()
        fileStruct = self.getFileStruct()
        for t in fileStruct:
            procList = list()
            outFiles = list()
            for f in t:
                #print 'input  : %s' % f
                #print 'output : %s' % self.getOutFile( f )
                outfile = self.getOutFile( f )
                proc = self.process( f, outfile )
                procList.append( proc )
                outFiles.append( outfile )
                result.append( outfile )
                yield True
            stopper = [0] * len(t)
            while True:
                for i in range(len(procList)):
                    if procList[i].poll() != None:
                        if stopper[i] != 1:
                            stopper[i] = 1
                            if self.Events:
                                self.Events.progress_text.setText( 'txmake : %s' \
                                        % os.path.basename(outFiles[i]) )
                                #mari.app.processEvents()
                if not 0 in stopper:
                    break
        yield False
