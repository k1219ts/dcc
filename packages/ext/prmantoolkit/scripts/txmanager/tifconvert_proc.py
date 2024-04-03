#encoding=utf-8
#--------------------------------------------------------------------------------
#
#    Dexter RND
#
#        Daeseok Chae, cds7031@gmail.com
#
#    daeseok.chae 2018.12.05
#
#-------------------------------------------------------------------------------

import os
import glob
import subprocess
import platform

class TifMake:
    def __init__( self, files, isJpg = False ):
        self.files = files
        self.threads = 8
        self.Events = None
        self.isJpg = isJpg
        # Environment setup
        if platform.system() == 'Windows':
            pixar_apps = glob.glob( 'C:\Program Files\Pixar\RenderManProServer*' )
            pixar_apps.sort()
            os.environ['PATH'] += ';%s\\bin' % pixar_apps[-1]
        else:
             pixar_apps = glob.glob( '/opt/pixar/RenderManProServer*' )
             pixar_apps.sort()
             os.environ['PATH'] += ':%s/bin' % pixar_apps[-1]

    def process( self, source, target ):
        cmd  = 'txmake -t:8'
        cmd += ' -smode periodic -tmode periodic'
        cmd += ' -pattern diagonal'
        cmd += ' -format tiff'
        cmd += ' -compression LZW'
        cmd += ' -byte'
        cmd += ' -resize up-'

        cmd += ' %s %s' % (source, target)
        print cmd
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
        dirname  = os.path.dirname( inputfile ) # version dir
        basename = os.path.basename( inputfile ) # filename
        basename = os.path.splitext(basename)[0] # removing extension from filename

        pubPath = os.path.dirname(os.path.dirname(dirname)) # 'texture' directory
        versionName = dirname.split('/')[-1]
        outdir = os.path.join( pubPath, 'images', versionName)
        print os.path.join( pubPath, 'images', versionName)
        if not os.path.exists( outdir ):
            os.makedirs( outdir )
        return os.path.join( outdir, '%s.tif' % basename )

    # def versionCheckCopy( self, outfiles ):
    #     current_dir = os.path.dirname( outfiles[0] )
    #     current_version_files = list()
    #     for i in os.listdir(current_dir):
    #         if os.path.isfile(os.path.join(current_dir,i)) and os.path.splitext(i)[-1] == '.tex':
    #             current_version_files.append( i )
    #     vcheck = re.compile(r'v(\d+)').findall( current_dir )
    #     if vcheck:
    #         previous_dir = os.path.join( os.path.dirname(current_dir), 'v%03d' % (int(vcheck[0])-1) )
    #         if os.path.exists( previous_dir ):
    #             if self.Events:
    #                 self.Events.progress_text.setText( 'previous version file copying ...' )
    #                 self.Events.pbar.setValue( 95 )
    #             ifever = 0
    #             for i in os.listdir(previous_dir):
    #                 if os.path.isfile(os.path.join(previous_dir,i)) and os.path.splitext(i)[-1] == '.tex':
    #                     if not i in current_version_files:
    #                         ifever += 1
    #                         shutil.copy2( os.path.join(previous_dir,i), current_dir )
    #             if ifever:
    #                 return True

    def convert( self ):
        result = list()
        self.outFiles = list()
        fileStruct = self.getFileStruct()
        for t in fileStruct:
            procList = list()
            for f in t:
                outfile = self.getOutFile( f )
                proc = self.process( f, outfile )
                procList.append( proc )
                self.outFiles.append( outfile )
                result.append( outfile )
                yield True
            stopper = [0] * len(t)
            while True:
                for i in range(len(procList)):
                    if procList[i].poll() != None:
                        stopper[i] = 1

                if not 0 in stopper:
                    break

        yield False
