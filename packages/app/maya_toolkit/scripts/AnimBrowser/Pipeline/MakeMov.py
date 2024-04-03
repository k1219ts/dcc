import pymodule.Qt as Qt

if "Side" in Qt.__binding__:
    import maya.cmds as cmds
    import maya.mel as mel
    
import subprocess
from subprocess import CalledProcessError
import os
import MessageBox as MessageBox

class MakeMov():
    def __init__(self, baseDirPath, category, tier1Text, tier2Text, tier3Text, fileNum):
        self.baseDirPath = baseDirPath
        self.category = category
        self.tier1Path = tier1Text
        self.tier2Path = tier2Text
        self.tier3Path = tier3Text
        self.fileNum = fileNum
        self.outputPath = "{0}/{1}/{2}/{3}/{4}/{5}/{2}_{3}_{4}".format(self.baseDirPath, self.category, self.tier1Path, self.tier2Path, self.tier3Path, self.fileNum)
        
        self.updateData()
        # self.takePlayblast()
        # self.frameCount = self.movToGif()
  
    def updateData(self):
        self.startTime = int(cmds.playbackOptions(q=True, minTime=True))
        self.endTime = int(cmds.playbackOptions(q=True, maxTime=True))

        self.width = cmds.getAttr("defaultResolution.width")  # int
        self.height = cmds.getAttr("defaultResolution.height")  # int

        self.width = 1280
        self.height = 720
        self.scale = 100

        # reset file output
        cmds.setAttr("defaultRenderGlobals.imageFormat", 8)  # JPG
#         cmds.setAttr("defaultRenderGlobals.imageFormat", 7)  # IFF
        
    def takePlayblast(self):
        cmds.playblast(format = "image", clearCache = True,
                       viewer = False, showOrnaments = True,
                       f = self.outputPath, 
                       framePadding = 4, percent = self.scale,
                       widthHeight = [self.width, self.height])

        outputFolder = os.path.dirname(self.outputPath)

        mpegCommand = ['-r', '24']

        if self.startTime != 0:
           mpegCommand.append('-start_number')
           mpegCommand.append('{0}'.format(str(self.startTime)))

        mpegCommand2 = ['-i', '{0}.%04d.jpg'.format(self.outputPath),
                       '-r', '24',
                       '-an', '-vcodec',
                       'libx264', '-pix_fmt',
                       'yuv420p', '-preset',
                       'slow', '-profile:v',
                       'baseline', '-b:a',
                       '6000k', '-tune',
                       'zerolatency', '-y',
                       '{0}/{1}_{2}_{3}.mov'.format(outputFolder, self.tier1Path, self.tier2Path, self.tier3Path)]

        args = ['/opt/ffmpeg/bin/ffmpeg'] + mpegCommand + mpegCommand2

        _env = os.environ.copy()
        _env['LD_LIBRARY_PATH'] = "/opt/ffmpeg/lib:" + _env['LD_LIBRARY_PATH']
        subprocess.Popen( args, env=_env ).wait()

        return '{0}/{1}_{2}_{3}.mov'.format(outputFolder, self.tier1Path, self.tier2Path, self.tier3Path)

    def movToGif(self, movFileName):
        print "movToGif : ", movFileName

        movFileName = movFileName.replace('.mov', '')
        outputFolder = os.path.dirname(self.outputPath)
        # frameCount = len(os.listdir(outputFolder))

        if not os.path.isdir(outputFolder):
            os.makedirs(outputFolder)

        gifFileName = "{0}/{1}_{2}_{3}".format(outputFolder, self.tier1Path, self.tier2Path, self.tier3Path)

        makeGifCommand = ["/opt/ffmpeg/bin/ffmpeg",
                          "-i", '"%s.mov"' % movFileName,
                          "-vf", "scale=1280:-1,format=rgb8,format=rgb24",
                          "-r", "10",
                          "-f", "image2pipe",
                          "-vcodec", "ppm", "-", "|",
                          "convert", "-delay", "5",
                          "-loop", "0", "-", "gif:-",'"%s.gif"' % gifFileName]
                          # "|", "convert", "-layers", "Optimize", "-",

        cmd = " ".join(makeGifCommand)
        #
        # print "envrionment : ", os.environ['LD_LIBRARY_PATH']
        print cmd
        # print "result :", os.system(cmd)

        os.environ['LD_LIBRARY_PATH'] = "/opt/ffmpeg/lib:" + os.environ['LD_LIBRARY_PATH']

        os.system(cmd)

        # try:
        #     o = subprocess.check_output(makeGifCommand, stderr=subprocess.STDOUT)
        #     returncode = 0
        # except CalledProcessError as ex:
        #     print ex.output
        #     print ex

        self.gifFilePath = "%s.gif" % gifFileName
        return self.gifFilePath

        # self.frameCount = frameCount
