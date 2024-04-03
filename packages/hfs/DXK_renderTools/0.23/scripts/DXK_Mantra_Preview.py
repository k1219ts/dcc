import os
import re
import hou
import datetime
import subprocess

class DXK_Mantra_Preview(object):
    def __init__(self, node):
        self.p = dict()  #storing interface parameters
        self.node = node
        self.userName = "test"

        self.getHoudiniParms()
        # self.checkParameters()

    def getHoudiniParms(self):
        self.DEPARTMENT = self.node.parm("DEPARTMENT").evalAsString()
        self.SHOW = self.node.parm("SHOW").evalAsString().lower()
        self.SEQ = self.node.parm("SEQ").evalAsString()#.upper()
        self.SHOT = self.node.parm("SHOT").evalAsString()#.upper()
        self.ELEMENT_NAME = self.node.parm("ELEMENT_NAME").eval()
        self.HIPNAME = self.node.parm("HIPNAME").eval()
        self.HIPNAMESPLIT = self.HIPNAME.split('-')[0]

        self.IS_SEQUENCE = self.node.parm("trange").eval()
        self.STARTFRAME = self.node.parm("f1").eval()
        self.ENDFRAME = self.node.parm("f2").eval()
        self.INCREMENT = self.node.parm("f3").eval()

    # def checkParameters(self):
    #     if re.match("v[0-9][0-9][0-9]", self.ELEMENT_VERSION) is None or len(self.ELEMENT_VERSION) != 4:
    #         print "Incorrect version"

    def getShowList(self):
        sho = []
        shows = os.listdir("/show/")
        shows.sort()

        for s in shows:
            sho.append(s)
            sho.append(s)
        return sho

    def getSequenceList(self):
        seq = []
        sequence_path = os.path.join('/show', self.node.parm('SHOW').evalAsString(), 'shot')
        if not os.path.exists(sequence_path):
            return seq

        sequences = os.listdir(sequence_path)
        sequences.sort()

        for s in sequences:
            seq.append(s)
            seq.append(s)
        return seq

    def getShotList(self):
        shot = []
        shot_path = os.path.join('/show', self.node.parm('SHOW').evalAsString(), 'shot', self.node.parm('SEQ').evalAsString())
        if not os.path.exists(shot_path):
            return shot

        shots = os.listdir(shot_path)
        shots.sort()

        for s in shots:
            shot.append(s)
            shot.append(s)
        return shot

    @property
    def fileName(self):
        return self.HIPNAMESPLIT + "_" + self.ELEMENT_NAME

    @property
    def frameNumber(self):
        return str(int(hou.frame())).zfill(4)

    @property
    def renderPath(self):
        renderPath = self.renderDirPath

        if self.IS_SEQUENCE:
            renderPath += self.fileName + "." + self.frameNumber + ".jpg"
        else:
            renderPath += self.fileName + ".jpg"

        return renderPath

    @property
    def renderDirPath(self):
        path = "/show"
        path += "/" + self.SHOW
        path += "/shot"
        path += "/" + self.SEQ
        path += "/" + self.SHOT
        path += "/" + self.DEPARTMENT
        path += "/dev/scenes/houdini/preview/jpg/" + self.HIPNAMESPLIT +"/"

        return path

    @property
    def preRenderScript(self):
        renderPath = self.renderDirPath
        movDirPath = renderPath.split("/")
        movDirPath = "/".join(movDirPath[:-2])
        movDirPath = movDirPath.replace('jpg', 'mov')
        script = "umkdir -p "
        script += movDirPath + "/"
        return script

    @property
    def postRenderScript(self):
        renderPath = self.renderDirPath
        movDirPath = renderPath.split("/")
        movDirPath = "/".join(movDirPath[:-2])
        movDirPath = movDirPath.replace('jpg', 'mov')
        movPath = movDirPath + "/" + self.fileName + ".mov"
        if self.IS_SEQUENCE:
            renderPath += self.fileName + ".%04d.jpg"
        else:
            renderPath += self.fileName + ".jpg"

        script = "/backstage/bin/DCC rez-env "
        script += "ffmpeg-4.2.1 --ffmpeg"
        script += "-framerate 24 "
        script += "-start_number " + str(int(self.STARTFRAME))
        script += " -i " + '\"' + renderPath + '\"'
        # script += " -f mp4 "
        script += " -vcodec libx264 "
        # script += "-s hd1080 "
        script += "-b:v 10000k "
        script += "-pix_fmt yuv420p "
        script += '\"'+ movPath + '\"'
        return script

    def openFileBrowser(self):
        #os.system("nautilus " + self.renderDirPath)
        renderPath = self.renderDirPath
        subprocess.Popen(["/usr/bin/nautilus", renderPath])

    def openMplay(self):
        rp = self.renderPath[:-8] + "*"
        print rp
        os.system("$HFS/bin/mplay " + rp)





