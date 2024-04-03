import os
import re
import hou
import datetime
import subprocess

class DXK_Mantra_Simple(object):
    def __init__(self, node):
        self.p = dict()  #storing interface parameters
        self.node = node
        self.userName = "test"

        self.getHoudiniParms()
        self.checkParameters()

    def getHoudiniParms(self):
        self.DEPARTMENT = self.node.parm("DEPARTMENT").evalAsString()
        self.SHOW = self.node.parm("SHOW").evalAsString().lower()
        self.SEQ = self.node.parm("SEQ").evalAsString()#.upper()
        self.SHOT = self.node.parm("SHOT").evalAsString()#.upper()
        self.FX_GROUP = self.node.parm("FX_GROUP").eval()
        self.ELEMENT_NAME = self.node.parm("ELEMENT_NAME").eval()
        self.ELEMENT_VERSION = self.node.parm("ELEMENT_VERSION").eval()

        self.IS_SEQUENCE = self.node.parm("trange").eval()
        self.STARTFRAME = self.node.parm("f1").eval()
        self.ENDFRAME = self.node.parm("f2").eval()
        self.INCREMENT = self.node.parm("f3").eval()

    def checkParameters(self):
        if re.match("v[0-9][0-9][0-9]", self.ELEMENT_VERSION) is None or len(self.ELEMENT_VERSION) != 4:
            print "Incorrect version"

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
        sequence_path = '/show/' + self.node.parm('SHOW').evalAsString() + '/works/PFX/shot'
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
        shot_path = '/show/' + self.node.parm('SHOW').evalAsString() + '/works/PFX/shot/' + self.node.parm('SEQ').evalAsString()
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
        ##return self.SHOT + "__" + self.FX_GROUP + "__" + self.ELEMENT_NAME + "_" + self.ELEMENT_VERSION
        return self.SHOT + "_" + self.ELEMENT_NAME + "_" + self.ELEMENT_VERSION

    @property
    def frameNumber(self):
        return str(int(hou.frame())).zfill(4)

    @property
    def renderPath(self):
        renderPath = self.renderDirPath
        renderPath += self.FX_GROUP + "/" + self.ELEMENT_VERSION + "/"

        if self.IS_SEQUENCE:
            renderPath += self.fileName + "." + self.frameNumber + ".exr"
        else:
            renderPath += self.fileName + ".exr"

        return renderPath

    @property
    def renderDirPath(self):
        path = "/show"
        path += "/" + self.SHOW
        path += "/_2d/shot"
        path += "/" + self.SEQ
        path += "/" + self.SHOT
        path += "/" + self.DEPARTMENT
        path += "/dev/images/"

        return path

    def openFileBrowser(self):
        #os.system("nautilus " + self.renderDirPath)
        renderPath = self.renderDirPath
        subprocess.Popen(["/usr/bin/nautilus", renderPath])

    def openMplay(self):
        rp = self.renderPath[:-8] + "*"
        print rp
        os.system("$HFS/bin/mplay " + rp)


