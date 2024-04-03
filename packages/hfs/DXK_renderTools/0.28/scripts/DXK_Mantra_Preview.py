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
        path += "/works/PFX/shot"
        path += "/" + self.SEQ
        path += "/" + self.SHOT
        path += "/scenes/houdini/preview/jpg/" + self.HIPNAMESPLIT + "/" + self.ELEMENT_NAME +"/"

        return path

    @property
    def movPath(self):
        renderPath = self.renderDirPath
        movDirPath = renderPath.split("/")
        movDirPath = "/".join(movDirPath[:-3])
        movDirPath = movDirPath.replace('jpg', 'mov')
        movPath = movDirPath + "/" + self.fileName + ".mov"
        return movPath

    def openFileBrowser(self):
        #os.system("nautilus " + self.renderDirPath)
        renderPath = self.renderDirPath
        subprocess.Popen(["/usr/bin/nautilus", renderPath])

    def openMplay(self):
        rp = self.renderPath[:-8] + "*"
        print rp
        os.system("$HFS/bin/mplay " + rp)






