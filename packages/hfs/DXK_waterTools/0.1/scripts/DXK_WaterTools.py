import os
import re
import hou
import json
import subprocess


class WaterTools(object):
    def __init__(self, node):

        self.node = node
        self.getHoudiniParms()

    ################## SET OUTPUT ##################

    def getHoudiniParms(self):
        self.DEPARTMENT = self.node.parm("DEPARTMENT").evalAsString()
        self.SHOW = self.node.parm("SHOW").evalAsString().lower()
        self.SEQ = self.node.parm("SEQ").evalAsString()#.upper()
        self.SHOT = self.node.parm("SHOT").evalAsString()#.upper()
        self.ELEMENT_NAME = self.node.parm("ELEMENT_NAME").eval()

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
        return "ocean"

    @property
    def frameNumber(self):
        return str(int(hou.frame())).zfill(4)

    @property
    def renderPath(self):
        renderPath = self.renderDirPath
        renderPath += self.ELEMENT_NAME + "/"

        if self.IS_SEQUENCE:
            renderPath += self.fileName + "." + self.frameNumber + ".exr"
        else:
            renderPath += self.fileName + ".exr"

        return renderPath

    @property
    def renderDirPath(self):
        path = "/show"
        path += "/" + self.SHOW
        path += "/shot"
        path += "/" + self.SEQ
        path += "/" + self.SHOT
        path += "/" + self.DEPARTMENT
        path += "/dev/ocean_source/"

        return path

    def openMplay(self):
        rp = self.renderPath[:-8] + "*"
        print (rp)
        os.system("$HFS/bin/mplay " + rp)

    def openFileBrowser(self):
        renderPath = self.renderDirPath
        subprocess.Popen(["/usr/bin/nautilus", renderPath])

    ################## SET JSON ##################

    def saveParams(self):
        ### json data set
        data = {}
        data['pxr'] = {}


        ### set jsonPath
        mapPath = self.node.parm("copoutput").eval()
        jsonPath = "/".join(mapPath.split("/")[:-1])
        # print jsonPath

        if not os.path.exists(jsonPath):
            os.makedirs(jsonPath)

        ### set fileName
        fileName = "/".join(mapPath.split("/")[-1:])
        fileName = ".".join(fileName.split(".")[:1])

        jsonFilePath = jsonPath + '/' + fileName + '.spectrum'
        print ("JSON PATH : " + jsonFilePath)

        ### get parms from ocean spectrum
        oceanspectrum = self.node.inputs()[1]
        for parm in oceanspectrum.parms():
            if parm.name()=='gridsize':
                name = parm.name()
                gridvalue = parm.eval()

                ### add data
                data['pxr'] = {
                    name : gridvalue
                }


            elif parm.name()=='loopperiod':
                name = parm.name()
                durationvalue = parm.eval() * hou.fps()

                ### add data
                data['pxr']['duration'] = durationvalue


        data['pxr']['fileName'] = fileName

        ### write json
        with open(jsonFilePath, 'w') as outfile:
            json.dump(data, outfile, indent=4)


