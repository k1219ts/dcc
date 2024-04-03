import os
import re
import hou
import datetime
import subprocess
import json

class Publish(object):
    def __init__(self, node):
        self.p = dict()  #storing interface parameters
        self.node = node
        self.cachePath = ""
        self.hipFile = hou.hipFile.path()
        self.JSONPath = ""
        # self.userName = os.getlogin()
        self.userName = "test"
        self.dateStamp = ""

        self.getHoudiniParms()
        self.checkParameters()

    def getHoudiniParms(self):
        self.DEPARTMENT = self.node.parm("DEPARTMENT").evalAsString()
        self.SHOW = self.node.parm("SHOW").evalAsString().lower()
        self.SEQ = self.node.parm("SEQ").evalAsString().upper()
        self.SHOT = self.node.parm("SHOT").evalAsString().upper()
        self.FX_GROUP = self.node.parm("FX_GROUP").eval().lower()
        self.PUBLISH_VERSION = self.node.parm("PUBLISH_VERSION").eval().lower()
        self.USER_COMMENT = self.node.parm("USER_COMMENT").eval()

        self.cacheList = []
        self.matList = []
        self.geoList = []
        self.mantraList = []
        self.ropnetList = []

        numCaches = self.node.parm("caches").eval()
        if numCaches:
            for x in range(numCaches):
                cache = self.node.parm("fileIO_" + str(x+1)).eval()
                self.cacheList.append(cache)

        numMats = self.node.parm("materials").eval()
        if numMats:
            for x in range(numMats):
                mat = self.node.parm("mat_" + str(x+1)).eval()
                self.matList.append(mat)

        numGeos = self.node.parm("geos").eval()
        if numGeos:
            for x in range(numGeos):
                geo = self.node.parm("geo_" + str(x+1)).eval()
                self.geoList.append(geo)

        numMantras = self.node.parm("mantras").eval()
        if numMantras:
            for x in range(numMantras):
                mantra = self.node.parm("mantra_" + str(x+1)).eval()
                self.mantraList.append(mantra)

        numRopnets = self.node.parm("ropnets").eval()
        if numRopnets:
            for x in range(numRopnets):
                ropnet = self.node.parm("ropnet_" + str(x+1)).eval()
                self.ropnetList.append(ropnet)

    def publish(self):
        if not os.path.exists(self.assetDirPath):
            self.createDirectories()
            self.makeDateStamp()
            self.saveHipFile()
            self.saveJSONFile()
            self.createHDA()
        else:
            print "Publish version already exists, please increment!"

    def createHDA(self):
        try:

            assetName = self.SHOW + "_" + self.SEQ + "_" +self.SHOT + "_" + self.FX_GROUP + "_" + self.PUBLISH_VERSION
            subnet = hou.node("/obj").createNode("subnet", assetName)
            cacheContainer = subnet.createNode("geo", "Caches")
            cacheContainer.moveToGoodPosition()
            cacheContainer.children()[0].destroy()
            matContainer = subnet.createNode("matnet", "Materials")
            matContainer.moveToGoodPosition()
            ropContainer = subnet.createNode("ropnet", "ROPNetwork")
            ropContainer.moveToGoodPosition()

            for cache in self.cacheList:
                cacheNode = hou.node(cache)
                cacheContainer.copyItems((cacheNode,))

            for mat in self.matList:
                matNode = hou.node(mat)
                matContainer.copyItems((matNode,))

            for geo in self.geoList:
                geoNode = hou.node(geo)
                copiedGeo = subnet.copyItems((geoNode,))
                for geo in copiedGeo:
                    geo.moveToGoodPosition()

            for mantra in self.mantraList:
                mantraNode = hou.node(mantra)
                ropContainer.copyItems((mantraNode,))

            for ropnet in self.ropnetList:
                ropnetNode = hou.node(ropnet)
                copiedRopnet = subnet.copyItems((ropnetNode,))
                for ropnet in copiedRopnet:
                    ropnet.moveToGoodPosition()



            asset = subnet.createDigitalAsset(name="PUBLISH_" + assetName,
                                                hda_file_name = self.assetDirPath + "/" + assetName + ".hda",
                                                description = "PUBLISH_" + assetName,
                                                min_num_inputs = 0, max_num_inputs = 0,
                                                comment = self.USER_COMMENT,
                                                version = self.PUBLISH_VERSION,
                                                change_node_type = True,
                                                ignore_external_references = True)
        except Exception as e:
            print "PUBLISH FAILED", e

    def createDirectories(self):

        if subprocess.check_output(['mkdir', '-p', self.assetDirPath]):
            print "Directory creation failed"

    def checkParameters(self):

        if re.match("v[0-9][0-9][0-9]", self.PUBLISH_VERSION) is None or len(self.PUBLISH_VERSION) != 4:
            print "Incorrect publish version"

    def getShowList(self):
        sho = []
        shows = os.listdir("/show/")
        shows.sort()

        for s in shows:
            sho.append(s)
            sho.append(s)
        return sho

    def getSequenceList(self):
        if self.node.parm("SHOW"):
            seq = []
            seq_path = '/show/' + self.node.parm('SHOW').evalAsString() + '/shot'
            if os.path.exists(seq_path):
                sequences = os.listdir(seq_path)
                sequences.sort()
                for s in sequences:
                    seq.append(s)
                    seq.append(s)
            return seq
        else:
            return list()

    def getShotList(self):
        if self.node.parm("SHOW") and self.node.parm("SEQ"):
            shot = []
            shot_path = '/show/' + self.node.parm('SHOW').evalAsString() + '/shot/' + self.node.parm('SEQ').evalAsString() + '/'
            if os.path.exists(shot_path):
                shots = os.listdir(shot_path)
                shots.sort()
                for s in shots:
                    shot.append(s)
                    shot.append(s)
            return shot
        else:
            return list()

    def makeDateStamp(self):
        self.dateStamp = "".join(str(datetime.datetime.now()).split(".")[:-1])
        self.dateStamp = "-".join(self.dateStamp.split(" "))

    @property
    def assetDirPath(self):
        assetDirPath = "/show"
        assetDirPath += "/" + self.SHOW
        assetDirPath += "/shot"
        assetDirPath += "/" + self.SEQ
        assetDirPath += "/" + self.SHOT
        assetDirPath += "/" + self.DEPARTMENT
        assetDirPath += "/pub"
        assetDirPath += "/" + self.FX_GROUP
        assetDirPath += "/" + self.PUBLISH_VERSION
        return assetDirPath


    def buildJSON(self):
        JSONDic = {
            "userName": self.userName,
            "userComment": self.USER_COMMENT
        }

        return JSONDic

    def saveJSONFile(self):

        JSONDir = self.assetDirPath
        JSONName = hou.hscriptExpandString("$HIPNAME")

        self.JSONPath = JSONDir + "/" + JSONName + "-" + self.userName + "-" + self.dateStamp + ".json"

        print "JSON path:", self.JSONPath

        js = json.dumps(self.buildJSON(), indent=4)

        try:

            with open(self.JSONPath, "w") as f:
                f.write(js)
        except Exception as e:
            print "Writing JSON failed: ", e

    def saveHipFile(self):
        """Back up hip file and open .hdas when submitting on the farm"""
        try:
            oldFile = self.hipFile
            hou.hipFile.save()

            hipDir = self.assetDirPath
            hipName = hou.hscriptExpandString("$HIPNAME")
            extension = hou.hscriptExpandString("$HIPFILE").split(".")[-1]

            self.hipFile = hipDir + "/" + hipName + "-" + self.userName + "-" + self.dateStamp + "." + extension

            hou.hipFile.save(file_name = self.hipFile, save_to_recent_files=False)
            print "Hip file saved: {hipfile}".format(hipfile=self.hipFile)
            hou.hipFile.setName(oldFile)

        except hou.OperationFailed as e:
            print "Saving failed", e
