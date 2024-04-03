import os
import re
import hou
import datetime
import subprocess
import shutil
import json
import getpass

class FileIO(object):
    def __init__(self, node):
        self.p = dict()  #storing interface parameters
        self.node = node
        self.cachePath = ""
        self.hipFile = hou.hipFile.path()
        self.JSONPath = ""
        self.userName = getpass.getuser()

        self.getHoudiniParms()
        self.checkParameters()
        self.buildCachePath()

    def getHoudiniParms(self):
        self.FX_CACHE = self.node.parm("FX_CACHE").eval().lower()
        self.SHOW = self.node.parm("SHOW").evalAsString().lower()
        self.SEQ = self.node.parm("SEQ").evalAsString().upper()
        self.SHOT = self.node.parm("SHOT").evalAsString().upper()
        self.FX_GROUP = self.node.parm("FX_GROUP").eval().lower()
        self.INPUT_TAKE = self.node.parm("INPUT_TAKE").eval().lower()
        self.DATA_TYPE = self.node.parm("DATA_TYPE").evalAsString()
        self.ELEMENT_NAME = self.node.parm("ELEMENT_NAME").eval().lower()
        self.ELEMENT_VERSION = self.node.parm("ELEMENT_VERSION").eval().lower()
        self.ELEMENT_VERSION_POSTFIX = self.node.parm("ELEMENT_VERSION_POSTFIX").eval().lower()

        if self.ELEMENT_VERSION_POSTFIX:
            self.ELEMENT_VERSION_POSTFIX = "_" + self.ELEMENT_VERSION_POSTFIX

        self.IS_SEQUENCE = self.node.parm("trange").eval()
        self.STARTFRAME = self.node.parm("f1").eval()
        self.ENDFRAME = self.node.parm("f2").eval()
        self.INCREMENT = self.node.parm("f3").eval()
        self.SUBSTEPS = self.node.parm("substeps").eval()
        self.SAVE_HIPFILE = self.node.parm("saveHipFile").eval()
        self.FILE_TYPE = self.node.parm("FILE_TYPE").evalAsString().lower()

        self.ROPNETWORK = self.node.parm("ROPNetwork").eval()
        self.FETCHROP = self.node.parm("fetchROP").eval()

        self.USER_COMMENT = self.node.parm("USER_COMMENT").eval()

        self.ISPUBLISHED = self.node.parm("isPublished").eval()

    def preRender(self):
        if self.ISPUBLISHED:
            return
        self.createDirectories()

    def render(self):
        renderType = self.FILE_TYPE.split(".")[0]
        self.node.node("render_{fileType}".format(fileType=renderType)).parm("execute").pressButton()

    def renderbg(self):
        renderType = self.FILE_TYPE.split(".")[0]
        if renderType == "usd":
            self.node.node("render_{fileType}".format(fileType=renderType)).parm("execute").pressButton()
        else:
            self.node.node("render_{fileType}".format(fileType=renderType)).parm("executebackground").pressButton()

    def calcPlaybackPercentage(self):
        frame = hou.frame()
        percentage = (frame - self.STARTFRAME)/(self.ENDFRAME - self.STARTFRAME)*100
        percentage = str(int(percentage))+"%"
        return percentage

    def descriptiveParameter(self):
        d = "SHOW:  " + self.SHOW + " | " + "SEQ:  " + self.SEQ + " | " + "SHOT:  " + self.SHOT + "\n"
        d += "-"*50 + "\n"
        d += "FX_GROUP:   " + self.FX_GROUP + " | " + "INPUT_TAKE:   " + self.INPUT_TAKE + "\n"
        d += "-"*50 + "\n"
        d += "DATA_TYPE: " + self.DATA_TYPE + "\n"
        d += "ELEMENT_NAME: " + self.ELEMENT_NAME + "\n"
        d += "ELEMENT_VERSION: " + self.ELEMENT_VERSION + "\n"
        d += "ELEMENT_POSTFIX: " + self.ELEMENT_VERSION_POSTFIX + "\n"
        d += "-"*50 + "\n"
        d += "RANGE: " + str(self.STARTFRAME) + " - " + str(self.ENDFRAME) + "\n"
        d += "-"*50 + "\n"
        d += self.USER_COMMENT + "\n"

        return d

    def createDirectories(self):
        if not os.path.exists(self.buildCacheDirPath()):
            os.makedirs(self.buildCacheDirPath())
            print 'Cache Path', self.cachePath

    def checkParameters(self):
        if re.match("v[0-9][0-9][0-9]", self.INPUT_TAKE) is None or len(self.INPUT_TAKE) != 4:
            print "Incorrect input take"

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

    def buildCachePath(self):
        # frameCounter
        if self.SUBSTEPS == 1:
            frameNumber = str(int(hou.frame())).zfill(4)
        else:
            frameNumberInt = str(int(hou.frame())).zfill(4)
            frameNumberFrac = str(hou.frame() - int(hou.frame()))[2:].zfill(4)
            frameNumber = frameNumberInt + "." + frameNumberFrac
        self.cachePath = self.buildCacheDirPath()

        # cachePath
        if self.IS_SEQUENCE:
            if "abc" in self.FILE_TYPE and self.node.parm("render_full_range").eval():
                self.cachePath += "/cache"
            elif "usd" in self.FILE_TYPE and self.node.parm("fileperframe").eval() == 0:
                self.cachePath += "/cache"
            else:
                self.cachePath += "/cache." + frameNumber
        else:
            self.cachePath += "/cache"

        self.cachePath += "." + self.FILE_TYPE

    def publish(self):
        cachePath = self.buildCacheDirPath()
        publishPath = cachePath.replace(self.SHOT + "/dev/", self.SHOT + "/pub/")

        if not os.path.exists(publishPath):
            if subprocess.check_output(['mkdir', '-p', publishPath]):
                print "Publish directory creation failed"
            os.system("mv " + cachePath + "/* " + publishPath +"/")
        else:
            print "Already published"

        self.node.parm("isPublished").set(1)
        fetch = hou.node(self.FETCHROP)
        if fetch:
            fetch.bypass(1)

    def buildAssetDirPath(self):
        cachePath = self.FX_CACHE
        cachePath += "/" + self.SHOW
        cachePath += "/" + self.SEQ
        cachePath += "/" + self.SHOT

        if self.ISPUBLISHED:
            cachePath += "/pub"
        else:
            cachePath += "/dev"

        cachePath += "/" + self.FX_GROUP
        return cachePath

    def buildCacheDirPath(self):
        cachePath = self.buildAssetDirPath()
        cachePath += "/" + self.INPUT_TAKE
        cachePath += "/" + self.DATA_TYPE
        cachePath += "_" + self.ELEMENT_NAME
        cachePath += "/" + self.ELEMENT_VERSION
        cachePath += self.ELEMENT_VERSION_POSTFIX
        return cachePath

    def buildJSON(self):
        JSONDic = {
            "userName": self.userName,
            "userComment": self.USER_COMMENT
        }
        return JSONDic

    def saveHipFile(self):
        """Back up hip file and open .hdas when submitting on the farm"""
        try:
            if not self.ISPUBLISHED and self.SAVE_HIPFILE:
                self.createDirectories()


                hipDir = self.buildCacheDirPath()
                if not os.path.exists(hipDir):
                    os.makedirs(hipDir)

                shutil.copy2(self.hipFile, hipDir)
                print '# Debug : FileIO saved HipFile : {NODE} -> {FILE}'.format(NODE=self.node, FILE=self.hipFile)

                # json log
                jsonfile = os.path.basename(self.hipFile).replace('.hip', '.json')
                jsonpath = os.path.join(hipDir, jsonfile)
                js = json.dumps(self.buildJSON(), indent=4)
                try:
                    with open(jsonpath, 'w') as f:
                        f.write(js)
                except Exception as e:
                    print '# Error : Writing JSON failed: ', e
        except hou.OperationFailed as e:
            print "Saving failed", e

    def createROPNode(self):
        RN = hou.node(self.ROPNETWORK)
        if not RN:
            hou.ui.displayMessage("ROP network doesn't exist")
        fetchName = self.node.name()
        fetchNode = RN.createNode("fetch", fetchName)
        fetchNode.moveToGoodPosition()

        fileIONode = fetchNode.relativePathTo(self.node)

        fetchNode.setExpressionLanguage(hou.exprLanguage.Python)
        fetchNode.addSpareParmTuple(hou.StringParmTemplate("pathToFileIO", "Path To File I/O Node", 1,
                                                           string_type=hou.stringParmType.NodeReference))
        fetchNode.parm("pathToFileIO").set(fileIONode)

        expression = hou.StringKeyframe()
        e = 'hou.pwd().parm("pathToFileIO").evalAsString()+"/render_"' \
            ' + hou.node(hou.pwd().parm("pathToFileIO").evalAsString()).parm("FILE_TYPE").evalAsString().lower().split(".")[0]'
        expression.setExpression(e)
        fetchNode.parm("source").setKeyframe(expression)
        fetchNode.parm("source").lock(1)
        #fetchNode.parm("source").hide(1)

        self.node.parm("fetchROP").set(fetchNode.path())

        #old
        '''
        fetch.parm("source").set(sourcePath+"/render")
        self.node.parm("fetchROP").set(fetch.path())
        '''

    @staticmethod
    def nameChange(node):
        cacheNodePath = node.path()
        fetchName = node.name()
        fetchPath = node.parm("fetchROP").eval()
        fetch = hou.node(fetchPath)
        if fetch:
            fetch.setName(fetchName)

    def openCacheInNautilus(self):
        print self.buildCacheDirPath()
        # os.system("nautilus " + self.buildCacheDirPath())
        subprocess.Popen(["/usr/bin/nautilus", self.buildCacheDirPath()])

    def openUsdViewer(self):
        # in bashshell
        # /backstage/bin/DCC rez-env baselib-2.0 usdtoolkit-19.11 -- usdviewer /home/youngshin.lee/Documents/cache.1001.usd
        print self.cachePath
        os.system("usdview " + self.cachePath)

    def printDiskUsage(self):
        cachePath = self.buildCacheDirPath()
        os.system("du -h -c --max-depth=2 " + cachePath + "|sort -h")

