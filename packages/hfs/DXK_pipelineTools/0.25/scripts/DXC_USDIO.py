# -*- coding: utf-8 -*-

import os, re, hou,datetime,subprocess, shutil, json,getpass,signal,math
from datetime import date

"""

Brief Changelog
- 23.11.13 : Cloned from /stdrepo/PFX/FXteamPath/otls
             + functions defined in asset moved to this file, path build function update (for sequence output and etc)

As Reminder, Folder conventions are the following 2023.11
/show/_3d
   ->  /show/{PROJNAME}/_3d/{SEQ}/{SHOT}/fx/ELENAME/ELEVERSION/ELENAME_ELEVERSION.usd
 /fx_cache/
   ->  /fx_cache/{PROJNAME}/{SEQ}/{SHOT}/fx/ELENAME/ELEVERSION/ELENAME_ELEVERSION.usd

"""

class USDIO(object):
    def __init__(self, node):
        self.p = dict()  #storing interface parameters
        self.node = node
        self.cachePath = ""
        self.hipFile = hou.hipFile.path()
        self.JSONPath = ""
        self.userName = getpass.getuser()

        self.getHoudiniParms()
        #self.checkParameters()
        self.buildCachePath()

    def getHoudiniParms(self):
        self.FX_CACHE = "/fx_cache"
        self.SHOW = self.node.parm("SHOW").evalAsString().lower()
        self.SEQ = self.node.parm("SEQ").evalAsString().upper()
        self.SHOT = self.node.parm("SHOT").evalAsString().upper() 
        self.ELEMENT_NAME = self.node.parm("element_name").eval().lower()
        self.ELEMENT_VERSION = self.node.parm("element_version").eval().lower()
        self.STORAGE_STATUS = self.node.parm("storage").eval()
        self.IS_SEQUENCE = self.node.parm("trange").eval()
        self.STARTFRAME = self.node.parm("f1").eval()
        self.ENDFRAME = self.node.parm("f2").eval()
        self.INCREMENT = self.node.parm("f3").eval()
        self.STORETYP = self.node.parm("frameseq_check").eval()
        #self.SAVE_HIPFILE = self.node.parm("saveHipFile").eval()
        #self.FILE_TYPE = self.node.parm("FILE_TYPE").evalAsString().lower()

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

    def createDirectories(self):
        if not os.path.exists(self.buildCacheDirPath()):
            os.makedirs(self.buildCacheDirPath())
            print ('Cache Path', self.cachePath)

    def checkParameters(self):
        if re.match("v[0-9][0-9][0-9]", self.INPUT_TAKE) is None or len(self.INPUT_TAKE) != 4:
            pass
            #print "Incorrect input take"

        if re.match("v[0-9][0-9][0-9]", self.ELEMENT_VERSION) is None or len(self.ELEMENT_VERSION) != 4:
            pass
            #print "Incorrect version"

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
            seq_path = '/show/' + self.node.parm('SHOW').evalAsString() + '/works/PFX/shot'
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
            shot_path = '/show/' + self.node.parm('SHOW').evalAsString() + '/works/PFX/shot/' + self.node.parm('SEQ').evalAsString()
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

        frameNumber = str(int(hou.frame())).zfill(4)
        self.cachePath = self.buildCacheDirPath()

        #nself.cachePath += "." + self.FILE_TYPE

    def publish(self):
        cachePath = self.buildCacheDirPath()
        #publishPath = cachePath.replace(self.SHOT + "/dev/", self.SHOT + "/pub/")

        if not os.path.exists(publishPath):
            if subprocess.check_output(['mkdir', '-p', publishPath]):
                print ("Publish directory creation failed")
            os.system("mv " + cachePath + "/* " + publishPath +"/")
        else:
            print ("Already published")

        self.node.parm("isPublished").set(1)
        fetch = hou.node(self.FETCHROP)
        if fetch:
            fetch.bypass(1)

    def buildAssetDirPath(self):
        if self.STORAGE_STATUS == 0 :
            cachePath = "/show"
            cachePath += "/" + self.SHOW
            cachePath += "/_3d/shot"
            cachePath += "/" + self.SEQ
            cachePath += "/" + self.SHOT
            cachePath += "/fx"
        elif self.STORAGE_STATUS == 1 :
            cachePath = self.FX_CACHE
            cachePath += "/" + self.SHOW
            cachePath += "/" + self.SEQ
            cachePath += "/" + self.SHOT
            cachePath += "/fx"

        return cachePath

    def buildCacheDirPath(self):
        frameNumber = str(int(hou.frame())).zfill(4)
        cachePath = self.buildAssetDirPath()

        cachePath += "/" + self.ELEMENT_NAME
        cachePath += "/" + self.ELEMENT_VERSION
        if self.STORETYP == 0 :
            cachePath += "/" + self.ELEMENT_NAME + "_" + self.ELEMENT_VERSION + ".usd"
        elif self.STORETYP == 1 :
            cachePath += "/" + self.ELEMENT_NAME + "_" + self.ELEMENT_VERSION + "." + frameNumber + ".usd"
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
                print ('# Debug : FileIO saved HipFile : {NODE} -> {FILE}'.format(NODE=self.node, FILE=self.hipFile))
                
                # json log
                jsonfile = os.path.basename(self.hipFile).replace('.hip', '.json')
                jsonpath = os.path.join(hipDir, jsonfile)
                js = json.dumps(self.buildJSON(), indent=4)
                try:
                    with open(jsonpath, 'w') as f:
                        f.write(js)
                except Exception as e:
                    print ('# Error : Writing JSON failed: ', e)
        except hou.OperationFailed as e:
            print ("Saving failed", e)

    def createROPNode(self):
        RN = hou.node(self.ROPNETWORK)
        if not RN:
            hou.ui.displayMessage("ROP network doesn't exist")
        fetchName = self.node.name()
        fetchNode = RN.createNode("fetch", fetchName)
        # v1.1 recommended for 19.5 and above. Author : yongjun.cho
        config = RN.createNode("fx::DXC_config::1.1", str(fetchName+"_config"))
        submitter = RN.createNode("fx::DXC_submitter::1.1", str(fetchName+"_config"))
        fetchNode.moveToGoodPosition()
        config.moveToGoodPosition()
        submitter.moveToGoodPosition()
        mode = self.node.parm("makeFetch").eval()
        fileIONode = fetchNode.relativePathTo(self.node)
        fileIONode = config.relativePathTo(self.node)
        fileIONode = submitter.relativePathTo(self.node)
        config.setInput(0, hou.node(fetchNode.path()))
        submitter.setInput(0, hou.node(fetchNode.path()))
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
        self.sub = submitter
        
        if 'sim' in fetchNode.path() :
            print("This is sim")
            self.setSim(self.sub, mode)
        elif 'data' in fetchNode.path() :
            print("This is cache")
            self.setCache(self.sub, mode)
        elif 'render' in fetchNode.path() :
            print("This is render")
            self.setRender(self.sub, mode)

    def setSim(self, submitter, mode):
        try:
            node = self.sub
            input = node.inputs()[0]
            inputs = node.inputs()
            count = len(inputs)
            output = input.outputs()
            outputlist = list(output)
            outputlist.remove(node)
            config = hou.node(outputlist[0].path())
                                
            setsim = config.parm('isSimulation')
            setsim.set(1)
            setChunkSize = config.parm('chunkSize')
            setChunkSize.set(1)

            config.setColor(hou.Color(0.4,0.8,1))
            config.setName('sim', 1)
            
            title = node.parm('title')
            maxact = node.parm('maxActive')
            priority = node.parm('priority')
            node.parm('preset').set(1)
            title.set("`$HIPNAME`_sim_" + inputs[count-1].name())
            maxact.set(count)
            priority.set(200)

            if mode == 0 : 
                for n in output :
                    n.destroy()
            #node.setName("Simulation_"+ inputs[count-1].name().capitalize())
        except:
            pass

    def setRender(self, submitter, mode):        
        try:
            node = self.sub
            input = node.inputs()[0]
            inputs = node.inputs()
            count = len(inputs)
            output = input.outputs()
            outputlist = list(output)
            outputlist.remove(node)
            config = hou.node(outputlist[0].path())
            
            setsim = config.parm('isSimulation')
            setsim.set(0)
            setChunkSize = config.parm('chunkSize')
            setChunkSize.set(1)

            config.setColor(hou.Color(1,0.5,0.9))
            config.setName('render', 1)
            
            title = node.parm('title')
            maxact = node.parm('maxActive')
            priority = node.parm('priority')
            node.parm('preset').set(2)
            title.set("`$HIPNAME`_render_"+inputs[count-1].name())
            maxact.set(30)
            priority.set(100)
            
            if mode == 0 : 
                for n in output :
                    n.destroy()
            #node.setName("Render_"+ inputs[count-1].name().capitalize())
        except:
            pass
            
    def setCache(self, submitter,mode):
        try:
            node = self.sub
            input = node.inputs()[0]
            inputs = node.inputs()
            count = len(inputs)
            output = input.outputs()
            outputlist = list(output)
            outputlist.remove(node)
            config = hou.node(outputlist[0].path())
            
            setsim = config.parm('isSimulation')
            setsim.set(0)
            setChunkSize = config.parm('chunkSize')
            setChunkSize.set(1)
            
            config.setColor(hou.Color(1,0.5,0.3))
            config.setName('cache', 1)

            title = node.parm('title')
            maxact = node.parm('maxActive')
            priority = node.parm('priority')
            node.parm('preset').set(0)
            title.set("`$HIPNAME`_cache_"+inputs[count-1].name())
            maxact.set(30)
            priority.set(110)
            #node.setName("Cache_"+ inputs[count-1].name().capitalize())
            if mode == 0 : 
                for n in output :
                    n.destroy()
        except:
            print("Printing Self")
            print(self.sub)
            pass

    @staticmethod
    def nameChange(node):
        cacheNodePath = node.path()
        fetchName = node.name()
        fetchPath = node.parm("fetchROP").eval()
        fetch = hou.node(fetchPath)
        if fetch:
            fetch.setName(fetchName)

    def openCacheInNautilus(self):
        path = os.path.dirname(self.buildCacheDirPath())
        subprocess.Popen(["xdg-open",path])
        
    def getSizeofCache(self):
        #Function Author : yongjun.cho
        cachePath = self.buildCacheDirPath()
        size_name = ("B","KB","MB","GB")
        total_size = 0
        
        for dirpath, dirnames, filenames in os.walk(cachePath):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if not os.path.islink(fp):
                    total_size += os.path.getsize(fp)

        if total_size!=0:
            i = int(math.floor(math.log(total_size, 1024)))
            p = math.pow(1024,i)
            s = round(total_size / p, 2)
            value = "%s%s" % (s,size_name[i])
        else:
            value = "DOES NOT EXIST. Check your cache"

        self.node.parm("Disk_Usage").set(str(value))

    def bladeMode(self):
        #Function Author : yongjun.cho
        path = self.node.parm("fetchROP").eval()
        bladeMode = self.node.parm("selection").eval()
        childnode=hou.node(path)
        if path == "" :
            hou.ui.displayMessage("Path is empty", severity=hou.severityType.Error)
        else :
            type = childnode.outputs()
            if type[0].type().name() == "DXC_config" :
                submitter = childnode.outputs()[1]
                if bladeMode == 0 :
                    submitter.parm("selection").set(0)
                    print("blade status : ALL")
                elif bladeMode == 1 :
                    submitter.parm("selection").set(1)
                    print("blade status : Robust")
            elif type[1].type().name() == "DXC_config" :
                submitter = childnode.outputs()[0]
                if bladeMode == 0 :
                    submitter.parm("selection").set(0)
                    print("blade status : ALL")
                elif bladeMode == 1 :
                    submitter.parm("selection").set(1)
                    print("blade status : Robust")

    def quickSubmit(self):
        #Function Author : yongjun.cho
        path = self.node.parm("fetchROP").eval()
        childnode=hou.node(path)
        print(childnode.outputs())
        
        if path == "" :
            hou.ui.displayMessage("Path is empty", severity=hou.severityType.Error)
        else :
            type = childnode.outputs()
            if type[0].type().name() == "DXC_config" :
                submitter = childnode.outputs()[1]
                submitter.parm("Submit").pressButton()
            elif type[1].type().name() == "DXC_config" :
                submitter = childnode.outputs()[0]
                submitter.parm("Submit").pressButton()

    def setROPtype(self):
        val = self.node.parm("ropType").eval()
        if val == 0 :
            self.node.parm("ROPNetwork").set("/obj/WORK")
        elif val == 1 :
            self.node.parm("ROPNetwork").set("/obj/WORK/data")
        elif val == 2 :
            self.node.parm("ROPNetwork").set("/obj/WORK/sim")
        elif val == 3 :
            self.node.parm("ROPNetwork").set("/obj/WORK/render")

    def frameSwitch(self):
        #self_ = hou.pwd()
        fSwitchparm = self.node.parm("Frame_Switch").eval()
        if(fSwitchparm == 0):
            self.node.parm("trange").setExpression('ch("/obj/DX_Scene_Setting1/trange")')
            self.node.parm("f1").setExpression('ch("/obj/DX_Scene_Setting1/f1")')
            self.node.parm("f2").setExpression('ch("/obj/DX_Scene_Setting1/f2")')
            self.node.parm("f3").setExpression('ch("/obj/DX_Scene_Setting1/f3")')
        else:
            self.node.parm("trange").deleteAllKeyframes()
            self.node.parm("f1").deleteAllKeyframes()
            self.node.parm("f2").deleteAllKeyframes()
            self.node.parm("f3").deleteAllKeyframes()