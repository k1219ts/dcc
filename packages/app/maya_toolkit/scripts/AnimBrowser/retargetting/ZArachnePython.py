import os
import Zelos
import json
import pprint


class ConvertBVHtoJSON(object):
    
    def __init__(self):
        self.s = Zelos.skeleton()
        self.linearData = []
        self.data = {}
        
    def doIt(self):
        infile = "/dexter/Cache_DATA/RND/jeongmin/AnimBrowser_retarget/retarget_data/proxy_retarget.bvh"
        outfile = "/dexter/Cache_DATA/RND/jeongmin/AnimBrowser_retarget/retarget_data/output.json"
        self.s.load(infile)
        self.convertFile(infile, outfile)    
        
    def insertDict(self, parent, child):
        if not child.keys()[0] in parent[parent.keys()[0]]:
            parent[parent.keys()[0]][child.keys()[0]] = {}
        else:
            parent[parent.keys()[0]][child.keys()[0]] = child[child.keys()[0]]

        return parent


    def readRecur(self, joint):
        parentn = self.s.jointName(joint)
        parentDic = {str(parentn): {}}
        self.linearData.append(joint)
        for i in range(self.s.childNum(joint)):
            child = self.s.childAt(joint, i)
            childn = self.s.jointName(child)
            childDic = {str(childn): {}}

            if joint == self.s.getParent(child):
                resultDict = self.insertDict(parentDic, childDic)
                self.data.update(resultDict)

            self.readRecur(child)


    def insertRecur(self, data):
        for i in data:
            for k in data:
                if i in data[k]:
                    data[k][i] = data[i]


    def translation(self, joint, frame):
        offset = self.s.getOffsets(joint)
        return [float(self.s.translation(joint, frame)[0]),
                float(self.s.translation(joint, frame)[1]),
                float(self.s.translation(joint, frame)[2])]


    def rotation(self, joint, frame):
        return [float(self.s.orientation(joint, frame)[0]),
                float(self.s.orientation(joint, frame)[1]),
                float(self.s.orientation(joint, frame)[2])]




    def saveJSON(self, infile="", outfile=""):
        root = self.s.getRoot()
        rootname = str(self.s.jointName(root))

        self.readRecur(root)
        self.insertRecur(self.data)
        wdata = {}

        wdata["joint_size"] = self.s.numJoints()
        wdata["frame_size"] = self.s.numFrames()
        wdata["input_file"] = infile
        wdata["HIERARCHY"] = {rootname: self.data[rootname]}
        keyData = {}
        for i in self.linearData:
            key = str(self.s.jointName(i))
            keyData[key] = {}
            for f in range(self.s.numFrames()):
                keyData[key][f] = {}
                channels = self.s.getChannels(i)
                rot = trans = False
                for chname in channels:
                    if "rotate" in str(chname) and rot == False:
                        rot = True
                        keyData[key][f]["rotate"] = self.rotation(i, f)

                    if "translate" in str(chname) and trans == False:
                        trans = True
                        keyData[key][f]["translate"] = self.translation(i, f)

        wdata["MOTIONDATA"] = keyData

        with open(outfile, "w") as outfile:
            json.dump(wdata, outfile, sort_keys=True, indent=4)


    def convertFile(self, infile, outfile):
        if os.path.splitext(outfile)[-1] == '.json':
            self.saveJSON(infile, outfile)
        else:
            self.s.save(outfile)

 
