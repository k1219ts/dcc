#!/usr/bin/python

'''
TODO:
- import abc file
- reduce
- output abc file
'''

import argparse
import hou
import os

renameDict = {"70" : "LOD1", 
                "50" : "LOD2", 
                "15" : "LOD3", 
                "5" : "LOD4", 
                "2" : "LOD5"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-input", type = str, help = "input abc file")
    parser.add_argument("-objpath", type = str, default = "", help = "object path of component abc")
    parser.add_argument("-reduce", type = int, help = "reduce level")

    args = parser.parse_args()

    rootNode = hou.node("obj")
    geoNode = rootNode.createNode("geo")
    # geoNode.node("file1").destroy()
    abcNode = geoNode.createNode("alembic")
    if not args.objpath == "":
        abcNode.parm("objectPath").set(args.objpath)
    abcNode.parm('fileName').set(args.input)
    # abcNode.parm("groupnames").set(3)

    unpackNode = geoNode.createNode("unpack")
    unpackNode.setFirstInput(abcNode)
    unpackNode.parm("transfer_attributes").set("path")
    
    convertNode = geoNode.createNode("convert")
    convertNode.setFirstInput(unpackNode)

    reduceNode = geoNode.createNode("polyreduce::2.0") # check
    reduceNode.setFirstInput(convertNode)
    # reduceNode.parm("preservequads").set(True)
    reduceNode.parm("percentage").set(args.reduce)

    attributePromptNode = geoNode.createNode("attribpromote") # check
    attributePromptNode.setFirstInput(reduceNode)
    attributePromptNode.parm("inname").set("rnml")
    attributePromptNode.parm("outclass").set(3)
    attributePromptNode.parm("useoutname").set(True)
    attributePromptNode.parm("outname").set("__Nref")

    attributeNode = geoNode.createNode("attribute") # check
    attributeNode.setFirstInput(attributePromptNode)
    attributeNode.parm("frompt0").set("rest")
    attributeNode.parm("topt0").set("__Pref")

    ropAbcNode = geoNode.createNode("rop_alembic")
    ropAbcNode.setFirstInput(attributeNode)

    filePath, extension = os.path.splitext(args.input)
    curDir = os.path.dirname(filePath)
    fileName = os.path.basename(filePath)
    splitName = fileName.split("_")

    outputPath = os.path.join(curDir, "%s_%s_%s_%s%s" % ("_".join(splitName[:-2]), renameDict[str(args.reduce)], splitName[-2], splitName[-1], extension))
    ropAbcNode.parm("filename").set(outputPath)
    ropAbcNode.parm("build_from_path").set(True)
    ropAbcNode.render()

# cmd : hython $PATH/reduceModel.py -input *.abc -reduce 50
