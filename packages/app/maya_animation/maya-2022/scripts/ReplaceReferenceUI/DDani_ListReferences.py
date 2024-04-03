__author__ = 'gyeongheon.jeong'

import maya.cmds as cmds
import os

def ListReferences():
    refNode = cmds.ls(references=1)
    refFileName = []
    refFilePath = []

    for i in refNode:
        try:
            fileName_PathList = cmds.referenceQuery(i, filename = True).split(os.sep)
            refNodeName = cmds.referenceQuery(i, referenceNode = True)

            refFileName.append( refNodeName + " " + fileName_PathList[-1] )
            refFilePath.append( os.sep.join( fileName_PathList[:-1] ) )
        except:
            pass

    return refFileName, refFilePath

def listReferenceDirs(filePath):
    fileList = os.listdir( filePath )
    folderList = []

    for F in fileList:
        if os.path.isdir(filePath + os.sep + F):
            folderList.append(F)

    return folderList

def listReferenceDirFiles(filePath):
    fileList = os.listdir( filePath )
    MbFileList = []

    for F in fileList:
        if F.endswith("mb"):
            MbFileList.append(F)

    MbFileList.sort(reverse=True)
    return MbFileList

def ReplaceReferences(oldfilePath, newFilePath):
    referencenode = cmds.file(oldfilePath, q = True, rfn = True)

    NewReferencenode = cmds.file(newFilePath, loadReference = referencenode, options = "v=0")
