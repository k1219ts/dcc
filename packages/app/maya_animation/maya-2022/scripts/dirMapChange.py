# -*- coding: utf-8 -*-
import maya.cmds as cmds
import os

def dirMapChange():
    confirm = cmds.confirmDialog(b=("OK","Cancel"), db="OK", ds="Cancel", m="show ������ ������θ� ������ �ּ���")
    
    if confirm == "OK":
        oldDir = "show"
        oldDexterDir = "dexter/show"
        newDir = None 
        newDir = cmds.fileDialog2( okc="Select", fm=3) # D:/test/project/anything/
        
        if newDir:
            newDir = os.sep.join( [newDir[0], oldDir] )
            AbsDir = os.path.abspath(newDir)
            cmds.dirmap(en=True)
            cmds.dirmap(m=("/{0}/".format(oldDir), 
                            AbsDir))
            print cmds.dirmap(cd="/{0}/".format(oldDir))
            cmds.dirmap(m=("/{0}/".format(oldDexterDir), 
                            AbsDir))
            print cmds.dirmap(cd="/{0}/".format(oldDexterDir))

def referenceRemap():
    confirm = cmds.confirmDialog(b=("Yes","Cancel"), db="OK", ds="Cancel", m="���۷��� ��θ� �ϰ������� �����մϴ�.")
    
    if confirm == "Yes":
        listRef = cmds.file(q=1, reference=1)
        
        for i in listRef:
            refNodeName = cmds.referenceQuery(i, rfn=1)
            refFileName = cmds.referenceQuery(refNodeName, f=True)
            print refFileName
            cmds.file(refFileName, loadReference=refNodeName, options="v=0")
