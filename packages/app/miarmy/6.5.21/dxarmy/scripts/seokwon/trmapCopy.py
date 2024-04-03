# encoding:utf-8
# !/usr/bin/env python

import os
import maya.cmds as cmds
import McdGeneral as mg
import McdSimpleCmd as sc

from Qt import QtCore, QtGui, QtWidgets, load_ui
currentpath = os.path.abspath(__file__)
uiFile = os.path.join(os.path.dirname(currentpath), "../ui/trmapCopy.ui")

class Window(QtGui.QMainWindow):

    def __init__(self, parent = None):
        super(Window, self).__init__(parent)
        self.ui = load_ui(uiFile)
        self.connectSignal()

    def connectSignal(self):
        self.ui.pickBtn.clicked.connect(self.pickTrmap)
        self.ui.assignBtn.clicked.connect(self.assignTrmap)

    def pickTrmap(self):
        self.activeAgent = cmds.getAttr("McdGlobal1.activeAgentName")  # 활성화 되어있는 Agent 이름
        stNode = "State_" + str(self.activeAgent)  # State 노드
        self.states = []
        for i in cmds.ls(stNode, dag=True, ap=True):
            if i != stNode:
                self.states.append(str(i).split("_state_")[0])  # self.states : State 노드 이름 목록
        self.acNode = dict()
        for x in self.states:                                   # self.acNode : Action Node 목록
            nm = str(x) + "_state_" + str(self.activeAgent)
            for c in cmds.listConnections(nm, type="McdActionShell"):
                if c.count("Dummy") != 1:
                    if cmds.getAttr(str(c) + ".group") != None:
                        self.acNode[str(c)] = cmds.getAttr(str(c) + ".group")
                    else:
                        self.acNode[str(c)] = ""
                else:
                    pass
        self.stMap = dict()
        for k in self.states:
            self.stMap[k] = dict()
            stName = k + "_state_" + self.activeAgent
            cntNode = cmds.listConnections(stName, c=True, p=True, scn=True)
            for y in range(len(cntNode) / 2):
                self.stMap[k][str(cntNode[2 * y])] = list()
            for j in range(len(cntNode) / 2):
                self.stMap[k][str(cntNode[2 * j])].append(str(cntNode[2 * j + 1]))
        actshNode = "ActionShell_" + str(self.activeAgent)
        dumList = []
        for e in cmds.ls(actshNode, dag=True, ap=True):
            if str(e).count("ActionShell_Dummy") == 1:
                dumList.append(e)
        self.trstNum = len(dumList)
        for w in dumList:
            self.stMap[str(w)] = dict()
            dmNode = cmds.listConnections(w, c=True, p=True, scn=True)
            for f in range(len(dmNode) / 2):
                self.stMap[str(w)][str(dmNode[2 * f])] = str(dmNode[2 * f + 1])

    def mcdCrStateCmd(self, stName):
        newState = stName
        if not mg.isReferenceScene():
            activeAgentName = mg.McdGetActiveAgentName()
            mg.McdGetOrCreateTransitionMapGrp(activeAgentName, 1)
            mg.McdCreateState(newState, activeAgentName)
        else:
            activeAgentName = mg.McdGetActiveAgentName()
            mg.McdCreateStateReference(newState, activeAgentName)

    def checkActions(self, tarAgent):
        for i in self.acNode.keys():
            dk = i.replace(self.activeAgent, tarAgent)
            nwAcNode = dk.replace("_actionShell_", "_action_")
            if len(cmds.ls(nwAcNode)) != 0:
                pass
            else:
                cmds.warning("There is no " + nwAcNode + " node.")

#################################### McdState TransActSetup.py ########################################

    def McdCreateActionShellCmd_OLD(self, acName, prompt=True):
        newAction = acName
        isValid = mg.CheckStringIsValid(newAction)
        if isValid == False:
            cmds.confirmDialog(t="Error", m="New action shell name not valid.")
            raise Exception("New action shell name not valid.")

        if not mg.isReferenceScene():
            activeAgentName = mg.McdGetActiveAgentName()
            mg.McdGetOrCreateActionShellGrp(activeAgentName, 1)
            newAS = self.McdCreateAction_OLD(newAction, activeAgentName, prompt)  # McdStateTransActSetup
        else:
            activeAgentName = mg.McdGetActiveAgentName()
            newAS = self.McdCreateAction_OLDReference(newAction, activeAgentName, prompt)

    def stateTransAddTransitionAction(self, asNode):
        stateNode1 = cmds.text("stateNode1", q=True, l=True)
        stateNode2 = cmds.text("stateNode2", q=True, l=True)
        if stateNode1 == "" or stateNode2 == "":
            cmds.confirmDialog(t="Error", m="No source and target states detected.\nPlease frist establish the transition from source and target states.")
            return
        newAS = self.McdCreateActionShellCmd_OLD()
        cmds.connectAttr(stateNode1 + ".exitAction", newAS + ".input")
        for i in range(100):
            stri = str(i)
            try:
                cmds.connectAttr(newAS + ".output", stateNode2 + ".entryAction[" + stri + "]")
                break
            except:
                pass
        cmds.select(asNode)

    def McdCreateAction_OLD(self, actionName, activeAgentName, prompt=True):
        actionNameLong = actionName + "_actionShell_" + activeAgentName
        temp = cmds.ls(actionNameLong)
        if temp == None or temp == []:
            cmds.createNode("McdActionShell", n=actionNameLong, ss=True)
            try:
                cmds.parent(actionNameLong, "ActionShell_" + activeAgentName)
            except:
                pass
        else:
            if prompt:
                cmds.confirmDialog(t="Note", m="ActionShell exist.")
                raise
        return actionNameLong

    def McdCreateAction_OLDReference(self, actionName, activeAgentName, prompt=True):
        subActiveAgentName = mg.getSubActiveAgentName(activeAgentName)
        actionNameLong = activeAgentName + ":" + actionName + "_actionShell_" + subActiveAgentName
        temp = cmds.ls(actionNameLong)
        if temp == None or temp == []:
            cmds.createNode("McdActionShell", n=actionNameLong, ss=True)
            try:
                cmds.parent(actionNameLong, activeAgentName + ":ActionShell_" + subActiveAgentName)
            except:
                pass
        else:
            if prompt:
                cmds.confirmDialog(t="Note", m="ActionShell exist.")
                raise
        return actionNameLong

    def createActshell(self, agName):
        for i in self.acNode.keys():
            actionName = str(i).split("_actionShell_")[0]
            if len(cmds.ls(actionName + "_acrtionShell_" + str(agName))) == 0:
                self.McdCreateActionShellCmd_OLD(actionName)
            else:
                pass

    def assignTrmap(self):
        targetAgent = cmds.getAttr("McdGlobal1.activeAgentName")
        if not self.states != None or len(self.states) != 0:    # StateNode 생성
            for i in self.states:
                self.mcdCrStateCmd(i)
        self.checkActions(targetAgent)
        for j in range(self.trstNum):                           # Transition Node 생성
            sc.McdCreateActionShellCmd()
        actNode = "ActionShell_" + str(targetAgent)
        duList = []
        for e in cmds.ls(actNode, dag=True, ap=True):           # duList : 생성한 Transition Node 목록
            if str(e).count("ActionShell_Dummy") == 1:
                duList.append(str(e))
        scList = []
        for r in self.stMap.keys():                             # scList : 소스 Transition Node 목록
            if str(r).count("ActionShell_Dummy") == 1:
                scList.append(r)
        dmyList = dict()
        for k in range(self.trstNum):
            for w in self.stMap[scList[k]].keys():
                if w.count("input") == 1 or w.count("entry") == 1:
                    cmds.connectAttr(self.stMap[scList[k]][w].replace(str(self.activeAgent), str(targetAgent)) , duList[k] + "." + w.split(".")[1])
                elif w.count("output") == 1 or w.count("exit") == 1:
                    cmds.connectAttr(duList[k] + "." + w.split(".")[1], self.stMap[scList[k]][w].replace(str(self.activeAgent), str(targetAgent)))
                else:
                    cmds.warning(w + " : Invalid plug.")
            dmyList[str(scList[k])] = str(duList[k])
        for q in range(self.trstNum):                           # 연결된 Transition Node 삭제
            del self.stMap[scList[q]]
        # dmyList {'ActionShell_Dummy1': 'ActionShell_Dummy4', 'ActionShell_Dummy0': 'ActionShell_Dummy5', 'ActionShell_Dummy2': 'ActionShell_Dummy3'}
        # duList ['ActionShell_Dummy3', 'ActionShell_Dummy4', 'ActionShell_Dummy5']
        # scList ['ActionShell_Dummy2', 'ActionShell_Dummy1', 'ActionShell_Dummy0']
        self.createActshell(targetAgent)
        # i j e r k w q m n
        for m in range(len(self.stMap.keys())):                 # m : 소스 state 노드 갯수
            for n in self.stMap[self.states[m]].keys():         # n : 각 state 노드의 키 값
                for h in range(len(self.stMap[self.states[m]][n])):
                    if n.count("input") == 1 or n.count("entry") == 1:
                        if self.stMap[self.states[m]][n][h].count("Dummy") != 1:
                            if cmds.isConnected((self.stMap[self.states[m]][n][h]).replace(str(self.activeAgent), str(targetAgent)), n.replace(str(self.activeAgent), str(targetAgent))) != 1:
                                cmds.connectAttr((self.stMap[self.states[m]][n][h]).replace(str(self.activeAgent), str(targetAgent)), n.replace(str(self.activeAgent), str(targetAgent)))
                            else:
                                pass
                        else:
                            if cmds.isConnected(dmyList[(self.stMap[self.states[m]][n][h]).split(".")[0]] + "." + (self.stMap[self.states[m]][n][h]).split(".")[1], n.replace(str(self.activeAgent), str(targetAgent))) != 1:
                                cmds.connectAttr(dmyList[(self.stMap[self.states[m]][n][h]).split(".")[0]] + "." + (self.stMap[self.states[m]][n][h]).split(".")[1], n.replace(str(self.activeAgent), str(targetAgent)))
                            else:
                                pass
                    elif n.count("output") == 1 or n.count("exit") == 1:
                        if self.stMap[self.states[m]][n][h].count("Dummy") != 1:
                            if cmds.isConnected(n.replace(str(self.activeAgent), str(targetAgent)), (self.stMap[self.states[m]][n][h]).replace(str(self.activeAgent), str(targetAgent))) != 1:
                                cmds.connectAttr(n.replace(str(self.activeAgent), str(targetAgent)), (self.stMap[self.states[m]][n][h]).replace(str(self.activeAgent), str(targetAgent)))
                            else:
                                pass
                        else:
                            if cmds.isConnected(n.replace(str(self.activeAgent), str(targetAgent)), dmyList[(self.stMap[self.states[m]][n][h]).split(".")[0]] + "." + (self.stMap[self.states[m]][n][h]).split(".")[1]) != 1:
                                cmds.connectAttr(n.replace(str(self.activeAgent), str(targetAgent)), dmyList[(self.stMap[self.states[m]][n][h]).split(".")[0]] + "." + (self.stMap[self.states[m]][n][h]).split(".")[1])
                            else:
                                pass
                    elif n.count("message") == 1:
                        pass
                    else:
                        cmds.warning(n + " : Invalid plug.")
        for df in self.acNode.keys():
            nf = str(df).replace(str(self.activeAgent), str(targetAgent))
            cmds.setAttr(nf + ".group", str(self.acNode[df]), type="string")

def main():
    global myWindow
    myWindow = Window()
    myWindow.ui.show()

if __name__ == '__main__':
    main()
