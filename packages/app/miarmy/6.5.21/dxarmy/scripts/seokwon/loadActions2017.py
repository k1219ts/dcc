# encoding:utf-8
# !/usr/bin/env python

import os
import McdAgentManager
import McdLoadActions
import maya.cmds as cmds
import McdGeneral as mg
import McdSimpleCmd as sc
import dxUI
from Qt import QtCore, QtGui, QtWidgets, load_ui
currentpath = os.path.abspath(__file__)
uiFile = os.path.join(os.path.dirname(currentpath), "../ui/loadActions.ui")

def hconv(text):
    return unicode(text, 'utf-8')

class Window(QtWidgets.QMainWindow):
    def __init__(self, parent = None):
        super(Window, self).__init__(parent)
        dxUI.setup_ui(uiFile, self)
        self.connectSignal()
        self.agDict = {}
        self.updateAgentList()
        self.actPath = []
        self.actName = []

    def connectSignal(self):
        self.impBtn.clicked.connect(self.impAct)
        self.delBtn.clicked.connect(self.delAct)
        self.dropBtn.clicked.connect(self.dropAct)
        self.refBtn.clicked.connect(self.updateAgentList)
        self.taCombo.currentIndexChanged.connect(self.tarAgChange)
        self.delAgBtn.clicked.connect(self.delAg)
        self.loadBtn.clicked.connect(self.loadAgent)
        self.pickBtn.clicked.connect(self.pickProc)
        self.assignBtn.clicked.connect(self.assignProc)
        self.raBtn.clicked.connect(self.connectMultiAction)
        self.refrBtn.clicked.connect(self.refr_stateList)
        self.refr_stateList()

    def updateAgentList(self):
        self.agAct.clear()
        agList = McdAgentManager.McdGetAllAgentTypeNIDWithColor()[0]
        if not len(agList) == 0:
            self.taCombo.clear()
            self.taCombo.addItems(agList)
        self.agDict = {}
        for e in agList:
            self.agDict[e] = []

    def impAct(self):
        getActLis = []
        getAct = cmds.fileDialog2(startingDirectory="/dexter/Cache_DATA/CRD", fileMode=4, fileFilter="Maya Ascii (*.ma)", caption="Import Action")
        for i in getAct:
            self.actPath.append(str(i))
            self.actName.append(os.path.basename(str(i)).split(".")[0])
            getActLis.append(os.path.basename(str(i)).split(".")[0])
        self.allAct.addItems(getActLis)
        self.reData(self.actPath, self.actName)

    def delAct(self):
        selItem = self.allAct.selectedItems()
        if not selItem:
            return
        for j in selItem:
            self.allAct.takeItem(self.allAct.row(j))
            del self.actPath[self.actName.index(str(j.text()))]
            del self.actName[self.actName.index(str(j.text()))]
        self.reData(self.actPath, self.actName)

    def reData(self, path, name):
        self.pathInfo = {}
        for v in range(len(name)):
            self.pathInfo[name[v]] = path[v]

    def dropAct(self):
        getList = self.allAct.selectedItems()
        temList = []
        for w in getList:
            self.agDict[self.taCombo.currentText()].append(str(w.text()))
            temList.append(w.text())
        self.agAct.addItems(temList)

    def tarAgChange(self):
        if not len(self.agDict) == 0:
            if not len(str(self.taCombo.currentText())) == 0:
                curAg = str(self.taCombo.currentText())
                self.agAct.clear()
                if not len(self.agDict[curAg]) == 0:
                    self.agAct.addItems(self.agDict[curAg])
            else:
                return
        else:
            return

    def delAg(self):
        selItem = self.agAct.selectedItems()
        agItems = []
        for g in xrange(self.agAct.count()):
            agItems.append(str(self.agAct.item(g).text()))
        if not selItem:
            return
        for t in selItem:
            del self.agDict[self.taCombo.currentText()][agItems.index(str(t.text()))]
            self.agAct.takeItem(self.agAct.row(t))

    def loadAgent(self):
        getList = []
        loadList = []
        for g in xrange(self.agAct.count()):
            getList.append(str(self.agAct.item(g).text()))
        for s in getList:
            loadList.append(self.pathInfo[s])
        cmds.setAttr("McdGlobal1.activeAgentName", str(self.taCombo.currentText()), type="string")
        for z in loadList:
            McdLoadActions.McdLoadActions(z)

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

################## McdState TransActSetup.py ####################################################

    def actionPick(self):
        acNode = "Action_" + str(self.activeAgent)  # State 노드
        actions = []
        for i in cmds.ls(acNode, dag=True, ap=True):
            if i != acNode:
                actions.append(str(i).split("_action_")[0])  # self.states : State 노드 이름 목록
        return actions

    def actionAssign(self, tarAgent):
        if self.actList:
            dupList = list()
            for j in self.actList:
                targetName = j + "_action_" + tarAgent
                if len(cmds.ls(targetName)) == 0:
                    dupList.append(str(cmds.duplicate(j + "_action_" + self.activeAgent, n=targetName)[0]))
                else:
                    print(targetName + " node is already exist.")
            tarNode = "Action_" + str(tarAgent)
            if len(cmds.ls(tarNode)) != 0:
                if len(dupList) != 0:
                    for k in dupList:
                        cmds.parent(k, tarNode)
                else:
                    pass
            else:
                cmds.error("Check action group for " + str(tarAgent) + " Agent.")
        else:
            cmds.error("Action Node List Error.")

    def decisionPick(self):
        '''
        복사해야할 Decision 지정.
        :return:
            decision - 복사해야할 Decision 노드 이름 리스트. ex) def, actA, ground ...
            subT -  Parent 구조로 되어있는 Decision 노드와 그 하위 노드 이름 딕셔너리. ex) { "ground" : ["groundUp", "groundDown, ...]}
        '''
        dcNode = "Decision_" + str(self.activeAgent)  # State 노드
        decision = []
        subT = dict()
        for i in cmds.ls(dcNode, dag=True, ap=True):
            if cmds.listRelatives(str(i), c=True, type="transform"):
                if i != dcNode:
                    decision.append(str(i).split("_decision_")[0])  # self.states : State 노드 이름 목록
                    subT[str(i).split("_decision_")[0]] = [str(er).split("_decision_")[0] for er in cmds.listRelatives(str(i), c=True, type="transform")]
                else:
                    pass
            else:
                if i != dcNode:
                    decision.append(str(i).split("_decision_")[0])  # self.states : State 노드 이름 목록
                else:
                    pass

        return decision, subT

    def decisionAssign(self, tarAgent):
        '''
        Decision 복사해서 붙여넣기 적용.
        :param tarAgent: 복사된 Decision 노드들을 넣을 Agent 이름 
        :return: None
        '''
        # 페어런츠 구조인 Decision의 경우 Duplicate 할 때 하위 노드까지 복사되면서 같은 이름의 노드가 두개 이상이 되어버리기 때문에 페어런츠 구조를 잠시 풀어둔다.
        for p in self.decList[1].keys():
            for w in self.decList[1][str(p)]:
                cmds.parent(str(w) + "_decision_" + self.activeAgent, "Decision_" + self.activeAgent)
        # Decision 노드들을 대상 에이전트 이름이 붙은 Decision 노드로 Duplicate.
        if self.decList[0]:
            dupList = list()
            for j in self.decList[0]:
                targetName = j + "_decision_" + tarAgent
                if len(cmds.ls(targetName)) == 0:
                    dupList.append(str(cmds.duplicate(j + "_decision_" + self.activeAgent, n=targetName)[0]))
                else:
                    print(targetName + " node is already exist.")
        # 대상 에이전트로 복사된 Decision 노드들 페어런츠
            tarNode = "Decision_" + str(tarAgent)
            if len(cmds.ls(tarNode)) != 0:
                if len(dupList) != 0:
                    for k in dupList:
                        cmds.parent(k, tarNode)
                else:
                    pass
            else:
                cmds.error("Check decision group for " + str(tarAgent) + " Agent.")
        else:
            cmds.error("Decision Node List Error.")
        # 복사된 Decision 노드들 중에서 페어런츠 구조였던 노드들을 복원.
        for e in self.decList[1].keys():
            for w in self.decList[1][str(e)]:
                cmds.parent(str(w) + "_decision_" + tarAgent, str(e) + "_decision_" + tarAgent)
        # 맨 처음에 잠시 풀어두었던 페어런츠 구조를 다시 복원.
        for s in self.decList[1].keys():
            for z in self.decList[1][str(s)]:
                cmds.parent(str(z) + "_decision_" + self.activeAgent, str(s) + "_decision_" + self.activeAgent)

    def pickProc(self):
        self.activeAgent = cmds.getAttr("McdGlobal1.activeAgentName")  # 활성화 되어있는 Agent 이름
        if self.acChk.isChecked() == True:
            self.actList = self.actionPick()
        else:
            pass
        if self.trChk.isChecked() == True:
            self.transitionPick()
        else:
            pass
        if self.deChk.isChecked() == True:
            self.decList = self.decisionPick()
        else:
            pass

    def assignProc(self):
        targetAgent = cmds.getAttr("McdGlobal1.activeAgentName")  # 활성화 되어있는 Agent 이름
        if self.acChk.isChecked() == True:
            self.actionAssign(targetAgent)
        else:
            pass
        if self.trChk.isChecked() == True:
            self.transitionAssign()
        else:
            pass
        if self.deChk.isChecked() == True:
            self.decisionAssign(targetAgent)
        else:
            pass

    def transitionPick(self):
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

    def McdCreateActionShellCmd_OLD(self, acName, prompt=True):
        newAction = acName
        isValid = mg.CheckStringIsValid(newAction)
        if isValid == False:
            cmds.confirmDialog(t="Error", m="New action shell name not valid.")
            raise Exception("New action shell name not valid.")
        if not mg.isReferenceScene():
            activeAgentName = mg.McdGetActiveAgentName()
            mg.McdGetOrCreateActionShellGrp(activeAgentName, 1)
            self.McdCreateAction_OLD(newAction, activeAgentName, prompt)  # McdStateTransActSetup
        else:
            activeAgentName = mg.McdGetActiveAgentName()
            self.McdCreateAction_OLDReference(newAction, activeAgentName, prompt)

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
                cmds.error(actionName + "_acrtionShell_" + str(agName) + " Node is already exist.")

    def crtAs(self, agName, actList):      # Action Multi Register to state Node
        for i in actList:
            actionName = str(i).split("_action_")[0]
            if len(cmds.ls(actionName + "_acrtionShell_" + str(agName))) == 0:
                self.McdCreateActionShellCmd_OLD(actionName)
            else:
                cmds.error(actionName + "_acrtionShell_" + str(agName) + " Node is already exist.")

    def refr_stateList(self):
        acvName = str(cmds.getAttr("McdGlobal1.activeAgentName"))
        stList = [str(i) for i in cmds.ls("*_state_" + acvName)]
        self.nodeCmb.clear()
        if len(stList) != 0:
            self.nodeCmb.addItems(stList)
        else:
            pass

    def connectMultiAction(self):
        acvName = str(cmds.getAttr("McdGlobal1.activeAgentName"))
        seList = [str(i) for i in cmds.ls(sl=True) if str(i).count("_action_" + acvName)]
        asList = list()
        if len(seList) != 0:
            for k in seList:
                asList.append(str(k).replace("_action_","_actionShell_"))
        else:
            cmds.error("Select action file.")
            return
        self.crtAs(acvName, seList)
        curNd = str(self.nodeCmb.currentText())
        inList = list()
        for c in range(len(cmds.listConnections(curNd, s=True))):
            if cmds.listConnections(curNd + ".entryAction[" + str(c) + "]", s=True):
                inList.append(c)
        for w,i in enumerate(range(len(inList), len(inList) + len(asList))):
            cmds.connectAttr(asList[w] + ".output", curNd + ".entryAction[" + str(i) + "]")
            cmds.connectAttr(curNd + ".exitAction", asList[w] + ".input")

    def transitionAssign(self):
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
                duList.append(e)
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
    try:
        myWindow.close()
    except:
        pass
    myWindow = Window()
    myWindow.show()

if __name__ == '__main__':
    main()