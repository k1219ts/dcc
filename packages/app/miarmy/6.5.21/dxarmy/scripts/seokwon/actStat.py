# encoding:utf-8
# Miamry Duplicate Actions

import os
import maya.cmds as cmds
import McdGeneral as mg
import McdSimpleCmd as sc
import McdAgentManager
from pymodule.Qt import QtCore, QtGui, QtWidgets, QtCompat
chList = list()
chName = list()

currentpath = os.path.abspath(__file__)
uiFile = os.path.join(os.path.dirname(currentpath), "ui", "actStat.ui")

def setup_ui(ui, base_instance=None):
    for member in dir(ui):
        if not member.startswith('__') and member is not 'staticMetaObject':
            setattr(base_instance, member, getattr(ui, member))

class DragDropTest(QtWidgets.QListWidget):
    def __init__(self, parent):
        QtWidgets.QListWidget.__init__(self, parent)
        self.setAcceptDrops(True)
        self.setDragEnabled(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls:
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls:
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasUrls:
            event.accept()
            lst = list()
            for i in event.mimeData().urls():
                lst.append(i.toLocalFile().split(os.sep)[-1].split(".ma")[0])
                chList.append(str(i.toLocalFile()))
                chName.append(str(i.toLocalFile().split(os.sep)[-1]).split(".ma")[0])
            self.addItems(lst)
        else:
            event.ignore()

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Delete:
            self._del_item()

    def _del_item(self):
        selItem = self.selectedItems()
        if not selItem:
            return
        for j in selItem:
            self.takeItem(self.row(j))
            del chList[chName.index(str(j.text()))]
            del chName[chName.index(str(j.text()))]

class Window(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(Window, self).__init__(parent)
        ui = QtCompat.load_ui(uiFile)
        setup_ui(ui, self)
        self.connectSignal()
        self.mayafile_tmp = ''

    def connectSignal(self):
        # Action Register
        self.clBtn.clicked.connect(self.checkShellNode)
        self.refrBtn.setIcon(QtGui.QPixmap("/dexter/Cache_DATA/animation/A0_Artist/Choi_SeokWon/icons/ref.png"))
        self.refrBtn.clicked.connect(self.refr_agentList)
        self.raBtn.clicked.connect(self.connectMultiAction)
        self.cpActBtn.clicked.connect(self.cpAction)
        self.acChk.stateChanged.connect(self.checkActChk)
        self.agentCmb.currentIndexChanged.connect(self.refr_ag)
        self.nodeCmb.currentIndexChanged.connect(self.refr_stActList)
        self.refr_agentList()
        self.checkActChk()
        # Contents Copy
        self.pickBtn.clicked.connect(self.pickProc)
        self.assignBtn.clicked.connect(self.assignProc)
        self.refBtn.clicked.connect(self.refr_ContCopy)
        self.refBtn.setIcon(QtGui.QPixmap("/dexter/Cache_DATA/animation/A0_Artist/Choi_SeokWon/icons/ref.png"))
        self.refr_ContCopy()

    def checkActChk(self):
        cst = self.acChk.checkState()
        if str(cst) == "PySide2.QtCore.Qt.CheckState.Unchecked":
            self.copyLw.close()
            self.allAct.close()
        else:
            self.copyLw.show()
            self.allAct.show()

    def refr_ag(self):
        curAg = str(self.agentCmb.currentText())
        cmds.setAttr("McdGlobal1.activeAgentName", curAg, type="string")
        self.refr_stateList()
        self.refr_allActionList()
        self.refr_avbList()

    def chkDigit(self, name):
        i = 1
        while name[-i:].isdigit() == 1:
            i += 1
            if len(name) < i:
                break
        return i - 1

    def autoNumb(self, name, *postName):    # Auto Numbering Func ( Name, *postname )
        if postName:
            newName = name + postName[0]
            nmb = self.chkDigit(name)
            if nmb == 0:
                n = 1
                while cmds.ls(newName):
                    n += 1
                    newName = newName.split(postName[0])[0] + (("%0" + str(len(str(n))) + "d") % n) + postName[0]
            else:
                n = 0
                while cmds.ls(newName):
                    n += 1
                    newName = newName.split(postName[0])[0][:-nmb] + (("%0" + str(nmb) + "d") % (int(name.split(postName[0])[0][-nmb:]) + n)) + postName[0]
        else:
            newName = name
            nmb = self.chkDigit(newName)
            if nmb == 0:
                n = 1
                while cmds.ls(newName):
                    n += 1
                    newName = newName + (("%0" + str(len(str(n))) + "d") % n)
            else:
                n = 0
                while cmds.ls(newName):
                    n += 1
                    newName = newName[:-nmb] + (("%0" + str(nmb) + "d") % (int(name[-nmb:]) + n))
        return newName

    def refr_ContCopy(self):
        self.agentCmb2.clear()
        self.agentCmb3.clear()
        agentList = [str(i) for i in McdAgentManager.McdGetAllAgentTypeNIDWithColor()[0]]
        self.agentCmb2.addItems(agentList)
        self.agentCmb3.addItems(agentList)
        self.copyLw.clear()

    def refr_agentList(self):
        self.agentCmb.clear()
        agentList = [str(i) for i in McdAgentManager.McdGetAllAgentTypeNIDWithColor()[0]]
        self.agentCmb.addItems(agentList)  # AgentList Refresh

    def checkActions(self, tarAgent):
        for i in self.acNode.keys():
            dk = i.replace(self.activeAgent, tarAgent)
            nwAcNode = dk.replace("_actionShell_", "_action_")
            if len(cmds.ls(nwAcNode)) != 0:
                pass
            else:
                cmds.warning("There is no " + nwAcNode + " node.")

    def refr_allActionList(self):
        curAg = str(self.agentCmb.currentText())
        actList = [str(i).split("_action_")[0] for i in cmds.listRelatives("Action_" + curAg, c=True)]
        self.actLw.clear()
        self.actLw.addItems(actList)

    def refr_stateList(self):
        acvName = str(self.agentCmb.currentText())
        if cmds.ls("*_state_" + acvName):
            stList = [str(i).split("_state_")[0] for i in cmds.ls("*_state_" + acvName)]
        else:
            self.nodeCmb.clear()
            self.stActLw.clear()
            return
        self.nodeCmb.clear()
        self.nodeCmb.addItems(stList)
        self.refr_stActList()

    def refr_stActList(self):
        curAg = str(self.agentCmb.currentText())
        curSt = str(self.nodeCmb.currentText())
        if curSt:
            stActList = list()
            if cmds.listConnections(curSt + "_state_" + curAg + ".exitAction", s=True, t="McdActionShell"):
                lst = cmds.listConnections(curSt + "_state_" + curAg + ".exitAction", s=True, t="McdActionShell")
                for i in lst:
                    if cmds.listConnections(str(i) + ".output")[0] == (curSt + "_state_" + curAg):
                        stActList.append(str(i).split("_actionShell_")[0])
                self.stActLw.clear()
                self.stActLw.addItems(stActList)
            else:
                self.stActLw.clear()
            self.refr_avbList()
        else:
            self.refr_avbList()

    def refr_avbList(self):
        curAg = str(self.agentCmb.currentText())
        curSt = str(self.nodeCmb.currentText())
        actList = [str(i).split("_action_")[0] for i in cmds.listRelatives("Action_" + curAg, c=True)]
        if curSt:
            stList = [str(i).split("_state_")[0] for i in cmds.ls("*_state_" + curAg)]
            for i in stList:
                if cmds.listConnections(str(i) + "_state_" + curAg + ".exitAction", s=True, t="McdActionShell"):
                    stLs = cmds.listConnections(str(i) + "_state_" + curAg + ".exitAction", s=True, t="McdActionShell")
                    stActList = list()
                    for k in stLs:
                        if cmds.listConnections(str(k) + ".output")[0] == (str(i) + "_state_" + curAg):
                            stActList.append(str(k).split("_actionShell_")[0])
                    for j in stActList:
                        if j in actList:
                            actList.remove(j)
                else:
                    self.avbLw.clear()
                    self.avbLw.addItems(actList)
                    return
            avbList = list()
            for z in actList:
                if not cmds.ls(z + "_actionShell_" + curAg, type="McdActionShell"):
                    avbList.append(z)
                else:
                    if cmds.listConnections(z + "_actionShell_" + curAg, type="McdState") != None:
                        asL = cmds.listConnections(z + "_actionShell_" + curAg, type="McdState")
                        if asL[0] != asL[1]:
                            pass
                        else:
                            cmds.error("Check " + z + "_actionShell_" + curAg + " Node.")
                    else:
                        cmds.delete(z + "_actionShell_" + curAg)
                        avbList.append(z)
            self.avbLw.clear()
            self.avbLw.addItems(avbList)
        else:
            self.avbLw.clear()
            self.avbLw.addItems(actList)
            return

    def checkShellNode(self):
        dsnList = list()
        agn = str(self.agentCmb.currentText())
        if agn:
            pass
        else:
            cmds.confirmDialog(title="Warning", message="Select Agent Name")
            return
        for i in cmds.ls("*_" + agn, type="McdActionShell"):
            if cmds.listConnections(str(i)) == None:
                dsnList.append(str(i))
            elif len(cmds.listConnections(str(i))) == 1:
                dsnList.append(str(i))
            elif not cmds.ls(str(i).split("_actionShell_")[0] + "_action_" + agn, type="McdAction"):
                dsnList.append(str(i))
        te = ""
        for v in dsnList:
            te += str(v) + "\n"
        if dsnList:
            reslt = cmds.confirmDialog(title='Disconnected Node List', message=te, button=['Delete', 'Cancel'], defaultButton='Delete', cancelButton='Cancel', dismissString='Cancel')
            if reslt == 'Delete':
                for w in dsnList:
                    cmds.delete(w)
        else:
            cmds.confirmDialog(title='Notice', message="Nothing to clean up")

    def mcdCrStateCmd(self, stName):
        newState = stName
        if not mg.isReferenceScene():
            activeAgentName = mg.McdGetActiveAgentName()
            mg.McdGetOrCreateTransitionMapGrp(activeAgentName, 1)
            mg.McdCreateState(newState, activeAgentName)
        else:
            activeAgentName = mg.McdGetActiveAgentName()
            mg.McdCreateStateReference(newState, activeAgentName)

    def cpAction(self):
        curAg = str(self.agentCmb.currentText())
        seList = [str(i.text()) for i in self.actLw.selectedItems()]
        for j in seList:
            newName = self.autoNumb(j, "_action_" + curAg)
            cmds.duplicate(j + "_action_" + curAg, rr=1, n=newName)
        self.refr_agentList()

    def connectMultiAction(self):
        acvName = str(self.agentCmb.currentText())
        seList = [str(cn.text())+"_action_"+acvName for cn in self.avbLw.selectedItems()]
        asList = list()
        if len(seList) != 0:
            for k in seList:
                asList.append(str(k).replace("_action_","_actionShell_"))
        else:
            cmds.error("Select action file.")
            return
        self.crtAs(acvName, seList)
        curNd = str(self.nodeCmb.currentText()) + "_state_" + acvName
        inList = list()
        exl = list()
        if cmds.listConnections(curNd, s=True):
            for c in range(len(cmds.listConnections(curNd, s=True))):
                if cmds.listConnections(curNd + ".entryAction[" + str(c) + "]", s=True):
                    inList.append(c)
            for ex in range(len(inList)):
                if ex in inList:
                    pass
                else:
                    exl.append(ex)
            for w, i in enumerate(range(len(inList) + len(exl), len(inList) + len(exl) + len(asList))):
                cmds.connectAttr(asList[w] + ".output", curNd + ".entryAction[" + str(i) + "]")
                cmds.connectAttr(curNd + ".exitAction", asList[w] + ".input")
        else:
            for v, m in enumerate(asList):
                cmds.connectAttr(asList[v] + ".output", curNd + ".entryAction[" + str(v) + "]")
                cmds.connectAttr(curNd + ".exitAction", m + ".input")
        self.refr_stActList()
        self.refr_avbList()

    def crtAs(self, agName, actList):      # Action Multi Register to state Node
        for i in actList:
            actionName = str(i).split("_action_")[0]
            if len(cmds.ls(actionName + "_acrtionShell_" + str(agName))) == 0:
                self.McdCreateActionShellCmd_OLD(actionName)
            else:
                cmds.error(actionName + "_acrtionShell_" + str(agName) + " Node is already exist.")

    def McdCreateActionShellCmd_OLD(self, acName, prompt=True):
        newAction = acName
        isValid = mg.CheckStringIsValid(newAction)
        if isValid == False:
            cmds.confirmDialog(t="Error", m="New action shell name not valid.")
            raise Exception("New action shell name not valid.")
        if not mg.isReferenceScene():
            activeAgentName = str(cmds.getAttr("McdGlobal1.activeAgentName"))
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

    def pickProc(self):
        self.activeAgent = str(self.agentCmb2.currentText())  # 활성화 되어있는 Agent 이름
        if self.acChk.isChecked() == True:
            self.actList = self.actionPick()
        if self.trChk.isChecked() == True:
            self.transitionPick()
        if self.deChk.isChecked() == True:
            self.decList = self.decisionPick()

    def assignProc(self):
        targetAgent = str(self.agentCmb3.currentText())  # 활성화 되어있는 Agent 이름
        cmds.setAttr("McdGlobal1.activeAgentName", targetAgent, type="string")
        if self.acChk.isChecked() == True:
            self.actionAssign(targetAgent)
        if self.trChk.isChecked() == True:
            self.transitionAssign()
        if self.deChk.isChecked() == True:
            self.decisionAssign(targetAgent)

    def actionPick(self):
        acNode = "Action_" + str(self.activeAgent)  # State 노드
        actions = []
        for i in cmds.listRelatives(acNode, c=True):
            actions.append(str(i).split("_action_")[0])  # self.states : State 노드 이름 목록
        self.copyLw.clear()
        self.copyLw.addItems(actions)
        return actions

    def createActshell(self, agName):
        for i in self.acNode.keys():
            actionName = str(i).split("_actionShell_")[0]
            if len(cmds.ls(actionName + "_acrtionShell_" + str(agName))) == 0:
                self.McdCreateActionShellCmd_OLD(actionName)
            else:
                cmds.error(actionName + "_acrtionShell_" + str(agName) + " Node is already exist.")

    def actionAssign(self, tarAgent):
        if self.allAct.isChecked() == True:
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
        else:
            seList = [str(i.text()) for i in self.copyLw.selectedItems()]
            if seList:
                dupList = list()
                for j in seList:
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

    def transitionAssign(self):
        targetAgent = str(self.agentCmb3.currentText())
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
                elif w.count("message") == 1:
                    pass
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