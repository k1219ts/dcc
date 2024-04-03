import nuke, nukescripts

from PySide2 import QtWidgets, QtCore

changeAllBool = False
syncKnobDic = {}

def updateKnobDic():
    global syncKnobDic
    syncKnobDic = {}

    for i in nuke.selectedNodes():
        nName = i.name()
        syncKnobDic[nName] = {}
        for j in i.allKnobs():
            kName = j.name()
            cValue = i.knob(kName).toScript()
            syncKnobDic[nName][kName] = cValue

def changeAll():
    global changeAllBool
    global syncKnobDic

    sn = nuke.selectedNodes()

    if changeAllBool:
        # Return to Original
        nuke.toNode('preferences')['UIBackColor'].setValue(842150655)
        changeAllBool = False
        syncKnobDic = {}
        nuke.removeKnobChanged(knobSync)

    else:
        # Modify Mode
        nuke.toNode('preferences')['UIBackColor'].setValue(1392531199)
        changeAllBool = True
        updateKnobDic()
        nuke.addKnobChanged(knobSync)



def knobSync():
    sn = nuke.selectedNodes()
    global syncKnobDic
    returnList = ['xpos', 'ypos', 'selected', 'showPanel', 'hidePanel']

    changeKnobName = nuke.thisKnob().name()
    changeNodeName = nuke.thisNode().name()

    if nuke.thisKnob().name() in returnList:
        return

    changedValue = nuke.thisNode()[changeKnobName].toScript()
    mod = QtWidgets.QApplication.keyboardModifiers()

    if mod == QtCore.Qt.ShiftModifier:
        if not(syncKnobDic.get(changeNodeName)):
            updateKnobDic()

        orgValue = syncKnobDic[changeNodeName][changeKnobName]

        try:
            plusdelta = -(float(orgValue) - float(changedValue))
            syncKnobDic[changeNodeName][changeKnobName] = changedValue
            for i in sn:
                if i == nuke.thisNode():
                    continue
                if nuke.thisClass() == i.Class():
                    addValue = str(float(i[changeKnobName].toScript()) + plusdelta)
                    i[changeKnobName].fromScript(addValue)


        except ValueError:
            for i in sn:
                if nuke.thisClass() == i.Class():
                    i[changeKnobName].fromScript(changedValue)

    else:
        for i in sn:
            if nuke.thisClass() == i.Class():
                i[changeKnobName].fromScript(changedValue)
