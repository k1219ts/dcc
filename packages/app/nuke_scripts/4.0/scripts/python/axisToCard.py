import nuke


def camNamePanel():
    nPanel = nuke.Panel("Camera input")
    nPanel.addSingleLineInput("Camera_Name : ", "")
    retVar = nPanel.show()
    camVar = nPanel.value("Camera_Name : ")
    return camVar


def axisToCard():
    camName = camNamePanel()
    print(camName)

    # lookObject = camName

    allAxis = nuke.selectedNodes()
    cpXList = []
    cpYList = []
    cdList = []
    for node in allAxis:
        trans = node.knob('translate').getValue()
        cd = nuke.nodes.Card2()
        cd.knob('uniform_scale').setValue(5)
        cd.knob('translate').setValue(trans)

        xX = 'degrees(atan2(' + camName + '.translate.y-translate.y,sqrt(pow(' + camName + '.translate.x-translate.x,2)+pow(' + camName + '.translate.z-translate.z,2))))'
        yX = camName + '.translate.z-this.translate.z >= 0 ? 180+degrees(atan2(' + camName + '.translate.x-translate.x,' + camName + '.translate.z-translate.z)):180+degrees(atan2(' + camName + '.translate.x-translate.x,' + camName + '.translate.z-translate.z))'

        cd.knob('rotate').setExpression(xX, 0)
        cd.knob('rotate').setExpression(yX, 1)

        cpXList += [cd.xpos()]
        cpYList += [cd.ypos()]
        cdList += [cd.name()]
        node.setSelected(False)

    xpNum = len(cpXList)
    xpSum = sum(cpXList) / xpNum
    ypNum = len(cpYList)
    ypSum = sum(cpYList) / ypNum

    for node in cdList:
        scNode = nuke.toNode(node)
        scNode.setSelected(True)

    sc = nuke.createNode('Scene', inpanel=False)
    sc['xpos'].setValue(xpSum)
    sc['ypos'].setValue(200)

    ctList = []
    for i in range(0, sc.inputs()):
        ctList += [sc.input(i).name()]

    for node in ctList:
        ctNode = nuke.toNode(node)
        ctNode.setSelected(True)

    sc.setSelected(True)

    ctAllNode = nuke.selectedNodes()

    nuke.zoom(0.3, [xpSum, ypSum])
