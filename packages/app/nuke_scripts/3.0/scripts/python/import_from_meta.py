import json, nuke, nukescripts
from ast import literal_eval

def import_from_meta():
    baseNode = nuke.selectedNode()
    baseFile = baseNode['file'].value()
    if baseFile.endswith('.mov'):
        info = baseNode.metadata()['quicktime/nukeInfo']
        infoDic = literal_eval(info.replace('\\', ''))

    elif baseFile.endswith('.exr'):
        baseKey = ['renderNK', 'saveNK', 'writeNode', 'wFilePath']
        infoDic = {}
        for i in baseKey:
            infoDic[i] = baseNode.metadata()['exr/nuke/%s' % i]

    ok = nuke.ask('open new script??')
    if ok:
        nuke.scriptOpen(infoDic['renderNK'])

    else:
        beforeAllNode = nuke.allNodes()
        nuke.scriptReadFile(infoDic['renderNK'])
        afterAllNode = nuke.allNodes()

        loadedNodes = [item for item in afterAllNode if item not in beforeAllNode]
        xOffset = 0
        yOffset = 0

        if baseFile.endswith('.mov'):
            for i in loadedNodes:
                if i.Class() == 'Write':

                    if i['file'].value().endswith('jpg'):
                        print(i.name(), i['file'].value())
                        xOffset = baseNode.xpos() - i.xpos()
                        yOffset = baseNode.ypos() - i.ypos() - 50
                        break

        elif baseFile.endswith('.exr'):
            for i in loadedNodes:
                if i.Class() == 'Write':
                    if i['file'].value() == infoDic['wFilePath']:
                        xOffset = baseNode.xpos() - i.xpos()
                        yOffset = baseNode.ypos() - i.ypos() - 50
                        break

        print(xOffset, yOffset)
        for i in loadedNodes:
            i.setXYpos(i.xpos() + xOffset, i.ypos() + yOffset)
