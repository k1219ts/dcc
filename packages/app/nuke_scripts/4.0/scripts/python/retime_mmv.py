import nuke

def retimeDistort():
    srcDist = None
    srcTW = None
    srcFR = None
    fDic = {}
    ignoreList = ['xpos', 'ypos', 'selected', 'showPanel', 'hidePanel']

    # CHECK NODE CLASS
    #------------------------------------------------------------------------------
    sl = nuke.selectedNodes()
    for i in sl:
        print(i.Class())
        if i.Class().startswith('LD_3DE'):
            srcDist = i
        if i.Class() == 'TimeWarp':
            srcTW = i
        if i.Class() == 'FrameRange':
            srcFR = i
    #------------------------------------------------------------------------------

    startF = int(srcFR['first_frame'].value())
    endF = int(srcFR['last_frame'].value())

    for i in range(startF, endF+1):
        fDic[i] = srcTW['lookup'].valueAt(i)

    destDist = nuke.createNode(srcDist.Class())
    destDist.setInput(0, None)

    for knob in srcDist.allKnobs():

        if knob.name() in ignoreList:
            continue
        if hasattr(knob, 'animations'):
            if knob.isAnimated():
                destDist[knob.name()].setAnimated()
                for t in sorted(fDic.keys()):
                    fromValue = knob.valueAt(fDic[t])
                    destDist[knob.name()].setValueAt(fromValue, t)
            else:
                fromValue = knob.toScript()
                destDist[knob.name()].fromScript(fromValue)
        else:
            fromValue = knob.toScript()
            destDist[knob.name()].fromScript(fromValue)
    destDist.setInput(0, None)
    destDist.setXYpos(srcDist.xpos()+150, srcDist.ypos())
    destDist['tile_color'].setValue(65535)
