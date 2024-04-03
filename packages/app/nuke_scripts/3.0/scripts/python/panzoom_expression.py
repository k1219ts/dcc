import nuke

def panzoom_expression():
    camList = []
    for i in nuke.selectedNodes():
        if 'Camera' in i.Class():
            camList.append(i)
    print(camList)

    pan = None
    targets = []

    for cam in camList:
        if '2dpanzoom' in cam.name().lower():
            pan = cam
        else:
            targets.append(cam)

    for target in targets:
        target['win_translate'].setExpression(pan.name() + '.win_translate')
        target['win_scale'].setExpression(pan.name() + '.win_scale')
