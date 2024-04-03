import maya.cmds as mc

def bakeIntoWorldSpace():
    objList = mc.ls(sl=1)

    # hide all objects.
    for panName in mc.getPanel(all=True):
        if 'modelPanel' in panName:mc.isolateSelect(panName, state=1)
    ###
    for i in objList:
        run_bakeIntoWorldSpace(i)
    # show all objects.
    for panName in mc.getPanel(all=True):
        if 'modelPanel' in panName:mc.isolateSelect(panName, state=0)
    ###

    print "bake into world space done."

def run_bakeIntoWorldSpace(obj):
    tMatrix = []

    frameMin = int(mc.playbackOptions(q=1, min=1))
    frameMax = int(mc.playbackOptions(q=1, max=1))

    for i in range(frameMin, frameMax+1):
        mc.currentTime(i)
        tMatrix.append( mc.xform(obj, q=1, matrix=1, worldSpace=1) )

    try:
        mc.parent(obj, world=1)
    except:
        print "object is already in worldspace."
    count = 0

    for i in range(frameMin, frameMax+1):
        mc.currentTime(i)
        mc.xform(obj, matrix=tMatrix[count])
        mc.setKeyframe(obj)
        count += 1
