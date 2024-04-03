import os
import maya.cmds as mc

def DD_exportRetime2Nuke():
    # this version is for imageplane.
    ip = mc.ls(sl=1, dag=1, type="imagePlane")

    if len(ip) == 1:
        min = int(float(mc.playbackOptions(q=1, min=1)))
        max = int(float(mc.playbackOptions(q=1, max=1)))

        filename = mc.fileDialog2(fileFilter="*.nk", dialogStyle=2)
        f = open(filename[0], "w")
        if not f.closed:
            # hide all objects.
            for panName in mc.getPanel(all=True):
                if 'modelPanel' in panName:
                    mc.isolateSelect(panName, state=1)

            f.write("TimeWarp {\n")
            f.write(" lookup {{curve x%i"%min)
            for frame in range(min, max+1):
                mc.currentTime(frame, e=1)
                retimedFrame = mc.getAttr(ip[0]+".frameExtension")
                f.write(" %d"%retimedFrame)
            f.write("}}\n")
            f.write(" filter none\n")
            f.write(" name %s\n"%(os.path.split(mc.getAttr(ip[0]+".imageName"))[1]))
            f.write("}\n")

            f.close()

            # show all objects.
            for panName in mc.getPanel(all=True):
                if 'modelPanel' in panName:
                    mc.isolateSelect(panName, state=0)
    else:
        mc.confirmDialog(title="exportRetime2Nuke", message="Select one imagePlane retimed.", button=["Ok"], defaultButton="Ok")
