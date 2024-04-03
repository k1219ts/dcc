import maya.cmds as mc

def export_2dPanZoom_to_houdini():
	cam = mc.ls(sl=1, dag=1, ca=1)
	if len(cam) == 1:
		min = int(mc.playbackOptions(q=1, min=1))
		max = int(mc.playbackOptions(q=1, max=1))
		
		filename = mc.fileDialog2(fileFilter="*.cmd", dialogStyle=2)
		f = open(filename[0], "w")
		if not f.closed:
			f.write("set cam=/`opselectrecurse('/', 0)`\n")
			f.write("opparm $cam res (%s %s)\n"%(mc.getAttr("defaultResolution.width"), mc.getAttr("defaultResolution.height")))
			f.write("chadd $cam winx winy\n")
			f.write("chadd $cam winsizex winsizey\n")
			f.write("frange %i %i\n"%(min, max))
			f.write("fcur %i\n"%min)
			for frame in range(min, max+1):
				mc.currentTime(frame, e=1)
				winx = mc.getAttr(cam[0]+".horizontalPan") / mc.getAttr(cam[0]+".hfa")
				winy = mc.getAttr(cam[0]+".verticalPan") / mc.getAttr(cam[0]+".vfa")
				winsizex = mc.getAttr(cam[0]+".zoom")
				winsizey = mc.getAttr(cam[0]+".zoom")
				f.write("chkey -f %i -v %.15f $cam/winx\n"%(frame, winx))
				f.write("chkey -f %i -v %.15f $cam/winy\n"%(frame, winy))
				f.write("chkey -f %i -v %.15f $cam/winsizex\n"%(frame, winsizex))
				f.write("chkey -f %i -v %.15f $cam/winsizey\n"%(frame, winsizey))

			f.close()
	else:
		mc.confirmDialog(title="Export 2D Pan/Zoom to Houdini", message="Select a Camera First!", button=["Ok"], defaultButton="Ok")
