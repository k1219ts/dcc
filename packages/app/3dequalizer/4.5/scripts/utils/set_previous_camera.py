#
#
# 3DE4.script.name:	Set Previous Camera
#
# 3DE4.script.comment:	Make the previous Camera the current one.
#
#

cam	= tde4.getCurrentCamera()
camList = tde4.getCameraList()

if cam!=None:
	while cam!=None:
		if cam==None: cam = tde4.getFirstCamera()

		camIndex = 0
		for i in camList:
			if i == cam:
				break
			camIndex = camIndex + 1

		tde4.setCurrentCamera(camList[camIndex-1])
		tde4.updateGUI()
		break
