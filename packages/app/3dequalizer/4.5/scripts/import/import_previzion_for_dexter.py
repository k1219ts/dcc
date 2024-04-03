# Lightcraft Technology
# 
# March 11, 2011
#
# 3DE4.script.name: 	Import Previzion for Dexter
#
# 3DE4.script.version: 	v0.31
#
# 3DE4.script.gui:	Main Window::Dexter::Import Data
#
# 3DE4.script.comment:	Imports Previzion 3D rot/pos/lens data
# 3DE4.script.comment:	based on import_kuper & import_3ality scripts
# 3DE4.script.comment:	assumes left eye is first camera, and is selected
#


# import sdv's python vector lib

from tde4 import *
import string
import sys
from vl_sdv import *


# main script

leftCam = tde4.getCurrentCamera()
leftZoomCurve = tde4.getCameraZoomCurve(leftCam)

ioCurve = getCameraStereoInterocularCurve(leftCam)

pg  = tde4.getCurrentPGroup()

req = tde4.createCustomRequester()
tde4.addFileWidget(req, "file_browser", "Filename...", "*.txt")
ret = tde4.postCustomRequester(req,"Import Previzion...",700,0,"Ok","Cancel")
if ret==1:
	filename = tde4.getWidgetValue(req, "file_browser")
	tde4.deleteAllCurveKeys(leftZoomCurve)
	tde4.deleteAllCurveKeys(ioCurve)
	
	# open will throw if this fails
	f = open(filename,"r")

	frame = 1
	frameFloat = 1.0
	degToRadConv = 3.141592654/180.0
	frameMax = tde4.getCameraNoFrames(leftCam)
	
	while frame <= frameMax:
		line = f.readline()
		if line == "": break
		data = line.split()

		frameNum = data[0]
		leftPosX = float(data[1])
		leftPosY = float(data[2])
		leftPosZ = float(data[3])
		leftRotX = float(data[4]) * degToRadConv
		leftRotY = float(data[5]) * degToRadConv
		leftRotZ = float(data[6]) * degToRadConv
		leftFocalLength = float(data[7])/10.0
		#interocular = float(data[8])

		tde4.setPGroupPosition3D(pg, leftCam, frame, [leftPosX, leftPosY, leftPosZ])

		r3d_l = mat3d(rot3d(leftRotX, leftRotY, leftRotZ, VL_APPLY_XYZ))

		r3dLeft0 = [[r3d_l[0][0],r3d_l[0][1],r3d_l[0][2]],[r3d_l[1][0],r3d_l[1][1],r3d_l[1][2]],[r3d_l[2][0],r3d_l[2][1],r3d_l[2][2]]]

		tde4.setPGroupRotation3D(pg, leftCam, frame, r3dLeft0)

		tde4.createCurveKey(leftZoomCurve,[frameFloat,float(leftFocalLength)])	

		#tde4.createCurveKey(ioCurve, [frameFloat, float(interocular)])
	
		frame = frame + 1
		frameFloat = frameFloat + 1.0
	f.close()
	tde4.setPGroupPostfilterMode(pg,"POSTFILTER_OFF")
	tde4.filterPGroup(pg,leftCam)
tde4.deleteCustomRequester(req)



		

