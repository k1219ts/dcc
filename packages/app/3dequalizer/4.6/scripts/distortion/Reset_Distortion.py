#
#
# 3DE4.script.name:	Reset Distortion...
#
# 3DE4.script.version:		v1
#
# 3DE4.script.comment:	Resets Distortion Values to its default.
#
# 3DE4.script.gui:	Lineup Controls::Edit
# 3DE4.script.gui:	Distortion Edit Controls::Edit
# 3DE4.script.gui.button:	Lineup Controls::Reset Distortion, align-bottom-right, 70, 20
# 3DE4.script.gui.button:	Distortion Edit Controls::Reset Distortion, align-bottom-right, 70, 20
#Author: Vinod Kumar Padakantoju(vinodh.vfxartist@gmail.com)

cam = tde4.getCurrentCamera()
frm = tde4.getCameraNoFrames(cam)
l = tde4.getCameraLens(cam)
model = tde4.getLensLDModel(l)
if model == "3DE Classic LD Model":
	for i in range(1, frm+1):
		fl = tde4.getCameraFocalLength(cam, i)
		fc = tde4.getCameraFocus(cam, i)
		para = tde4.getLDModelParameterName(model, 0)
		tde4.setLensLDAdjustableParameter(l, para, fl, fc,0)
		para = tde4.getLDModelParameterName(model, 1)
		tde4.setLensLDAdjustableParameter(l, para, fl, fc,1)
		for j in range(2,5):
			para = tde4.getLDModelParameterName(model, j)
			tde4.setLensLDAdjustableParameter(l, para, fl, fc,0)
if model == "3DE4 Radial - Standard, Degree 4":
	for i in range(1, frm+1):
		fl = tde4.getCameraFocalLength(cam, i)
		fc = tde4.getCameraFocus(cam, i)
		for j in range(0,8):
			para = tde4.getLDModelParameterName(model, j)
			tde4.setLensLDAdjustableParameter(l, para, fl, fc,0)
if model == "3DE4 Anamorphic - Standard, Degree 4":
	for i in range(1, frm+1):
		fl = tde4.getCameraFocalLength(cam, i)
		fc = tde4.getCameraFocus(cam, i)
		for j in range(0,11):
			para = tde4.getLDModelParameterName(model, j)
			tde4.setLensLDAdjustableParameter(l, para, fl, fc,0)
		para = tde4.getLDModelParameterName(model, 11)
		tde4.setLensLDAdjustableParameter(l, para, fl, fc,1)
		para = tde4.getLDModelParameterName(model, 12)
		tde4.setLensLDAdjustableParameter(l, para, fl, fc,1)
if model == "3DE4 Anamorphic, Degree 6":
	for i in range(1, frm+1):
		fl = tde4.getCameraFocalLength(cam, i)
		fc = tde4.getCameraFocus(cam, i)
		for j in range(0,18):
			para = tde4.getLDModelParameterName(model, j)
			tde4.setLensLDAdjustableParameter(l, para, fl, fc,0)			
if model == "3DE4 Radial - Fisheye, Degree 8":
		for i in range(1, frm+1):
			fl = tde4.getCameraFocalLength(cam, i)
			fc = tde4.getCameraFocus(cam, i)
			for j in range(0, 4):
				para = tde4.getLDModelParameterName(model, j)
				tde4.setLensLDAdjustableParameter(l, para, fl, fc,0)