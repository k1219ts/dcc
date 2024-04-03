#
#
# 3DE4.script.name:	Bake Pivot of 3D Models...
#
# 3DE4.script.version:	v1.0
#
# 3DE4.script.gui:	Orientation Controls::3D Models
#
# 3DE4.script.comment:	Bake pivot of every selected 3D Model. So They contain survey data...
#
#

# Daehwan Jang(daehwanj@gmail.com)

window_title = "Bake Pivot of 3D Models v1.0..."

def create_locator(m3d):
	mname = tde4.get3DModelName(pg, m)
	new_m = tde4.create3DModel(pg, 7)
	tde4.set3DModelName(pg, new_m, "baked_%s"%mname)
	tde4.add3DModelVertex(pg, new_m, [m3d[0], m3d[1], m3d[2]])
	tde4.add3DModelVertex(pg, new_m, [m3d[0]+10.0, m3d[1], m3d[2]])
	tde4.add3DModelVertex(pg, new_m, [m3d[0]-10.0, m3d[1], m3d[2]])
	tde4.add3DModelVertex(pg, new_m, [m3d[0], m3d[1]+10.0, m3d[2]])
	tde4.add3DModelVertex(pg, new_m, [m3d[0], m3d[1]-10.0, m3d[2]])
	tde4.add3DModelVertex(pg, new_m, [m3d[0], m3d[1], m3d[2]+10.0])
	tde4.add3DModelVertex(pg, new_m, [m3d[0], m3d[1], m3d[2]-10.0])
	for i in range(7):
		tde4.add3DModelLine(pg, new_m, [0, i])
	tde4.set3DModelSurveyFlag(pg, new_m, 1)

c = tde4.getCurrentCamera()
pg = tde4.getCurrentPGroup()
frame = tde4.getCurrentFrame(c)
if c!=None and pg!=None:
	ml = tde4.get3DModelList(pg, 1)
	if len(ml)!=0:
		for m in ml:
			m3d = tde4.get3DModelPosition3D(pg, m, c, frame)
			create_locator(m3d)
	else:
		tde4.postQuestionRequester(window_title, "Select 3d model to bake pivot.", "Ok")
else:
	tde4.postQuestionRequester(window_title, "There is no point group or camera.", "Ok")