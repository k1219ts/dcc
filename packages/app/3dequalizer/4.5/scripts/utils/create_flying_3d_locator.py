#
#
# 3DE4.script.name:	Flying 3D Locators...
#
# 3DE4.script.version:	v1.0
#
# 3DE4.script.gui:	Orientation Controls::3D Models::Create
#
# 3DE4.script.comment:	Create flying 3D locators for every selected point.
#
#

# Daehwan Jang(daehwanj@gmail.com)

window_title = "Create Flying 3D Locators v1.0..."

def create_locator():
	m = tde4.create3DModel(pg, 7)
	tde4.set3DModelName(pg, m, "flyingLocator")
	tde4.add3DModelVertex(pg, m, [0.0, 0.0, 0.0])
	tde4.add3DModelVertex(pg, m, [1.0, 0.0, 0.0])
	tde4.add3DModelVertex(pg, m, [-1.0, 0.0, 0.0])
	tde4.add3DModelVertex(pg, m, [0.0, 1.0, 0.0])
	tde4.add3DModelVertex(pg, m, [0.0, -1.0, 0.0])
	tde4.add3DModelVertex(pg, m, [0.0, 0.0, 1.0])
	tde4.add3DModelVertex(pg, m, [0.0, 0.0, -1.0])
	for i in range(7):
		tde4.add3DModelLine(pg, m, [0, i])
	tde4.set3DModelSurveyFlag(pg, m, 0)
	tde4.set3DModelPosition3D(pg, m, p3d)

c = tde4.getCurrentCamera()
pg = tde4.getCurrentPGroup()
if pg!=None and c!=None:
	pl = tde4.getPointList(pg, 1)
	if len(pl)!=0:
		for p in pl:
			if tde4.isPointCalculated3D(pg, p):
				p3d = tde4.getPointCalcPosition3D(pg, p)
				create_locator()
	else:
		tde4.postQuestionRequester(window_title, "Select point to create flying 3d locator.", "Ok")
else:
	tde4.postQuestionRequester(window_title, "There is no point group or camera.", "Ok")
