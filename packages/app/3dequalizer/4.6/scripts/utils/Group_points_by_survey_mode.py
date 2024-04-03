# 3DE4.script.name:	Group points by survey mode...
# 3DE4.script.version:	v1.0b1
# 3DE4.script.gui.config_menus: true
# 3DE4.script.gui:	Object Browser::Edit
# 3DE4.script.comment:	group all points by their survey flag.
# Author : Patcha Saheb (patchasaheb@gmail.com)
# 15 March 2016.

window_title = "Patcha Group points by Survey mode_v1.0b1"

def Apply_callback(req,widget,action):
	pg = tde4.getCurrentPGroup()
	cam = tde4.getCurrentCamera()
	frame = tde4.getCurrentFrame(cam)
	pl = tde4.getPointList(pg,0)

	if tde4.getWidgetValue(req,"survey") == 1:
		l = tde4.getSetList(pg,0)		
		for pointset in l:
			setname = tde4.getSetName(pg,pointset)
			if setname == "Survey":
				tde4.deleteSet(pg,pointset)
		survey_group = tde4.createSet(pg)
		tde4.setSetName(pg,survey_group,"Survey")		
		for point in pl:
			mode = tde4.getPointSurveyMode(pg,point)
			if mode == "SURVEY_EXACT":
				tde4.setPointSet(pg,point,survey_group)

	if tde4.getWidgetValue(req,"approx_survey") == 1:
		l = tde4.getSetList(pg,0)		
		for pointset in l:
			setname = tde4.getSetName(pg,pointset)
			if setname == "Approx Survey":
				tde4.deleteSet(pg,pointset)
		approx_survey_group = tde4.createSet(pg)
		tde4.setSetName(pg,approx_survey_group,"Approx Survey")		
		for point in pl:
			mode = tde4.getPointSurveyMode(pg,point)
			if mode == "SURVEY_APPROX":
				tde4.setPointSet(pg,point,approx_survey_group)		

	if tde4.getWidgetValue(req,"survey_free") == 1:
		l = tde4.getSetList(pg,0)		
		for pointset in l:
			setname = tde4.getSetName(pg,pointset)
			if setname == "Survey Free":
				tde4.deleteSet(pg,pointset)
		survey_free_group = tde4.createSet(pg)
		tde4.setSetName(pg,survey_free_group,"Survey Free")		
		for point in pl:
			mode = tde4.getPointSurveyMode(pg,point)
			if mode == "SURVEY_FREE":
				tde4.setPointSet(pg,point,survey_free_group)

	if tde4.getWidgetValue(req,"lineup") == 1:
		l = tde4.getSetList(pg,0)		
		for pointset in l:
			setname = tde4.getSetName(pg,pointset)
			if setname == "Lineup":
				tde4.deleteSet(pg,pointset)
		lineup_group = tde4.createSet(pg)
		tde4.setSetName(pg,lineup_group,"Lineup")		
		for point in pl:
			mode = tde4.getPointSurveyMode(pg,point)
			if mode == "SURVEY_LINEUP":
				tde4.setPointSet(pg,point,lineup_group)

	if tde4.getWidgetValue(req,"tri") == 1:
		l = tde4.getSetList(pg,0)		
		for pointset in l:
			setname = tde4.getSetName(pg,pointset)
			if setname == "Triangulated":
				tde4.deleteSet(pg,pointset)
		tri_group = tde4.createSet(pg)
		tde4.setSetName(pg,tri_group,"Triangulated")		
		for point in pl:
			status = tde4.getPointCalculated3DStatus(pg,point)
			if status == "TRIANGULATED":
				tde4.setPointSet(pg,point,tri_group)

	if tde4.getWidgetValue(req,"passive") == 1:
		l = tde4.getSetList(pg,0)		
		for pointset in l:
			setname = tde4.getSetName(pg,pointset)
			if setname == "Passive":
				tde4.deleteSet(pg,pointset)
		passive_group = tde4.createSet(pg)
		tde4.setSetName(pg,passive_group,"Passive")		
		for point in pl:
			calc_mode = tde4.getPointCalcMode(pg,point)
			if calc_mode == "CALC_PASSIVE":
				tde4.setPointSet(pg,point,passive_group)


req = tde4.createCustomRequester()
#survey toggle widget...
tde4.addToggleWidget(req,"survey","Survey",1)
tde4.setWidgetAttachModes(req,"survey","ATTACH_POSITION","ATTACH_POSITION","ATTACH_WINDOW","ATTACH_AS_IS")
tde4.setWidgetOffsets(req,"survey",20,25,15,0)
#approx survey toggle widget...
tde4.addToggleWidget(req,"approx_survey","Approx Survey",0)
tde4.setWidgetAttachModes(req,"approx_survey","ATTACH_POSITION","ATTACH_POSITION","ATTACH_WINDOW","ATTACH_AS_IS")
tde4.setWidgetOffsets(req,"approx_survey",55,60,15,0)
#survey free toggle widget...
tde4.addToggleWidget(req,"survey_free","Survey Free",0)
tde4.setWidgetAttachModes(req,"survey_free","ATTACH_POSITION","ATTACH_POSITION","ATTACH_WINDOW","ATTACH_AS_IS")
tde4.setWidgetOffsets(req,"survey_free",85,90,15,0)
#lineup toggle widget...
tde4.addToggleWidget(req,"lineup","Lineup",0)
tde4.setWidgetAttachModes(req,"lineup","ATTACH_POSITION","ATTACH_POSITION","ATTACH_WINDOW","ATTACH_AS_IS")
tde4.setWidgetOffsets(req,"lineup",20,25,50,0)
#triangulate toggle widget...
tde4.addToggleWidget(req,"tri","Triangulate",0)
tde4.setWidgetAttachModes(req,"tri","ATTACH_POSITION","ATTACH_POSITION","ATTACH_WINDOW","ATTACH_AS_IS")
tde4.setWidgetOffsets(req,"tri",55,60,50,0)
#passive toggle widget...
tde4.addToggleWidget(req,"passive","Passive",0)
tde4.setWidgetAttachModes(req,"passive","ATTACH_POSITION","ATTACH_POSITION","ATTACH_WINDOW","ATTACH_AS_IS")
tde4.setWidgetOffsets(req,"passive",85,90,50,0)
#add separator...
tde4.addSeparatorWidget(req,"sep1")
#apply button widget...
tde4.addButtonWidget(req,"apply_button","Apply",70,10)
tde4.setWidgetAttachModes(req,"apply_button","ATTACH_POSITION","ATTACH_POSITION","ATTACH_WINDOW","ATTACH_AS_IS")
tde4.setWidgetOffsets(req,"apply_button",75,95,95,0)
#callbacks...
tde4.setWidgetCallbackFunction(req,"apply_button","Apply_callback")
tde4.postCustomRequesterAndContinue(req,window_title,450,125)





