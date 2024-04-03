#
#
# 3DE4.script.name: Autocolor points by survey mode...
#
# 3DE4.script.version:		v1.1b1
# 3DE4.script.gui.config_menus: true
#
# 3DE4.script.gui:		Manual Tracking Controls::Edit
# 3DE4.script.gui:		Lineup Controls::Edit
#
# 3DE4.script.comment:
#
# Patcha Saheb(patchasaheb@gmail.com)
# 03 June 2016.

window_title = "Patcha Autocolor points by surveymode v1.1b1"

def Apply_Reset(req,widget,action):
	pg = tde4.getCurrentPGroup()
	cam = tde4.getCurrentCamera()
	frame = tde4.getCurrentFrame(cam)
	pl = tde4.getPointList(pg,0)

	if tde4.getWidgetValue(req,"exact_survey_red") == 1:
		exact_survey_color = 0
	if tde4.getWidgetValue(req,"exact_survey_green") == 1:
		exact_survey_color = 1
	if tde4.getWidgetValue(req,"exact_survey_blue") == 1:
		exact_survey_color = 2
	if tde4.getWidgetValue(req,"exact_survey_yellow") == 1:
		exact_survey_color = 3
	if tde4.getWidgetValue(req,"exact_survey_black") == 1:
		exact_survey_color = 4
	if tde4.getWidgetValue(req,"exact_survey_white") == 1:
		exact_survey_color = 5
	if tde4.getWidgetValue(req,"exact_survey_purple") == 1:
		exact_survey_color = 7

	if tde4.getWidgetValue(req,"approx_survey_red") == 1:
		approx_survey_color = 0
	if tde4.getWidgetValue(req,"approx_survey_green") == 1:
		approx_survey_color = 1	
	if tde4.getWidgetValue(req,"approx_survey_blue") == 1:
		approx_survey_color = 2
	if tde4.getWidgetValue(req,"approx_survey_yellow") == 1:
		approx_survey_color = 3
	if tde4.getWidgetValue(req,"approx_survey_black") == 1:
		approx_survey_color = 4
	if tde4.getWidgetValue(req,"approx_survey_white") == 1:
		approx_survey_color = 5
	if tde4.getWidgetValue(req,"approx_survey_purple") == 1:
		approx_survey_color = 7

	if tde4.getWidgetValue(req,"survey_free_red") == 1:
		survey_free_color = 0	
	if tde4.getWidgetValue(req,"survey_free_green") == 1:
		survey_free_color = 1
	if tde4.getWidgetValue(req,"survey_free_blue") == 1:
		survey_free_color = 2
	if tde4.getWidgetValue(req,"survey_free_yellow") == 1:
		survey_free_color = 3
	if tde4.getWidgetValue(req,"survey_free_black") == 1:
		survey_free_color = 4
	if tde4.getWidgetValue(req,"survey_free_white") == 1:
		survey_free_color = 5
	if tde4.getWidgetValue(req,"survey_free_purple") == 1:
		survey_free_color = 7

	if tde4.getWidgetValue(req,"reset_red") == 1:
		reset_color = 0
	if tde4.getWidgetValue(req,"reset_green") == 1:
		reset_color = 1
	if tde4.getWidgetValue(req,"reset_blue") == 1:
		reset_color = 2
	if tde4.getWidgetValue(req,"reset_yellow") == 1:
		reset_color = 3
	if tde4.getWidgetValue(req,"reset_black") == 1:
		reset_color = 4
	if tde4.getWidgetValue(req,"reset_white") == 1:
		reset_color = 5
	if tde4.getWidgetValue(req,"reset_purple") == 1:
		reset_color = 7		

	if widget == "apply":
		for point in pl:
			survey_mode = tde4.getPointSurveyMode(pg,point)
			if survey_mode == "SURVEY_EXACT":
				tde4.setPointColor3D(pg,point,exact_survey_color)
			if survey_mode == "SURVEY_APPROX":
				tde4.setPointColor3D(pg,point,approx_survey_color)
			if survey_mode == "SURVEY_FREE":
				tde4.setPointColor3D(pg,point,survey_free_color)
	if widget == "reset":
		for point in pl:
			survey_mode = tde4.getPointSurveyMode(pg,point)
			if survey_mode == "SURVEY_EXACT" or survey_mode == "SURVEY_APPROX" or survey_mode == "SURVEY_FREE":
				tde4.setPointColor3D(pg,point,reset_color)				


#GUI...
req = tde4.createCustomRequester()

#add menu bar...
tde4.addMenuBarWidget(req,"menu_bar")
tde4.setWidgetAttachModes(req,"menu_bar","ATTACH_POSITION","ATTACH_POSITION","ATTACH_WINDOW","ATTACH_AS_IS")
tde4.setWidgetOffsets(req,"menu_bar",0,100,1,0)	

#add menu widgets...
tde4.addMenuWidget(req,"exactly_survey_menu","Exact Survey Color","menu_bar",1)
tde4.addMenuWidget(req,"approx_survey_menu","Approx Survey Color","menu_bar",1)
tde4.addMenuWidget(req,"survey_free_menu","Survey free Color","menu_bar",1)
tde4.addMenuWidget(req,"reset_color_menu","Reset Color","menu_bar",1)

#exact survey color menu toggle widgets...
tde4.addMenuToggleWidget(req,"exact_survey_red","Red","exactly_survey_menu",1)
tde4.addMenuToggleWidget(req,"exact_survey_green","Green","exactly_survey_menu",0)
tde4.addMenuToggleWidget(req,"exact_survey_blue","Blue","exactly_survey_menu",0)
tde4.addMenuToggleWidget(req,"exact_survey_yellow","Yellow","exactly_survey_menu",0)
tde4.addMenuToggleWidget(req,"exact_survey_black","Black","exactly_survey_menu",0)
tde4.addMenuToggleWidget(req,"exact_survey_white","White","exactly_survey_menu",0)
tde4.addMenuToggleWidget(req,"exact_survey_purple","Purple","exactly_survey_menu",0)

#approx survey color menu toggle widgets...
tde4.addMenuToggleWidget(req,"approx_survey_red","Red","approx_survey_menu",0)
tde4.addMenuToggleWidget(req,"approx_survey_green","Green","approx_survey_menu",1)
tde4.addMenuToggleWidget(req,"approx_survey_blue","Blue","approx_survey_menu",0)
tde4.addMenuToggleWidget(req,"approx_survey_yellow","Yellow","approx_survey_menu",0)
tde4.addMenuToggleWidget(req,"approx_survey_black","Black","approx_survey_menu",0)
tde4.addMenuToggleWidget(req,"approx_survey_white","White","approx_survey_menu",0)
tde4.addMenuToggleWidget(req,"approx_survey_purple","Purple","approx_survey_menu",0)

#survey free color menu toggle widgets...
tde4.addMenuToggleWidget(req,"survey_free_red","Red","survey_free_menu",0)
tde4.addMenuToggleWidget(req,"survey_free_green","Green","survey_free_menu",0)
tde4.addMenuToggleWidget(req,"survey_free_blue","Blue","survey_free_menu",1)
tde4.addMenuToggleWidget(req,"survey_free_yellow","Yellow","survey_free_menu",0)
tde4.addMenuToggleWidget(req,"survey_free_black","Black","survey_free_menu",0)
tde4.addMenuToggleWidget(req,"survey_free_white","White","survey_free_menu",0)
tde4.addMenuToggleWidget(req,"survey_free_purple","Purple","survey_free_menu",0)

#reset color menu toggle widgets...
tde4.addMenuToggleWidget(req,"reset_red","Red","reset_color_menu",0)
tde4.addMenuToggleWidget(req,"reset_green","Green","reset_color_menu",1)
tde4.addMenuToggleWidget(req,"reset_blue","Blue","reset_color_menu",0)
tde4.addMenuToggleWidget(req,"reset_yellow","Yellow","reset_color_menu",0)
tde4.addMenuToggleWidget(req,"reset_black","Black","reset_color_menu",0)
tde4.addMenuToggleWidget(req,"reset_white","White","reset_color_menu",0)
tde4.addMenuToggleWidget(req,"reset_purple","Purple","reset_color_menu",0)

#apply button widget...
tde4.addButtonWidget(req,"apply","Apply color to points",70,10)
tde4.setWidgetAttachModes(req,"apply","ATTACH_POSITION","ATTACH_POSITION","ATTACH_WINDOW","ATTACH_WINDOW")
tde4.setWidgetOffsets(req,"apply",25,75,60,90)

#reset button widget...
tde4.addButtonWidget(req,"reset","Reset color",70,10)
tde4.setWidgetAttachModes(req,"reset","ATTACH_POSITION","ATTACH_POSITION","ATTACH_WINDOW","ATTACH_WINDOW")
tde4.setWidgetOffsets(req,"reset",25,75,110,40)

#callbacks...
tde4.setWidgetCallbackFunction(req,"apply","Apply_Reset")
tde4.setWidgetCallbackFunction(req,"reset","Apply_Reset")

tde4.postCustomRequesterAndContinue(req,window_title,400,170)
