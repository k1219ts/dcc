# 3DE4.script.name:    QCMan
# 3DE4.script.version: v2.1
# 3DE4.script.gui:    Main Window::Dexter
# 3DE4.script.gui.button:	Manual Tracking Controls::QC, align-bottom-right , 70 , 20
# 3DE4.script.gui.button:	Lineup Controls::QC, align-bottom-right, 70, 20
# 3DE4.script.comment: Gui for changing tracker 2D/3D colours that's slightly quicker than conventional ways. By a whole 200ths of a second.
# 3DE4.script.gui.config_menus: true
#
#v1.0 - April 2014  - has 2D and 3D lists. This probably ain't gonna work out well...
#v1.5 - August 2015 - Works when you switch pgroups, which I never noticed before.
#V2.0 - August 2015 - Looks prettier and reduced the amount of code. (159 vs 219 lines)
#v2.1 - Novermber 2016 - Update to allow to be added into a pane (3DE4R5 only)
#
#James Porter

# 0		Red         = 1,0,0
# 1		Green       = 0,1,0
# 2		Blue        = 0,0,1
# 3		Yellow      = 1,1,0
# 4		Black       = 0,0,0
# 5		White       = 1,1,1
# 6		Gray        = 0.5,0.5,0.5
# 7		Purple      = 1,0,1
# 8		Cyan        = 0,1,1
# 9		Light Red   = 1,0.5,0.5
# 10		Light Green = 0.5,1,0.5
# 11		Light Blue  = 0.5,0.5,1

import tde4
import os

tde4.clearConsole()

pg = tde4.getCurrentPGroup()
pointList = tde4.getPointList(pg, 1)
pglist = tde4.getPGroupList()
colourList={0:"RED", 1:"GREEN", 2:"BLUE", 3:"YELLOW", 4:"BLACK", 5:"WHITE", 6:"GRAY", 7:"PURPLE", 8:"CYAN", 9:"L-RED", 10:"L-GREEN", 11:"L-BLUE"}

#Ready for some hacky magic later? This will help colour the populated buttons. Less typing for Jimmy.
r = [1,0,0,1,0,1,0.5,1,0,1,0.5,0.5]
g = [0,1,0,1,0,1,0.5,0,1,0.5,1,0.5]
b = [0,0,1,0,0,1,0.5,1,1,0.5,0.5,1]

def runColourCode(requester,widget,action):
  pg = tde4.getCurrentPGroup()
  for poiints in pg:
    pointList = tde4.getPointList(pg, 0)
    survey     = 0
    approx     = 0
    surveyFree = 0
    calcOff    = 0
    passive    = 0
    for point in pointList:
      if tde4.getPointCalcMode(pg, point) == 'CALC_PASSIVE' or tde4.getPointCalcMode(pg, point) == 'CALC_OFF':
	tde4.setPointColor2D(pg, point, 6)
	tde4.setPointColor3D(pg, point, 6)
	passive = passive + 1
      elif tde4.getPointSurveyMode(pg, point) == 'SURVEY_EXACT':
	tde4.setPointColor2D(pg, point, 0)
	tde4.setPointColor3D(pg, point, 4)
	survey = survey + 1
      elif tde4.getPointSurveyMode(pg, point) == 'SURVEY_APPROX':
	tde4.setPointColor2D(pg, point, 0)
	tde4.setPointColor3D(pg, point, 2)
	approx = approx + 1
      elif tde4.getPointSurveyMode(pg, point) == 'SURVEY_FREE':
	tde4.setPointColor2D(pg, point, 0)
	tde4.setPointColor3D(pg, point, 1)
	surveyFree = surveyFree + 1
      else:
	tde4.setPointColor2D(pg, point, 6)
	tde4.setPointColor3D(pg, point, 6)

  print("3D points:")
  print("survey free             \t = green")
  print("survey approximate \t = blue")
  print("survey exact           \t = black")
  print("disabled/passive   \t = gray")

  print("\nYou've %d exactly surveyed points" %(survey))
  print("You've %d approx surveyed points" %(approx))
  print("You've %d survey free points" %(surveyFree))
  print("You've %d points switched off/passive points" %(passive))

def defaultColours(requester,widget,action):
  tde4.clearConsole()
  pg = tde4.getCurrentPGroup()

  print("Returning all to default colours")
  pointList = tde4.getPointList(pg, 0)
  for point in pointList:
    tde4.setPointColor2D(pg, point, 0)
    tde4.setPointColor3D(pg, point, 1)

def applyColour(requester,widget,action):

    ThreeD_TwoD = tde4.getWidgetValue(requester,"2Dor3D")
    AllOrSelected  = tde4.getWidgetValue(requester,"AllOrSelected")
    pg = tde4.getCurrentPGroup()

    #Convert widget's output name to the colourList index.

    keys=colourList.keys()
    values=colourList.values()
    colourIndex = keys[values.index(widget)]

    for index in colourList:

	if ThreeD_TwoD == 1:
	  trackers = tde4.setPointColor3D
	else:
	  trackers = tde4.setPointColor2D

	if AllOrSelected ==1:
	  #all selected, apply to all 2dtrackers

	  pointList = tde4.getPointList(pg, 0)
	  for point in pointList:
	    trackers(pg, point, colourIndex)

	elif AllOrSelected==0:
	  #Apply to selected tracker.

	  pointList = tde4.getPointList(pg, 1)
	  for point in pointList:
	    trackers(pg, point, colourIndex)

def main():

  #interface time!!
  req= tde4.createCustomRequester()

  tde4.addToggleWidget(req,"AllOrSelected", "All/Selected? (on/off)",1)
  tde4.addToggleWidget(req,"2Dor3D", "3D or 2D? (on/off)",1)
  tde4.setWidgetOffsets(req,"AllOrSelected",90,-2000,-2000,-2000)
  tde4.setWidgetOffsets(req,"2Dor3D",90,-2000,-2000,-2000)
  tde4.addSeparatorWidget(req,'separator')

  i=0
  #Time for some buttons. James ain't typing 12 buttons, colours and callbacks. That's what dictionaries, loops and arrays are for!
  for i in colourList:

    tde4.addButtonWidget(req,colourList[i],colourList[i])
    tde4.setWidgetBGColor(req,colourList[i],r[i],g[i],b[i])
    tde4.setWidgetCallbackFunction(req,colourList[i],"applyColour")
    tde4.setWidgetFGColor(req,colourList[i],0,0,0)
    i=i+1

  tde4.setWidgetFGColor(req,"BLACK",1,1,1)
  #Add colour code / default button
  tde4.addSeparatorWidget(req,'separator')
  tde4.addButtonWidget(req,"runColourCode","Colour code")
  tde4.addButtonWidget(req,"defaultColour","*ALL* default")

  #Start the fans, please!
  tde4.postCustomRequesterAndContinue(req, 'QC',200,500)

  #Call back based on button press? Or all oging to the same
  tde4.setWidgetCallbackFunction(req,"runColourCode","runColourCode")
  tde4.setWidgetCallbackFunction(req,"defaultColour","defaultColours")

main()
