#
#
# 3DE4.script.name: Copy Survey Data
#
# 3DE4.script.version:  v1.1
#
# 3DE4.script.gui:  Main Window::Dexter
#
# 3DE4.script.comment:  Copy Survey Data From First-Point To Second-Point.
#
#

pg = tde4.getCurrentPGroup()
p = tde4.getPointList(pg, 1)

if pg != None and len(p) == 2:
    point_name = []
    for i in p:
        point_name.append(tde4.getPointName(pg, i))
    #
    # open requester...
    req = tde4.createCustomRequester()
    tde4.addOptionMenuWidget(req, 'from_point', 'Source', *point_name)
    tde4.addOptionMenuWidget(req, 'to_point', 'Target', *point_name)
    tde4.addToggleWidget(req, 'merge_points', 'Merge to Target Point.', 1)
    tde4.addToggleWidget(req, 'survey_approx', 'Set to Approximately Surveyed', 0)
    ret	= tde4.postCustomRequester(req, 'Copy Survey Data...', 600, 0, 'Ok', 'Cancel')

    if ret == 1:
        from_point = int(tde4.getWidgetValue(req, 'from_point')) - 1
        to_point = int(tde4.getWidgetValue(req, 'to_point')) - 1
        merge_points = tde4.getWidgetValue(req, 'merge_points')
        survey_approx = tde4.getWidgetValue(req, 'survey_approx')

        if from_point != to_point:
            if tde4.getPointSurveyMode(pg, p[from_point]) == 'SURVEY_EXACT':
                v = tde4.getPointSurveyPosition3D(pg, p[from_point])

                if survey_approx == 0:
                    tde4.setPointSurveyMode(pg, p[to_point], 'SURVEY_EXACT')
                else:
                    tde4.setPointSurveyMode(pg, p[to_point], 'SURVEY_APPROX')
    
                tde4.setPointSurveyPosition3D(pg, p[to_point], v)
                if merge_points:
                    pname = tde4.getPointName(pg, p[from_point])
                    tde4.deletePoint(pg, p[from_point])
                    tde4.setPointName(pg, p[to_point], pname)
            else:
                tde4.postQuestionRequester('Copy Survey Data...', 'Source Point is not Exactly Surveyed.')
        else:
            tde4.postQuestionRequester('Copy Survey Data...', 'Error, Source Point and Target Point are equal.', 'Ok')
else:
    tde4.postQuestionRequester("Copy Survey Daya...", "There is No Point Group or Select Only Two Points.","Ok")
