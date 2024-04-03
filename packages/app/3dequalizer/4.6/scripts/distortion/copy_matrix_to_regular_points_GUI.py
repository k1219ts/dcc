#
#
# 3DE4.script.name:    Create Regular Points from Matrix - GUI
#
# 3DE4.script.version:    v1.1
#
# 3DE4.script.gui:    Distortion Edit Controls::Edit
#
# 3DE4.script.comment:    Creates a survey exact point for each valid point of the current matrix.
#
#

pg = tde4.getCurrentPGroup()
c = tde4.getCurrentCamera()
if c!=None:
    req = tde4.createCustomRequester()
    tde4.addTextFieldWidget(req, "width", "width", "2.054535")
    tde4.addTextFieldWidget(req, "height", "height", "2.1")
    ret = tde4.postCustomRequester(req, "Create Regular Points from Matrix", 500, 0, "Ok", "Cancel")
    if ret == 1:
        width = float(tde4.getWidgetValue(req, "width"))
        height = float(tde4.getWidgetValue(req, "height"))
        f = tde4.getCurrentFrame(c)
        d = tde4.getCameraMatrixDimensions(c)
        x = d[0]
        while x<=d[1]:
            y = d[2]
            while y<=d[3]:
                valid = tde4.getCameraMatrixPointValidFlag(c, x, y)
                if valid:
                    p2d = tde4.getCameraMatrixPointPos(c, x, y)
                    p = tde4.createPoint(pg)
                    tde4.setPointSurveyMode(pg, p, "SURVEY_EXACT")
                    tde4.setPointSurveyPosition3D(pg, p, [float(x-1000)*width, float(y-1000)*height, 0.0])
                    tde4.setPointPosition2D(pg, p, c, f, p2d)
                y += 1
            x += 1
