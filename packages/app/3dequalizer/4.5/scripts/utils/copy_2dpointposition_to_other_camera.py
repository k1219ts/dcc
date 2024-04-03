#
#
# 3DE4.script.name:	Copy 2D Point Position to Other Camera...
#
# 3DE4.script.version:	v1.0
#
# 3DE4.script.gui:	Main Window::Dexter
#
# 3DE4.script.comment:	Copy 2D Point Position to Other Camera.
#
#

#
# main script...

windowtitle = 'Copy 2D Point Position to Other Camera'
cams = tde4.getCameraList()
c = tde4.getCurrentCamera()
pg = tde4.getCurrentPGroup()

if c!=None and pg!=None:
    p = tde4.getContextMenuObject()

    if p!=None:
        pg = tde4.getContextMenuParentObject()
        l = tde4.getPointList(pg, 1)
    else:
        l = tde4.getPointList(pg, 1)

    if len(l)>0:
        req = tde4.createCustomRequester()

        camsname = []
        for ci in cams:
            camsname.append(tde4.getCameraName(ci))

        tde4.addOptionMenuWidget(req, 'targetcam', 'Target Camera', *camsname)
        tde4.addTextFieldWidget(req, 'targetframe', 'Frame to copy', '1')
        ret = tde4.postCustomRequester(req, windowtitle, 600, 0, 'Ok', 'Cancel')
        if ret == 1:
            targetcam = cams[ tde4.getWidgetValue(req, 'targetcam')-1 ]
            targetframe = int( tde4.getWidgetValue(req, 'targetframe') )
            for point in l:
                if tde4.isPointPos2DValid(pg, point, c, 1)!=0:
                    c2d = tde4.getPointPosition2D(pg, point, c, 1)
                    tde4.setPointPosition2D(pg, point, targetcam, targetframe, c2d)
    else:
        tde4.postQuestionRequester(windowtitle, 'There are no selected points.', 'Ok')
    
else:
    tde4.postQuestionRequester(windowtitle, 'There is no current point groupd or camera.', 'Ok')
