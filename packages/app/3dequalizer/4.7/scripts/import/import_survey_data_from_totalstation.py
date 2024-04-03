##################################################
# Description: Import a Survey data from Total Station.
# Written by Daehwan Jang (daehwanj@gmail.com)
# Last updated: 2012.03.19(v1.0)
##################################################
#
#
# 3DE4.script.name:    Import Survey Data From Total Station...
#
# 3DE4.script.version:    v1.0
#
# 3DE4.script.gui:    Main Window::Dexter::Import Data
#
# 3DE4.script.comment:    Import a Survey data from Total Station.
#
#

#
# main script...

pg    = tde4.getCurrentPGroup()
if pg!=None:
    req    = tde4.createCustomRequester()
    tde4.addFileWidget(req,"file_browser","Filename...","*.txt")
    tde4.addOptionMenuWidget(req,"mode_menu","Mode","Always Create New Points","Add Survey to Existing Points of Same Name")
    tde4.addTextFieldWidget(req, "text1", "CAUTION", "Total Station has Meter Unit as Default Unit.")
    tde4.addTextFieldWidget(req, "text2", "", "So if you want a Centimeter Unit, select Scale Factor to 1:100.")
    tde4.addOptionMenuWidget(req,"scale_menu","Import Scale Factor","1:1","1:10","1:100")
    
    ret    = tde4.postCustomRequester(req,"Import Survey Data From Total Station...",600,200,"Ok","Cancel")
    if ret==1:
        create_new = tde4.getWidgetValue(req,"mode_menu")
        scale = tde4.getWidgetValue(req,"scale_menu")
        scalef = 1
        if scale == "1":
            scalef = 1
        elif scale == "2":
            scalef = 10
        else:
            scalef = 100

        path    = tde4.getWidgetValue(req,"file_browser")
        if path!=None:
            #
            # main block...
            
            f    = open(path,"r")
            if not f.closed:
                string    = f.readline()
                a    = string.split(",")
                while len(a)==4:
                    if create_new==1:
                        p    = tde4.createPoint(pg)
                        tde4.setPointName(pg,p,a[0])
                    else:
                        p    = tde4.findPointByName(pg,a[0])
                        if p==None:
                            p    = tde4.createPoint(pg)
                            tde4.setPointName(pg,p,a[0])
                    tde4.setPointSurveyMode(pg,p,"SURVEY_EXACT")
                    tde4.setPointSurveyPosition3D(pg, p, [float(a[1])*(scalef), float(a[3])*(scalef), float(a[2])*-(scalef)])
                    
                    string    = f.readline()
                    a    = string.split(",")
                
                f.close()
            else:
                tde4.postQuestionRequester("Import Survey Data From Total Station...","Error, couldn't open file.","Ok")
            
            # end main block...
            #
else:
    tde4.postQuestionRequester("Import Survey Data From Total Station...","There is no current Point Group.","Ok")


