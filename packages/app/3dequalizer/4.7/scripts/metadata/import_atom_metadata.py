#
#
# 3DE4.script.name:    Import Atom Metadata...
#
# 3DE4.script.version:    v1.0
#
# 3DE4.script.gui:    Main Window::Dexter::Import Data
#
# 3DE4.script.comment:    Import Metadata from a Atom Metadata File.
#
#

# written by Daehwan Jang, 2012.08.24

# 137,IA,1.215,CV,inf,HT,-4.608,TC,20:57:21:04,F,0.000,I,0.000,Z,0.000,FOL,0x0000,IRL,0x0000,ZML,0x0000,FOR,0x0000,IRR,0x0000,ZMR,0x0000,0x16
# IA: Interaxial Distance in cm
# CV: Converfence Distance in unknown
# HT: unknown
# TC: Timecode
# F: Focus in unknown unit
# I: Iris in unknown
# FOL: Focus of Left Lens
# IRL: Iris of Left Lens
# ZML: Zoom of Left Lens
# FOR: Focus of Right Lens
# IRR: Iris of Right Lens
# ZMR: Zoom of Right Lens

import string

#
# main script...

cam = tde4.getCurrentCamera()
io_curve = tde4.getCameraStereoInterocularCurve(cam)
zoom_curve = tde4.getCameraZoomCurve(cam)

#
# open requester...
req = tde4.createCustomRequester()
tde4.addToggleWidget(req, 'io', 'Import IO Curve', 1)
#tde4.addToggleWidget(req, 'zoom', 'Import Zoom Curve', 1)
tde4.addFileWidget(req, 'file_browser', 'Filename...', '*')
ret = tde4.postCustomRequester(req, 'Import Atom Metadata...', 500, 140, 'Ok', 'Cancel')

if ret == 1:
    filename = tde4.getWidgetValue(req, 'file_browser')

    try:
        f = open(filename,'r');
        if not f.closed:
            frame = 1

            if tde4.getWidgetValue(req, 'io') and io_curve:
                tde4.deleteAllCurveKeys(io_curve)
            #if tde4.getWidgetValue(req, 'zoom') and zoom_curve:
            #    tde4.deleteAllCurveKeys(zoom_curve)

            for line in f:
                line = string.split(line, ',')

                IA = line[2]
                CV = line[4]
                HT = line[6]
                TC = line[8]
        
                if (tde4.getWidgetValue(req, 'io')):
                    tde4.createCurveKey(io_curve, [frame, float(IA) / 100.0])
                #if (tde4.getWidgetValue(req, 'zoom')):
                    #tde4.createCurveKey(zoom_curve, [frame, float(Z)])

                frame += 1
            tde4.postQuestionRequester('Import Atom Metadata...', 'Done.', 'Ok')
    except:
        tde4.postQuestionRequester('Import Atom Metadata...','Error, could not open file.','Ok')
    
    finally:
        f.close()
