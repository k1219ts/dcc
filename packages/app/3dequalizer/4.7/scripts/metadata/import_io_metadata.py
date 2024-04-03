#
#
# 3DE4.script.name:    Import IO Data...
#
# 3DE4.script.version:    v1.0
#
# 3DE4.script.gui:    Main Window::Dexter::Import Data
#
# 3DE4.script.comment:    Import IO Data From a File.
#
#

# written by Daehwan Jang, 2012.11.06

import string

#
# main script...

cam = tde4.getCurrentCamera()
io_curve = tde4.getCameraStereoInterocularCurve(cam)
zoom_curve = tde4.getCameraZoomCurve(cam)

#
# open requester...
req = tde4.createCustomRequester()
tde4.addFileWidget(req, 'file_browser', 'Filename...', '*')
tde4.addTextFieldWidget(req, 'scale_factor', 'Scale Factor', '10')
ret = tde4.postCustomRequester(req, 'Import IO Data...', 500, 0, 'Ok', 'Cancel')

if ret == 1:
    filename = tde4.getWidgetValue(req, 'file_browser')
    scale_factor = float(tde4.getWidgetValue(req, 'scale_factor'))

    try:
        f = open(filename,'r');
        if not f.closed:
            frame = 1

            if io_curve:
                tde4.deleteAllCurveKeys(io_curve)

            for line in f:
                tde4.createCurveKey(io_curve, [frame, float(line) * scale_factor])
                frame += 1
            tde4.postQuestionRequester('Import IO Data...', 'Done.', 'Ok')
    except:
        tde4.postQuestionRequester('Import IO Data...','Error, could not open file.','Ok')
    finally:
        f.close()
