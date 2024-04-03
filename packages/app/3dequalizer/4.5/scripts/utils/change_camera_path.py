#
#
# 3DE4.script.name:	Change Camera Path...
#
# 3DE4.script.version:	v1.0
#
# 3DE4.script.gui:	Main Window::Dexter
#
# 3DE4.script.comment:	Change Current Camera Path To Specific Path.
#
#

# import modules
import os

#
# main
cam_list = tde4.getCameraList(1)

#
# open requester...
req = tde4.createCustomRequester()
tde4.addFileWidget(req, 'user_path', 'Select Path...', '*', 'Select Only Path, Not Any File.')
ret	= tde4.postCustomRequester(req, 'Change Camera Path...', 600, 0, 'Ok', 'Cancel')

if ret == 1:
    user_path = tde4.getWidgetValue(req, 'user_path')
    for cam in cam_list:
        old_path = os.path.split(tde4.getCameraPath(cam))
        new_path = os.path.join(user_path, old_path[1])
        new_path = new_path.replace('\\', '/')

        tde4.setCameraPath(cam, new_path)
