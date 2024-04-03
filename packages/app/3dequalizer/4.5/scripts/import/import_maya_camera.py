#
#
# 3DE4.script.name:    Import Maya Camera...
#
# 3DE4.script.version:    v1.1
#
# 3DE4.script.gui:    Main Window::Dexter::Import Data
#
# 3DE4.script.comment:    Imports 3D rot/pos channels from a maya camera.
#
#

#
# import sdv's python vector lib...
#
#
# Change log
# v1.0: Initial script
# v1.1: add FocalLength curve


import sys
instpath    = tde4.get3DEInstallPath()
sys.path.append("%s/sys_data/py_vl_sdv"%instpath)
from vl_sdv import *

#
# main script...
        
cam    = tde4.getCurrentCamera()
pg    = tde4.getCurrentPGroup()
ZoomCurve = tde4.getCameraZoomCurve(cam)

if cam!=None and pg!=None:
    req    = tde4.createCustomRequester()
    tde4.addFileWidget(req,"file_browser","Filename...","*.txt")
    tde4.addToggleWidget(req,"import_pos_toggle","Import Positional Channels",1)
    tde4.addToggleWidget(req,"import_rot_toggle","Import Rotational Channels",1)
    tde4.addTextFieldWidget(req,"frame_offset_field","Frame Offset","0")
    ret    = tde4.postCustomRequester(req,"Import Maya Camera...",700,0,"Ok","Cancel")
    if ret==1:
        frames        = tde4.getCameraNoFrames(cam)
        import_pos    = tde4.getWidgetValue(req,"import_pos_toggle")
        import_rot    = tde4.getWidgetValue(req,"import_rot_toggle")
        frame_offset    = int(tde4.getWidgetValue(req,"frame_offset_field"))
        path        = tde4.getWidgetValue(req,"file_browser")
        if path!=None:
            f    = open(path,"r")
            if not f.closed:
                frame    = 1
                while True:
                    line    = f.readline()
                    line    = line.strip("\n")
                    data    = line.split()
                    if line=="<EOF>":
                        break

                    if len(data)>0:
                        pos_x = float(data[0])            # pos x
                        pos_y = float(data[1])            # pos y
                        pos_z = float(data[2])            # pos z
                        rot_x = float(data[3])            # rot x
                        rot_y = float(data[4])            # rot y
                        rot_z = float(data[5])            # rot z
                        FocalLength = float(data[6])/10.0
                            
                        if import_pos:
                            tde4.setPGroupPosition3D(pg, cam, frame+frame_offset, [pos_x, pos_y, pos_z])
                        if import_rot:
                            rot_x = (rot_x*3.141592654)/180.0
                            rot_y = (rot_y*3.141592654)/180.0
                            rot_z = (rot_z*3.141592654)/180.0

                            r3d = mat3d(rot3d(rot_x, rot_y, rot_z, VL_APPLY_ZXY))
                            r3d0 = [[r3d[0][0], r3d[0][1], r3d[0][2]], [r3d[1][0], r3d[1][1], r3d[1][2]], [r3d[2][0], r3d[2][1], r3d[2][2]]]
                            tde4.setPGroupRotation3D(pg, cam, frame+frame_offset, r3d0)
                        tde4.createCurveKey(ZoomCurve,[frame+frame_offset,float(FocalLength)])
                        frame = frame+1
            f.close
            tde4.setPGroupPostfilterMode(pg,"POSTFILTER_OFF")
            tde4.filterPGroup(pg,cam)
        else:
            tde4.postQuestionRequester("Import Maya Camera...","Error, couldn't open file.","Ok")
    tde4.deleteCustomRequester(req)
else:
    tde4.postQuestionRequester("Import Maya Camera...","Error, there is no current camera or point group.","Ok")
