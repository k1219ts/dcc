#
#
# 3DE4.script.name:  Export Undistorted Images for Asset...
#
# 3DE4.script.version:  v0.5.0
#
# 3DE4.script.gui:  Main Window::DD_MMV
#
# 3DE4.script.comment:  Export undistorted images for Asset.
#
#

# Use 'Weta Nuke Distortion Node' Script written by Wolfgang Niedermeier @ Weta Digital.
# Written by Daehwan Jang(daehwanj@gmail.com)
# Last Updated 2015.06.18

# Change log
# v1.0: Initial script
# v1.9: Redefine burn-in data
# v2.1: ...
# v2.3: Readd burn-in information
# v2.4: change project name    
# v2.4.1:change color space (for IPT) and even resolution
# v2.4.2: image size file_path error fix (hi,lo)

#
# import modules and get environments
import tde4
import string
import re

import os
import sys
import subprocess

import TDE4_common

#
# user defined variables...
VERSION = 'v2.4'
TMP = '/tmp'
#NUKE_VER = os.getenv('NUKE_VER')
NUKE_VER = '7.0v5'
#NUKE_PATH = os.environ['NUKE_PATH']
NUKE_PATH = '/netapp/backstage/dev/apps/nuke/Team_MMV'
NUKE_EXEC_PATH = 'Cannot Find Nuke.'
FONT_PATH = '/netapp/backstage/dev/apps/3de4/py_scripts/Vera.ttf'
OVERSCAN = 1.08

if NUKE_VER:
    NUKE_EXEC_PATH = '/usr/local/Nuke%s/Nuke%s'%(NUKE_VER, NUKE_VER.split("v")[0])
if not NUKE_PATH:
    NUKE_PATH = 'Cannot Find Nuke Environment Path.'

#
# user defined functions...
def _undistort_images_callback(requester, widget, action):
    if widget == 'overscan':
        mode = tde4.getWidgetValue(req, 'overscan')
        if mode == 0:
            OVERSCAN = 1.00
            tde4.setWidgetValue(req, 'os_width', str(seq_width))
            tde4.setWidgetValue(req, 'os_height', str(seq_height))
            tde4.setWidgetValue(req, 'final_size', 'Select Render Size Again.')
        elif mode == 1:
            OVERSCAN = 1.08
            tde4.setWidgetValue(req, 'os_width', str( int(seq_width*OVERSCAN) ) )
            tde4.setWidgetValue(req, 'os_height', str( int(seq_height*OVERSCAN) ) )
            tde4.setWidgetValue(req, 'final_size', 'Select Render Size Again.')

    if widget == 'size':
        mode = tde4.getWidgetValue(req, 'size')
        path = tde4.getWidgetValue(req, 'file_path')
        if mode == 1:
            tde4.setWidgetValue(req, 'file_path', path[:-2] + 'hi')
            tde4.setWidgetValue(req, 'final_size', str( tde4.getWidgetValue(req, 'os_width') )+' x '+str( tde4.getWidgetValue(req, 'os_height') ) )
        elif mode == 2:
            tde4.setWidgetValue(req, 'file_path', path[:-2] + 'lo')
            tde4.setWidgetValue(req, 'final_size', str( int( int(tde4.getWidgetValue(req,'os_width')) *0.5) )+' x '+str( int( int(tde4.getWidgetValue(req,'os_height')) *0.5) ) )
        elif mode == 3:
            tde4.setWidgetValue(req, 'file_path', path[:-2] + 'lo')
            tde4.setWidgetValue(req, 'final_size', str( int( int(tde4.getWidgetValue(req,'os_width')) *0.25) )+' x '+str( int( int(tde4.getWidgetValue(req,'os_height')) *0.25) ) )

def check_file_type(n):
    file_type = os.path.splitext(n)[1].lower()
    opt = ''

    if file_type == '.jpg' or file_type == '.jpeg':
        v = file_type[1:]
        opt = ', _jpeg_quality=1.0'
    else:
        v = file_type[1:]
    return v, opt


def get_dir(dir):
    all_dir = os.listdir(dir)
    result = []

    for i in all_dir:
        if os.path.isdir(os.path.join(dir, i)):
            result.append(i)

    result.sort()
    return result


def find_target_path(dir):
    p = dir.split('/')

    try:
        plate_index = p.index('pmodel')
    except:
        plate_index = None

    if p[1] == 'show' and plate_index != None:
        p2 = p[:plate_index+1]
        p2.extend(['pub', 'imageplane','hi'])
        r = '/'.join(p2)
    else:
        p.append('undist')
        r = '/'.join(p)
    return r


def find_shot_name(dir):
    p = dir.split('/')
    
    if p[0] == '':
        p.pop(0)
    if p[0] == 'show':
        return p[4]
    else:
        return 'unknown'


def make_dir(dir):
    if not os.path.isdir(dir):
        try:
            os.makedirs(dir)
            return 1
        except:
            print 'Making a Folder Failed.'
            return 0


#
# main
#if os.path.isfile("/tmp/.3de_shot_info"):
#    f_info = open("/tmp/.3de_shot_info", "r")
#    j = json.load(f_info)
#    f_info.close()

cam_list = tde4.getCameraList(1)

pgroup_list = tde4.getPGroupList()
pg = ''
for pgl in pgroup_list:
    if tde4.getPGroupType(pgl) == 'CAMERA':
        pg = pgl

proj_path = tde4.getProjectPath()
window_title = 'Export Undistorted Images for Asset %s...'%VERSION

if not proj_path:
    tde4.postQuestionRequester(window_title, 'Save a project first!', 'Ok')
elif not cam_list:
    tde4.postQuestionRequester(window_title, 'Select cameras to undistort first!', 'Ok')
else:
    for cam in cam_list:
        cam_type = tde4.getCameraType(cam) # result: 'SEQUENCE' or 'REFERENCE'
        cam_name = tde4.getCameraName(cam) # result: 'stA'
        cam_path = tde4.getCameraPath(cam) # result: '/stA/stA.####.exr'
        seq_attr = tde4.getCameraSequenceAttr(cam) # result: start, end, step
        seq_width = tde4.getCameraImageWidth(cam) # result: 1936
        seq_height = tde4.getCameraImageHeight(cam) # result: 1288
        seq_aspect = float(seq_width) / float(seq_height) # result: 1.5031055900621118
        seq_path = os.path.split(cam_path)[0] # result: '/stA'
        seq_name = os.path.split(cam_path)[1] # result: 'stA.####.exr'
        
        shot_name = find_shot_name(seq_path)

        tmp = seq_name.split('.') # result: ['stA', '####', 'exr']
        tmp[-1] = 'JPG' # result: ['stA', '####', 'JPG']
        tmp_name = '.'.join(tmp) # result: 'stA.####.JPG'

        frames = tde4.getCameraNoFrames(cam) # result: 95

        #
        # open requester...
        req = tde4.createCustomRequester()
        tde4.addFileWidget(req, 'file_path', 'Location...', '*', find_target_path(seq_path))
        tde4.addTextFieldWidget(req, 'file_name', 'File Name', tmp_name)
        tde4.addTextFieldWidget(req, 'start_frame', 'Start Frame', str(seq_attr[0]))

        tde4.addSeparatorWidget(req, 'sep01')
        tde4.addToggleWidget(req, 'burnin', 'Burn In Info', 0)

        #tde4.addOptionMenuWidget(req, 'overscan', 'Overscan', 'Pre-Defined(x1.08)', 'None', 'User-Defined')
        tde4.addToggleWidget(req, 'overscan', 'Overscan(x1.08)', 0)
        tde4.setWidgetCallbackFunction(req, 'overscan', '_undistort_images_callback')

        tde4.addTextFieldWidget(req, 'os_width', 'Overscan Width', str(seq_width))
        tde4.addTextFieldWidget(req, 'os_height', 'Overscan Height', str(seq_height))
        tde4.setWidgetSensitiveFlag(req, 'os_width', 0)
        tde4.setWidgetSensitiveFlag(req, 'os_height', 0)

        bbox = TDE4_common.bbdld_compute_bounding_box()
        if bbox[0] < 0.0000 or bbox[1] < 0.0000:
            tde4.setWidgetValue(req, 'overscan', '0')
            tde4.setWidgetValue(req, 'os_width', str( int(seq_width*OVERSCAN) ))
            tde4.setWidgetValue(req, 'os_height', str( int(seq_height*OVERSCAN) ))
        else:
            OVERSCAN = 1.00

        tde4.addSeparatorWidget(req, 'sep02')
        tde4.addOptionMenuWidget(req, 'size', 'Render Size', 'Full', 'Half', 'Quarter')
        tde4.setWidgetCallbackFunction(req, 'size', '_undistort_images_callback')
        tde4.addTextFieldWidget(req, 'final_size', '', str( int(seq_width*OVERSCAN) )+' x '+str( int(seq_height*OVERSCAN) ))
        tde4.setWidgetSensitiveFlag(req, 'final_size', 0)

        tde4.addFileWidget(req, 'nuke_exec_path', 'Nuke Exec File...', '*', NUKE_EXEC_PATH)
        tde4.addFileWidget(req, 'nuke_path', 'NUKE_PATH...', '*', NUKE_PATH)

        tde4.addSeparatorWidget(req, 'sep03')
        tde4.addToggleWidget(req, 'only_script', 'Only Script', 1)

        ret = tde4.postCustomRequester(req, window_title, 600, 0, 'Ok', 'Cancel')

        if ret==1:
            # get widget values from gui.
            file_path = tde4.getWidgetValue(req, 'file_path')
            file_name = tde4.getWidgetValue(req, 'file_name')
            overscan = tde4.getWidgetValue(req, 'overscan')
            size = tde4.getWidgetValue(req, 'size')
            burnin = tde4.getWidgetValue(req, 'burnin')
            start_frame = int(tde4.getWidgetValue(req, 'start_frame'))
            nuke_exec_path = tde4.getWidgetValue(req, 'nuke_exec_path')
            NUKE_PATH = tde4.getWidgetValue(req, 'nuke_path')
            only_script = tde4.getWidgetValue(req, 'only_script')

            if tde4.getWidgetValue(req, 'final_size') == 'Select Render Size Again.':
                tde4.postQuestionRequester(window_title, 'Select Render Size Again.', 'Ok')
                break
            if start_frame < 0:
                start_frame = 0
                tde4.postQuestionRequester(window_title, 'Start Frame Should not be Negative! It will be 0.', 'Ok')

            # set a new file name.
            new_file_path = os.path.join(file_path, file_name)

            # get a file type.
            file_type = check_file_type(file_name)

            # get a camera, lens data.
            lens = tde4.getCameraLens(cam)
            model = tde4.getLensLDModel(lens)
            nfr = tde4.getCameraNoFrames(cam)
            num_frames     = tde4.getCameraNoFrames(cam)
            w_fb_cm = tde4.getLensFBackWidth(lens)
            h_fb_cm = tde4.getLensFBackHeight(lens)
            lco_x_cm = tde4.getLensLensCenterX(lens)
            lco_y_cm = tde4.getLensLensCenterY(lens)
            pxa = tde4.getLensPixelAspect(lens)
            # xa,xb,ya,yb in unit coordinates, in this order.
            fov = tde4.getCameraFOV(cam)

            # set undistorted image to camera proxy footage.
            tde4.setCameraProxyFootage(cam, 3)
            tde4.setCameraSequenceAttr(cam, seq_attr[0], seq_attr[1], seq_attr[2])
            tde4.setCameraPath(cam, new_file_path)
            tde4.setCameraProxyFootage(cam, 0)

            # write a nuke python script per frame.
            nuke_pyscript_name = TDE4_common.valid_name(cam_name)+'.py'
            tde4.postProgressRequesterAndContinue(window_title, 'Undistorting, Frame 1...', frames, 'Stop')
            for frame in range(1, nfr+1):
                f = open(os.path.join(TMP, nuke_pyscript_name), 'w')
                try:
                    # write some comments.
                    f.write('# Nuke Python Script Written by Undistort Image %s Script\n\n'%VERSION)
                    
                    # write a read node.
                    f.write('# Create a read node for original images.\n')
                    f.write('seq = nuke.nodes.Read(file=\'%s\', format=\'%d %d 0 0 %d %d 1\', first=%d, last=%d, colorspace=0, on_error=1)\n\n'%(cam_path,
                             seq_width, seq_height, seq_width, seq_height, frame+(seq_attr[0]-1), frame+(seq_attr[0]-1)))

                    # write a blackoutside node.
                    f.write('# Create a blackoutside node to add 1 black pixel at each boundary edge of image.\n')
                    f.write('addblack = nuke.nodes.BlackOutside()\n')
                    f.write('addblack.setInput(0, seq)\n\n')
                        
                    # write a weta_lensDistortion node.
                    focal = tde4.getCameraFocalLength(cam, frame)
                    focus = tde4.getCameraFocus(cam, frame)
                    f.write('# Created by 3DEqualizer4 using Export Nuke Distortion Nodes export script\n')
                    f.write('ld = nuke.nodes.%s()\n'%("LD" + TDE4_common.nukify_name(model)))
                    f.write('ld[\'direction\'].setValue(\'undistort\')\n')
                    f.write('ld[\'tde4_focal_length_cm\'].setValue(%f)\n'%focal)
                    f.write('ld[\'tde4_custom_focus_distance_cm\'].setValue(%f)\n'%focus)
                    f.write('ld[\'tde4_filmback_width_cm\'].setValue(%f)\n'%w_fb_cm)
                    f.write('ld[\'tde4_filmback_height_cm\'].setValue(%f)\n'%h_fb_cm)
                    f.write('ld[\'tde4_lens_center_offset_x_cm\'].setValue(%f)\n'%lco_x_cm)
                    f.write('ld[\'tde4_lens_center_offset_y_cm\'].setValue(%f)\n'%lco_y_cm)
                    f.write('ld[\'tde4_pixel_aspect\'].setValue(%f)\n'%pxa)
                    f.write('ld[\'field_of_view_xa_unit\'].setValue(%f)\n'%fov[0])
                    f.write('ld[\'field_of_view_ya_unit\'].setValue(%f)\n'%fov[2])
                    f.write('ld[\'field_of_view_xb_unit\'].setValue(%f)\n'%fov[1])
                    f.write('ld[\'field_of_view_yb_unit\'].setValue(%f)\n'%fov[3])
                    for para in TDE4_common.getLDmodelParameterList(model):
                        f.write('ld[\'%s\'].setValue(%.7f)\n'%(TDE4_common.nukify_name(para), tde4.getLensLDAdjustableParameter(lens, para, focal, focus)))
                    f.write('ld[\'name\'].setValue(\'LD_3DE4_%s\')\n'%TDE4_common.decode_entities(tde4.getCameraName(cam)))

                    f.write('ld.setInput(0, addblack)\n\n')

                    # get bounding box for overscan from weta_lensDistortion node.
                    f.write('# Get bounding box from weta_lensDistortion node.\n')
                    if overscan == 1: # if pre-defined selected
                        f.write('# Pre-defined selected.\n')
                        os_width = int(tde4.getWidgetValue(req, 'os_width'))
                        os_height = int(tde4.getWidgetValue(req, 'os_height'))
                        f.write('bbw = %d\n'%os_width)
                        f.write('bbh = %d\n'%os_height)
                    elif overscan == 0: # if no overscan selected
                        f.write('# No overscan selected.\n')
                        f.write('bbw = %d\n'%seq_width)
                        f.write('bbh = %d\n'%seq_height)
                    f.write('\n')

                    # write a reformat node.
                    f.write('# Create a reformat node for overscan resolution.\n')
                    f.write('new_res = nuke.nodes.Reformat(type=0, format=\'%d %d 0 0 %d %d 1\'%(bbw, bbh, bbw, bbh), resize=0)\n')
                    f.write('new_res.setInput(0, ld)\n\n')

                    # write a text node for burning information.
                    if burnin == 1:
                        f.write('# Create a text node for filename.\n')
                        f.write('text_filename = nuke.nodes.Text(cliptype=\'no clip\', message=\'[python {nuke.thisNode().metadata()["input/filename"]}]\', font=\'%s\', size=\'{floor(width*0.01)}\', xjustify=\'left\', yjustify=\'bottom\', box=\'0 0 width height\')\n'%FONT_PATH)
                        f.write('text_filename.setInput(0, new_res)\n\n')
                        f.write('# Create a text node for timecode.\n')
                        f.write('text_timecode = nuke.nodes.Text(cliptype=\'no clip\', message=\'[timecode]\', font=\'%s\', size=\'{floor(width*0.01)}\', xjustify=\'right\', yjustify=\'bottom\', box=\'0 0 width height\')\n'%FONT_PATH)
                        f.write('text_timecode.setInput(0, text_filename)\n\n')
                    else:
                        f.write('text_timecode = nuke.nodes.NoOp(name=\'No_Burnin\')\n')
                        f.write('text_timecode.setInput(0, new_res)\n\n')

                    # write a reformat node for half size.
                    f.write('# Create a reformat node for resize.\n')
                    if size == 2:
                        f.write('resize = nuke.nodes.Reformat(type=2, scale=0.5, resize=1, black_outside=0)\n')
                    elif size == 3:
                        f.write('resize = nuke.nodes.Reformat(type=2, scale=0.25, resize=1, black_outside=0)\n')
                    else:
                        f.write('resize = nuke.nodes.NoOp(name=\'NO_HALF_SIZE\')\n')
                    f.write('resize.setInput(0, text_timecode)\n\n')

                    # write a write node.
                    f.write('# Create a write node.\n')
                    f.write('write = nuke.nodes.Write(file=\'%s\', file_type=\'%s\'%s)\n'%(new_file_path, file_type[0], file_type[1]))
                    f.write('write.setInput(0, resize)\n\n')

                    # write a excute command.
                    f.write('# Execute undistort python script.\n')
                    if only_script == 0:
                        f.write('nuke.execute(write, start=%d, end=%d, incr=%d)'%(frame+start_frame-1, frame+start_frame-1, seq_attr[2]))
                    else:
                        pass

                    # finally, make a folder to write undistorted images, if don't exist.
                    make_dir(file_path)

                finally:
                    f.close()
                    cont = tde4.updateProgressRequester(frame, "Undistorting, Frame %d..."%frame)
                    if not cont: break
                    if only_script == 0:
                        nuke_cmd = '%s -t %s'%(nuke_exec_path, os.path.join(TMP, nuke_pyscript_name))
                        run = subprocess.Popen(nuke_cmd, shell=True, env={'NUKE_PATH':NUKE_PATH})
                        run.wait()
                    else:
                        pass
            tde4.unpostProgressRequester()

