import os, getpass, json
import re, string

import tde4
import TDE4_common

import DD_common
import dxUIcommon
from imp import reload
reload(dxUIcommon)
from dxpublish import insertDB

# rez3deRoot = os.environ['REZ_3DEQUALIZER_ROOT']
rez3deRoot = '/backstage/dcc/packages/app/3dequalizer/4.7'
FONT_PATH = os.path.join(rez3deRoot, 'scripts/distortion/Vera.ttf')

knobChangeStr = """
import re
if nuke.thisKnob().name() == 'inputChange':
    rn = getTop(nuke.thisNode())
    rf = rn['file'].value()
    if rf.startswith('/netapp/dexter/show'):
        rf = rf.replace('/netapp/dexter/show', '/show')
    ele = re.split('[.]|_|/', rf)
    print ele
    if (ele[1] == 'show') & (ele[3] == 'shot') & (ele[7] == 'plates'):
        show = ele[2]
        seq = ele[4]
        shot = '_'.join([ele[5], ele[6]])
        plate = ele[8]

        nuke.thisNode()['show'].setValue(show)
        nuke.thisNode()['seq'].setValue(seq)
        nuke.thisNode()['shot'].setValue(shot)
        nuke.thisNode()['plate'].setValue(plate)
"""

def check_file_type(n):
    file_type = os.path.splitext(n)[1].lower()
    opt = ''

    if file_type == '.jpg' or file_type == '.jpeg':
        v = file_type[1:]
        opt = ', _jpeg_quality=1.0'
    else:
        v = file_type[1:]
    return v, opt

def createNukePyForUndistortImage(req, cam, fileName, data, frame, sendTractor=False, pmodel=False, onlyScript=False):
    dstDir = '/tmp'
    if sendTractor:
        dstDir = os.path.dirname(tde4.getProjectPath()).replace('3de', 'imageplane')
        if not os.path.exists(dstDir):
            os.mkdir(dstDir)
    nkFile = os.path.join(dstDir, fileName)
    f = open(nkFile, 'w')

    # try:
    # write some comments.
    f.write('# Nuke Python Script Written by Undistort Image Script\n\n')

    configFile = os.path.join('/show', data['show'], '_config', 'Project.config')
    # print('configFile:', configFile)
    with open(configFile, 'r') as config:
        configData = json.load(config)

    if 'ACES' in configData['colorSpace']:
        f.write('nuke.root().knob(\'colorManagement\').setValue(\'OCIO\')\n')
    else:
        f.write('nuke.root().knob(\'colorManagement\').setValue(\'Nuke\')\n')

    # write a read node.
    f.write('# Create a read node for original images.\n')
    f.write('seq = nuke.nodes.Read(file=\'%s\', format=\'%d %d 0 0 %d %d 1\', first=%d, last=%d, colorspace=\'%s\')\n\n' \
    %(data['seqFile'], data['seqWidth'], data['seqHeight'], data['seqWidth'], data['seqHeight'], frame+(data['seqAttr'][0]-1), frame+(data['seqAttr'][0]-1), data['seqColorspaceR']))

    # write a blackoutside node.
    f.write('# Create a blackoutside node to add 1 black pixel at each boundary edge of image.\n')
    f.write('addblack = nuke.nodes.BlackOutside()\n')
    f.write('addblack.setInput(0, seq)\n\n')

    # write a weta_lensDistortion node.
    focal = tde4.getCameraFocalLength(cam, frame)
    focus = tde4.getCameraFocus(cam, frame)
    f.write('# Created by 3DEqualizer4 using Export Nuke Distortion Nodes export script\n')
    f.write('ld = nuke.nodes.%s()\n' % ("LD" + TDE4_common.nukify_name(data['lensLDModel'])))
    f.write('ld[\'direction\'].setValue(\'undistort\')\n')
    f.write('ld[\'tde4_focal_length_cm\'].setValue(%f)\n' % focal)
    f.write('ld[\'tde4_custom_focus_distance_cm\'].setValue(%f)\n' % focus)
    f.write('ld[\'tde4_filmback_width_cm\'].setValue(%f)\n' % data['filmbackWidth'])
    f.write('ld[\'tde4_filmback_height_cm\'].setValue(%f)\n' % data['filmbackHeight'])
    f.write('ld[\'tde4_lens_center_offset_x_cm\'].setValue(%f)\n' % data['lensOffsetX'])
    f.write('ld[\'tde4_lens_center_offset_y_cm\'].setValue(%f)\n' % data['lensOffsetY'])
    f.write('ld[\'tde4_pixel_aspect\'].setValue(%f)\n' % data['pixelAspect'])
    f.write('ld[\'field_of_view_xa_unit\'].setValue(%f)\n' % data['fov'][0])
    f.write('ld[\'field_of_view_ya_unit\'].setValue(%f)\n' % data['fov'][2])
    f.write('ld[\'field_of_view_xb_unit\'].setValue(%f)\n' % data['fov'][1])
    f.write('ld[\'field_of_view_yb_unit\'].setValue(%f)\n' % data['fov'][3])
    for para in TDE4_common.getLDmodelParameterList(data['lensLDModel']):
        f.write('ld[\'%s\'].setValue(%.7f)\n' % (TDE4_common.nukify_name(para), tde4.getLensLDAdjustableParameter(data['lens'], para, focal, focus)))
    f.write('ld[\'name\'].setValue(\'LD_3DE4_%s\')\n' % TDE4_common.decode_entities(tde4.getCameraName(cam)))

    f.write('ld.setInput(0, addblack)\n\n')

    # get bounding box for overscan from weta_lensDistortion node.
    f.write('# Get bounding box from weta_lensDistortion node.\n')
    if data['overscan'] == 1: # if pre-defined selected
        f.write('# Pre-defined selected.\n')
        os_width = int(tde4.getWidgetValue(req, 'os_width'))
        os_height = int(tde4.getWidgetValue(req, 'os_height'))
        f.write('bbw = %d\n' % data['overscanWidth'])
        f.write('bbh = %d\n' % data['overscanHeight'])
    elif data['overscan'] == 0: # if no overscan selected
        f.write('# No overscan selected.\n')
        f.write('bbw = %d\n' % data['seqWidth'])
        f.write('bbh = %d\n' % data['seqHeight'])
    f.write('\n')

    # write a reformat node.
    f.write('# Create a reformat node for overscan resolution.\n')
    f.write('new_res = nuke.nodes.Reformat(type=0, format=\'%d %d 0 0 %d %d 1\'%(bbw, bbh, bbw, bbh), resize=0)\n')
    f.write('new_res.setInput(0, ld)\n\n')

    # write a text node for burning information.
    if data['burnIn'] == 1:
        lensName = '.'.join(tde4.getLensName(data['lens']).split('.')[:-4])
        if tde4.getCameraZoomingFlag(data['cameraId']):
            lensInfo = '%s(%.2fmm)' % (lensName, round(focal*10, 2))
        else:
            lensInfo = lensName
        f.write('# Create a text node for lensInfo.\n')
        f.write('text_lens = nuke.nodes.Text(cliptype=\'no clip\', message=\'%s\', font=\'%s\', size=\'{floor(width*0.01)}\', xjustify=\'center\', yjustify=\'bottom\', box=\'0 0 width height\')\n' % (lensInfo, FONT_PATH))
        f.write('text_lens.setInput(0, new_res)\n\n')

        f.write('# Create a text node for filename.\n')
        f.write('text_filename = nuke.nodes.Text(cliptype=\'no clip\', message=\' [python {os.path.basename(nuke.thisNode().metadata()["input/filename"])}]\', font=\'%s\', size=\'{floor(width*0.01)}\', xjustify=\'left\', yjustify=\'bottom\', box=\'0 0 width height\')\n'%FONT_PATH)
        f.write('text_filename.setInput(0, text_lens)\n\n')

        f.write('# Create a text node for timecode.\n')
        f.write('text_timecode = nuke.nodes.Text(cliptype=\'no clip\', message=\'  [timecode]\', font=\'%s\', size=\'{floor(width*0.01)}\', xjustify=\'right\', yjustify=\'bottom\', box=\'0 0 width height\')\n'%FONT_PATH)
        f.write('text_timecode.setInput(0, text_filename)\n\n')

    else:
        f.write('text_timecode = nuke.nodes.NoOp(name=\'No_Burnin\')\n')
        f.write('text_timecode.setInput(0, new_res)\n\n')

    # write a reformat node for half size.
    f.write('# Create a reformat node for resize.\n')
    if data['jpgSize'] == 2:
        f.write('resize = nuke.nodes.Reformat(type=2, scale=0.5, resize=1, black_outside=0)\n')
    elif data['jpgSize'] == 3:
        f.write('resize = nuke.nodes.Reformat(type=2, scale=0.25, resize=1, black_outside=0)\n')
    else:
        f.write('resize = nuke.nodes.NoOp(name=\'NO_HALF_SIZE\')\n')
    f.write('resize.setInput(0, text_timecode)\n\n')

    # write a write node.
    f.write('# Create a write node.\n')
    fileType = check_file_type(data['fileName'])
    f.write('write = nuke.nodes.Write(file=\'%s\', file_type=\'%s\'%s, colorspace=\'%s\')\n' % (data['jpgFile'], fileType[0], fileType[1], data['seqColorspaceW']))
    f.write('write.setInput(0, resize)\n\n')

    # rename write -> write_ImagePlane
    f.write('write.setName("Write_ImagePlane1")\n')

    # add metadata knob to write node
    f.write('showKnob = nuke.String_Knob("show")\n')
    f.write('seqKnob = nuke.String_Knob("seq")\n')
    f.write('shotKnob = nuke.String_Knob("shot")\n')
    f.write('plateKnob = nuke.String_Knob("plate")\n')
    f.write('hiloKnob = nuke.String_Knob("hilo")\n')
    f.write('write.addKnob(showKnob)\n')
    f.write('write.addKnob(seqKnob)\n')
    f.write('write.addKnob(shotKnob)\n')
    f.write('write.addKnob(plateKnob)\n')
    f.write('write.addKnob(hiloKnob)\n')

    # knobChanged Setting
    f.write('write["knobChanged"].setValue("""%s""")\n'% knobChangeStr)

    # set metadata values
    f.write('showKnob.setValue("%s")\n' % data['show'])
    f.write('seqKnob.setValue("%s")\n' % data['seq'])
    f.write('shotKnob.setValue("%s")\n' % data['shot'])
    f.write('plateKnob.setValue("%s")\n' % data['plateType'])
    if data['jpgSize'] == 1:
        f.write('hiloKnob.setValue("hi")\n')
    else:
        f.write('hiloKnob.setValue("lo")\n')

    # write a excute command.
    f.write('# Execute undistort python script.\n')
    if not onlyScript:
        f.write('nuke.execute(write, start=%d, end=%d, incr=%d)'%(frame+data['startFrame']-1, frame+data['startFrame']-1, data['seqAttr'][2]))

    # finally, make a folder to write undistorted images, if don't exist.
    DD_common.makeDir(data['filePath'])

    # finally:
    f.close()

    return nkFile


class NukeNode:
    ''' Class to represent a Nuke node, and the knob names & values that
    belong to it. The idea is to make writing nodes to a script easier. '''
    def __init__(self, node_name):
        self.knob_name_value_tuples = dict()
        self.ordered_knob_names = []
        self.node_name = node_name

    def add_knob(self, name, value, index=0):
        if not name in self.knob_name_value_tuples:
            self.ordered_knob_names.append(name)
        self.knob_name_value_tuples.setdefault(name,[]).append([index, value])

    def add_duplicate_knob(self, name, value):
        self.add_knob(name, value, -1)

    def write_out(self, file_handle):
    # write out the data from our dict
        file_handle.write('%s {\n' % self.node_name)
        for para_name in self.ordered_knob_names:
            para_vals = self.knob_name_value_tuples[para_name]
            # test how many indices we have, if we just have one, print that
            # and move on
            if len(para_vals) == 1:
                file_handle.write(' %s %s\n' % (para_name, para_vals[0][1]))
            else:
                # otherwise, we have a bit more work to do.

                # sort our para values by their knob indices, this is to allow multiple value knobs
                # like scale or position work correctly
                sorted_para_vals = sorted(para_vals, key=lambda para_val: para_val[0])

                # test if all of our knob indices are -1, if so print the param name for every line, e.g. addUserKnob
                if all(i[0] == -1 for i in sorted_para_vals):
                    for para_val in sorted_para_vals:
                        file_handle.write(' %s %s\n' % (para_name, para_val[1]))
                else:
                    # otherwise iterate over our sorted knob values and printing them to the script
                    # and write out the name if we haven't done so already
                    file_handle.write(' %s {' % para_name)
                    for para_val in sorted_para_vals:
                        file_handle.write('%s ' % para_val[1])
                    file_handle.write('}\n')
        file_handle.write('}\n')


class createNukeLD:
    def __init__(self, requester):
        self.req = requester
        self.windowTitle = ''

        if 'show' in os.environ:
            self.show = os.environ['show']
            self.seq = os.environ['seq']
            self.shot = os.environ['shot']
            self.plateType = os.environ['platetype']
        else:
            tde4.postQuestionRequester(self.windowTitle, 'Start working with \'dxOpen Project\'', 'Ok')
            # raise Exception("Parsing ENVKEY failed! It goes something wrong.")

        self.camList = tde4.getCameraList(1)
        if not self.camList:
            tde4.postQuestionRequester(self.windowTitle, 'Only selected cameras will be exported.', 'Ok')
            # raise Exception('     Only selected cameras will be exported     ')

        self.camNames = []
        for c in self.camList:
            self.camNames.append(tde4.getCameraName(c))

        # self.pubPath = os.path.join('/show', self.show, 'shot', self.seq, self.shot, 'matchmove/pub/nuke')
        # impPath = os.path.join('/show', self.show, '_2d', 'shot', self.seq, self.shot, 'iamgeplane')

        self.seqPath = os.path.split(tde4.getCameraPath(self.camList[0]))[0]
        self.pubPath = os.path.join(DD_common.find_target_path(self.seqPath), 'node')

        nuke_file = '%s_%s_LD'%(self.shot, self.plateType)
        pub_ver = DD_common.get_final_ver(self.pubPath, nuke_file+'_v', "*.nk")
        nuke_file += '_v%s.nk'%(pub_ver)
        self.nuke_path = os.path.join(self.pubPath, nuke_file)
        # self.nuke11_path = self.nuke_path.replace('_LD_v','_LD_nuke11_v')

    def _export_LD_callback(self, requester, widget, action):
        if widget == "stereo":
            mode = tde4.getWidgetValue(requester, "stereo")
            if mode == 0:
                tde4.setWidgetSensitiveFlag(requester, "right_camera", 0)
                tde4.setWidgetLabel(requester, "left_camera", "Main Camera")
                tde4.modifyOptionMenuWidget(requester, "right_camera", "", "")
            if mode == 1:
                tde4.setWidgetSensitiveFlag(requester, "right_camera", 1)
                tde4.setWidgetLabel(requester, "left_camera", "Left Camera")
                tde4.modifyOptionMenuWidget(requester, "right_camera", "Right Camera", *self.camNames)

    def sanitise_nuke_names(self, s):
        if s == "":
            return "_"
        if s[0] in "0123456789":
            t = "_"
        else:
            t = ""
        t += string.join(re.sub("[+,:; _-]+","_",s.strip()).split())
        return t

    def nukify_name(self, s):
        # ensure param name has valid characters
        t = self.sanitise_nuke_names(s)

        # map TDE4 param name to Nuke's LD name
        t = self.map_TDE4_paramname_to_nuke(t)[0]

        return t

    def nuke_para_index(self, s):

        # ensure param name has valid characters
        t = self.sanitise_nuke_names(s)

        # map TDE4 param name to Nuke's LD index
        t = self.map_TDE4_paramname_to_nuke(t)[1]

        return t

    def map_TDE4_paramname_to_nuke(self, tde4param):
        # set up the dict for mapping
        _knob_mapping = {
            "Anamorphic_Squeeze":         ["anamorphicSqueeze", 0], # Classic
            "Distortion":                 ["distortionNumerator0", 0], # Classic
            "Quartic_Distortion":         ["distortionNumerator1", 0], # Classic
            "Curvature_X":                ["distortionNumeratorX00", 0], # Classic
            "Curvature_Y":                ["distortionNumeratorY00", 0], # Classic
            "Distortion_Degree_2":        ["distortionNumerator0", 0], # Radial Standard Degree 4, Radial Fisheye Degree 8
            "U_Degree_2":                 ["distortionNumeratorT0", 0], # Radial Standard Degree 4
            "V_Degree_2":                 ["distortionNumeratorU0", 0], # Radial Standard Degree 4
            "Quartic_Distortion_Degree_4":["distortionNumerator1", 0], # Radial Standard Degree 4, Radial Fisheye Degree 8
            "U_Degree_4":                 ["distortionNumeratorT1", 0], # Radial Standard Degree 4
            "V_Degree_4":                 ["distortionNumeratorU1", 0], # Radial Standard Degree 4
            "Phi_Cylindric_Direction":    ["beamSplitterDirection", 0], # Radial Standard Degree 4
            "B_Cylindric_Bending":        ["beamSplitterBending", 0], # Radial Standard Degree 4
            "Degree_6":                   ["distortionNumerator2", 0], # Radial Fisheye Degree 8
            "Degree_8":                   ["distortionNumerator3", 0], # Radial Fisheye Degree 8
            "Cx02_Degree_2":              ["distortionNumeratorX00", 0], # Anamorphic Standard Degree 4
            "Cx22_Degree_2":              ["distortionNumeratorX10", 0], # Anamorphic Standard Degree 4
            "Cx04_Degree_4":              ["distortionNumeratorX01", 0], # Anamorphic Standard Degree 4
            "Cx24_Degree_4":              ["distortionNumeratorX11", 0], # Anamorphic Standard Degree 4
            "Cx44_Degree_4":              ["distortionNumeratorX20", 0], # Anamorphic Standard Degree 4
            "Cy02_Degree_2":              ["distortionNumeratorY00", 0], # Anamorphic Standard Degree 4
            "Cy22_Degree_2":              ["distortionNumeratorY10", 0], # Anamorphic Standard Degree 4
            "Cy04_Degree_4":              ["distortionNumeratorY01", 0], # Anamorphic Standard Degree 4
            "Cy24_Degree_4":              ["distortionNumeratorY11", 0], # Anamorphic Standard Degree 4
            "Cy44_Degree_4":              ["distortionNumeratorY20", 0], # Anamorphic Standard Degree 4
            "Lens_Rotation":              ["anamorphicTwist", 0], # Anamorphic Standard Degree 4
            "Squeeze_X":                  ["anamorphicScale", 0], # Anamorphic Standard Degree 4
            "Squeeze_Y":                  ["anamorphicScale", 1], # Anamorphic Standard Degree 4
            "Cx06_Degree_6":              ["distortionNumeratorX02", 0], # Anamorphic Standard Degree 4
            "Cx26_Degree_6":              ["distortionNumeratorX12", 0], # Anamorphic Standard Degree 4
            "Cx46_Degree_6":              ["distortionNumeratorX21", 0], # Anamorphic Standard Degree 4
            "Cx66_Degree_6":              ["distortionNumeratorX30", 0], # Anamorphic Standard Degree 4
            "Cy06_Degree_6":              ["distortionNumeratorY02", 0], # Anamorphic Standard Degree 4
            "Cy26_Degree_6":              ["distortionNumeratorY12", 0], # Anamorphic Standard Degree 4
            "Cy46_Degree_6":              ["distortionNumeratorY21", 0], # Anamorphic Standard Degree 4
            "Cy66_Degree_6":              ["distortionNumeratorY30", 0], # Anamorphic Standard Degree 4
            "Rescale":                    ["Rescale", 0], # Anamorphic Rescaled Degree 4
        }

        if not tde4param in _knob_mapping:
            raise Exception('     TDE4 knob name "%s" not mapped to Nuke LD node knobs.     ' % tde4param )
        else:
            return _knob_mapping[tde4param]

    # Nuke interprets entities like "<" and ">".
    def sanitise_and_decode_entities(self, s):
        s = self.sanitise_nuke_names(s)
        return s.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")

    def get_using_old_API(self, lens):
        old_api = True
        try:
            # if this fails, the new python API for getLensLDAdjustableParameter will be used.
            # there was a bug until version 1.3, which lead to old_api false, always.
            for para in (self.get_LD_model_parameter_list(model)):
                tde4.getLensLDAdjustableParameter(lens,para,1)
        except:
            old_api = False

        #print('oldapi?, ', old_api)
        return old_api

    def get_dynamic_distortion_mode(self, lens):
        try:
            dyndistmode  = tde4.getLensDynamicDistortionMode(lens)
        except:
            # For 3DE4 Release 1:
            if tde4.getLensDynamicDistortionFlag(lens) == 1:
                dyndistmode = "DISTORTION_DYNAMIC_FOCAL_LENGTH"
            else:
                dyndistmode = "DISTORTION_STATIC"

        return dyndistmode

    def are_all_values_equal_to(self, cam, lens, para, value):
        # get what dynamic distortion mode (if any), as well as the
        # if we're using the old API
        num_frames   = tde4.getCameraNoFrames(cam)
        dyndistmode  = self.get_dynamic_distortion_mode(lens)
        old_api = self.get_using_old_API(lens)

        # dynamic focal distance
        if old_api:
            if dyndistmode=="DISTORTION_DYNAMIC_FOCAL_LENGTH":
                for frame in range(1,num_frames + 1):
                    focal = tde4.getCameraFocalLength(cam,frame)
                    if value != tde4.getLensLDAdjustableParameter(lens, para, focal):
                        return False

            if dyndistmode=="DISTORTION_DYNAMIC_FOCUS_DISTANCE":
                # dynamic focus distance
                for frame in range(1,num_frames + 1):
                    # Older Releases do not have Focus-methods.
                    try:
                        focus = tde4.getCameraFocus(cam,frame)
                    except:
                        focus = 100.0
                    if value != tde4.getLensLDAdjustableParameter(lens, para, focus):
                        return False

            # static distortion
            if dyndistmode=="DISTORTION_STATIC":
                if value != tde4.getLensLDAdjustableParameter(lens, para, 1):
                    return False
        else:
        # new API
            if dyndistmode=="DISTORTION_STATIC":
                if value != tde4.getLensLDAdjustableParameter(lens, para, 1, 1):
                    return False
            else:
            # dynamic
                for frame in range(1,num_frames + 1):
                    focal = tde4.getCameraFocalLength(cam,frame)
                    focus = tde4.getCameraFocus(cam,frame)
                    if value != tde4.getLensLDAdjustableParameter(lens, para, focal, focus):
                        return False

        # if we made it here, then we can assume all values of the param are equal to value
        return True

    def get_lens_type_LD_value(self, cam, lens):
        model = tde4.getLensLDModel(lens)
        if model == '3DE Classic LD Model':
            has_anamorphic_squeeze = not self.are_all_values_equal_to(cam, lens, 'Anamorphic Squeeze', 1.0)
            has_curvature_x        = not self.are_all_values_equal_to(cam, lens, 'Curvature X', 0.0)
            has_curvature_y        = not self.are_all_values_equal_to(cam, lens, 'Curvature Y', 0.0)
            is_anamorphic = has_anamorphic_squeeze or has_curvature_x or has_curvature_y
        elif model == '3DE4 Anamorphic - Standard, Degree 4' or model == '3DE4 Anamorphic, Degree 6':
            is_anamorphic = True
        else:
            is_anamorphic = False

        if model == '3DE4 Radial - Standard, Degree 4':
            is_beamsplitter = True
        else:
            is_beamsplitter = False

        if is_anamorphic:
            return "Anamorphic"
        elif is_beamsplitter:
            return "Beam Splitter"
        else:
            return "Spherical"

    def set_model_preset_LD_values(self, ld_node, lens):

        model = tde4.getLensLDModel(lens)

        # write out the model, 3DE's names should match the LD presets
        ld_node.add_knob('distortionModelPreset', '"3DEqualizer/%s"' % model)

        if model == '3DE Classic LD Model':
            ld_node.add_knob('distortionOrder', '{2 0}')
            ld_node.add_knob('distortionDomain', 'Rectilinear')
            ld_node.add_knob('normalisationType', 'Diagonal')
        elif model == '3DE4 Radial - Standard, Degree 4':
            ld_node.add_knob('distortionModelType', '"Radial-Tangential/R-T Uncoupled"\n')
            ld_node.add_knob('distortionOrder', '{2 0}')
            ld_node.add_knob('distortionDomain', 'Rectilinear')
            ld_node.add_knob('normalisationType', 'Diagonal')
        elif model == '3DE4 Radial - Fisheye, Degree 8':
            ld_node.add_knob('projection', '"Fisheye/Fisheye Equisolid"')
            ld_node.add_knob('distortionOrder', '{4 0}')
            ld_node.add_knob('distortionDomain', 'Rectilinear')
            ld_node.add_knob('normalisationType', 'Diagonal')
        elif model == '3DE4 Anamorphic - Standard, Degree 4':
            ld_node.add_knob('distortionModelType', '"Radial Asymmetric"')
            ld_node.add_knob('distortionOrder', '{2 0}')
            ld_node.add_knob('distortionDomain', 'Rectilinear')
            ld_node.add_knob('normalisationType', 'Diagonal')
        elif model == '3DE4 Anamorphic, Degree 6':
            ld_node.add_knob('distortionModelType', '"Radial Asymmetric"')
            ld_node.add_knob('distortionOrder', '{3 0}')
            ld_node.add_knob('distortionDomain', 'Rectilinear')
            ld_node.add_knob('normalisationType', 'Diagonal')

    def get_LD_model_parameter_list(self, model):
        l = []
        for p in range(tde4.getLDModelNoParameters(model)):
            l.append(tde4.getLDModelParameterName(model, p))
        return l

    def export_fov_transform(self, f, cam):
        # xa,xb,ya,yb in unit coordinates, in this order.
        fov = tde4.getCameraFOV(cam)

        # Test if FOV is non-default
        if (fov[0] == 0.0 and fov[1] == 1.0 and fov[2] == 0.0 and fov[3] == 1.0):
            return

        # if it's not default, we need to create a Transform node before the
        # LD node.
        t_node = NukeNode('Transform')
        t_node.add_knob('translate', '{{"width/2 * (1 - (%f + %f))"} {"height/2 * (1 - (%f + %f))"}}' % (fov[0], fov[1], fov[2], fov[3]))
        t_node.add_knob('scale', '{{"1 / (%f - %f)"} {"1 / (%f - %f)" x1 1}}' % (fov[1], fov[0], fov[3], fov[2]))
        t_node.add_knob('center', '{{"width/2 * (%f + %f)"} {"height/2 * (%f + %f)"}}' % (fov[0], fov[1], fov[2], fov[3]))
        t_node.add_knob('black_outside', 'false')
        t_node.add_knob('name', 'Transform_FOV_' + self.sanitise_and_decode_entities(tde4.getCameraName(cam)))

        t_node.write_out(f)

    def export_film_back_transform(self, f, cam):
        lens   = tde4.getCameraLens(cam)
        w_fb_cm = tde4.getLensFBackWidth(lens)
        h_fb_cm = tde4.getLensFBackHeight(lens)

        # if it's not default, we need to create a Transform node before the
        # LD node.
        t_node = NukeNode('Transform')
        t_node.add_knob('scale', '{"%f / sqrt(%f * %f + %f * %f) * sqrt(width*width + height*height) / width"} ' % (w_fb_cm, w_fb_cm, w_fb_cm, h_fb_cm, h_fb_cm), 0)
        t_node.add_knob('scale', '{"%f / sqrt(%f * %f + %f * %f) * sqrt(width*width + height*height) / height"} ' % (h_fb_cm, w_fb_cm, w_fb_cm, h_fb_cm, h_fb_cm), 1)
        t_node.add_knob('center', '{{"width/2"} {"height/2"}}')
        t_node.add_knob('black_outside', 'false')
        t_node.add_knob('disable', '{{"abs(width/height - %f/%f) < 1e-4"}}' % (w_fb_cm, h_fb_cm) )
        t_node.add_knob('name', 'Transform_FilmBack_' + self.sanitise_and_decode_entities(tde4.getCameraName(cam)))

        t_node.write_out(f)

    def export_lensdistortion_node(self, f, cam, offset, nuke_path):
        lens     = tde4.getCameraLens(cam)
        model     = tde4.getLensLDModel(lens)
        num_frames     = tde4.getCameraNoFrames(cam)
        w_fb_cm = tde4.getLensFBackWidth(lens)
        h_fb_cm = tde4.getLensFBackHeight(lens)
        lco_x_cm = tde4.getLensLensCenterX(lens)
        lco_y_cm = tde4.getLensLensCenterY(lens)
        pxa = tde4.getLensPixelAspect(lens)
        # xa,xb,ya,yb in unit coordinates, in this order.
        fov = tde4.getCameraFOV(cam)

        old_api = self.get_using_old_API(lens)

        #print('camera: ', tde4.getCameraName(cam))
        #print('offset:', offset)
        #print('lens:', tde4.getLensName(lens))
        #print('model: ', model)

        # Lens Distortion node object to make setting up and writing our node out
        # easier.
        ld_node = NukeNode('LensDistortion2')

        ld_node.add_knob('output', 'Undistort')

        # write lens type
        ld_node.add_knob('lensType', '"%s"' % self.get_lens_type_LD_value(cam, lens))

        # write the model preset
        self.set_model_preset_LD_values(ld_node, lens)

        # write focal length curve if dynamic
        if tde4.getCameraZoomingFlag(cam):
            #print('dynamic focal length')
            focalStr = '{{curve '
            for frame in range(1,num_frames + 1):
                focalStr += 'x%i' % (frame+offset)
                focalStr += ' %.7f ' % tde4.getCameraFocalLength(cam,frame)
            focalStr += '}}'
            ld_node.add_knob('focal', focalStr)
        # write static focal length else
        else:
            #print('static focal length')
            ld_node.add_knob('focal', '%.7f ' % tde4.getCameraFocalLength(cam,1))

        # write focus distance curve if dynamic
        # Note, focus distance is a 3DE specific knob, we add it as a
        # user knob for LensDistortion
        ld_node.add_duplicate_knob('addUserKnob', '{20 3DE}')
        #     ld_node.add_duplicate_knob('addUserKnob', '{3 focusDistance l "Focus Distance"}')
        #
        #     if not old_api:
        #         try:
        #             if tde4.getCameraFocusMode(cam) == "FOCUS_DYNAMIC":
        #                 print('dynamic focus distance')
        #                 focusDistanceStr = '{{curve '
        #                 for frame in range(1,num_frames + 1):
        #                     focusDistanceStr += 'x%i' % (frame+offset)
        #                     focusDistanceStr += ' %.7f ' % tde4.getCameraFocus(cam,frame)
        #                 focusDistanceStr += '}}'
        #                 ld_node.add_knob('focusDistance', focusDistanceStr)
        #             else:
        #                 print('static focus distance')
        #                 ld_node.add_knob('focusDistance', '%.7f' % tde4.getCameraFocus(cam,1))
        #         except:
        # # For 3DE4 Release 1:
        #             ld_node.add_knob('focusDistance', '100,0')
        # # write static focus distance else
        #     else:
        # # For 3DE4 Release 1:
        #         ld_node.add_knob('focusDistance', '100,0')

        # write camera
        ld_node.add_knob('sensorSize', '{%.7f %.7f}' % (w_fb_cm, h_fb_cm) )
        ld_node.add_knob('centre', '{{" %.7f / sqrt(%.7f*%.7f + %.7f*%.7f) * 2} {" %.7f / sqrt(%.7f*%.7f + %.7f*%.7f) * 2}}' % (lco_x_cm, w_fb_cm, w_fb_cm, h_fb_cm, h_fb_cm, lco_y_cm, w_fb_cm, w_fb_cm, h_fb_cm, h_fb_cm))

        # write distortion parameters
        #
        # dynamic distortion

        # get what dynamic distortion mode (if any), as well as the
        # if we're using the old API

        dyndistmode    = self.get_dynamic_distortion_mode(lens)

        if old_api:
            if dyndistmode=="DISTORTION_DYNAMIC_FOCAL_LENGTH":
                #print('dynamic lens distortion, focal length')
                # dynamic focal length (zoom)
                for para in (self.get_LD_model_parameter_list(model)):
                    paraStr = ' {{curve '
                    for frame in range(1,num_frames + 1):
                        focal = tde4.getCameraFocalLength(cam,frame)
                        distance = tde4.getCameraFocus(cam,frame)
                        paraStr += 'x%i' % (frame+offset)
                        paraStr += ' %.7f '%tde4.getLensLDAdjustableParameter(lens, para, focal, distance)
                    paraStr += '}}'
                    ld_node.add_knob(self.nukify_name(para), paraStr, self.nuke_para_index(para))

            if dyndistmode=="DISTORTION_DYNAMIC_FOCUS_DISTANCE":
                #print('dynamic lens distortion, focus distance')
                # dynamic focus distance
                for para in (self.get_LD_model_parameter_list(model)):
                    paraStr = ' {{curve '
                    for frame in range(1,num_frames + 1):
                        # Older Releases do not have Focus-methods.
                        try:
                            focus = tde4.getCameraFocus(cam,frame)
                        except:
                            focus = 100.0
                        paraStr += 'x%i' % (frame+offset)
                        paraStr += ' %.7f '%tde4.getLensLDAdjustableParameter(lens, para, focus)
                    paraStr += '}}'
                    ld_node.add_knob(self.nukify_name(para), paraStr, self.nuke_para_index(para))

                    # static distortion
            if dyndistmode=="DISTORTION_STATIC":
                #print('static lens distortion')
                for para in (self.get_LD_model_parameter_list(model)):
                    ld_node.add_knob(self.nukify_name(para), '%.7f' % tde4.getLensLDAdjustableParameter(lens, para, 1), self.nuke_para_index(para))
        else:
            # new API
            if dyndistmode=="DISTORTION_STATIC":
                #print('static lens distortion')
                for para in (self.get_LD_model_parameter_list(model)):
                    ld_node.add_knob(self.nukify_name(para), '%.7f' % tde4.getLensLDAdjustableParameter(lens, para, 1, 1), self.nuke_para_index(para))
            else:
                #print('dynamic lens distortion,')
                # dynamic
                for para in (self.get_LD_model_parameter_list(model)):
                    # print(model)
                    if '3DE Classic LD Model' != model and 'Squeeze' in para:
                        paraStr = ' {curve '
                    else:
                        paraStr = ' {{curve '
                    for frame in range(1,num_frames + 1):
                        focal = tde4.getCameraFocalLength(cam,frame)
                        focus = tde4.getCameraFocus(cam,frame)
                        paraStr += 'x%i' % (frame+offset)
                        paraStr += ' %.7f '%tde4.getLensLDAdjustableParameter(lens, para, focal, focus)
                    if '3DE Classic LD Model' != model and 'Squeeze' in para:
                        paraStr += ' }'
                    else:
                        paraStr += '}}'
                    ld_node.add_knob(self.nukify_name(para), paraStr, self.nuke_para_index(para))

        ld_node.add_knob('name', 'LensDistortion_' + self.sanitise_and_decode_entities(tde4.getCameraName(cam)))

        ld_node.write_out(f)

    def export_film_back_inverse_transform(self, f, cam):
        lens   = tde4.getCameraLens(cam)
        w_fb_cm = tde4.getLensFBackWidth(lens)
        h_fb_cm = tde4.getLensFBackHeight(lens)

        # if it's not default, we need to create a Transform node after the
        # LD node.
        t_node = NukeNode('Transform')
        t_node.add_knob('scale', '{"1 / (%f / sqrt(%f * %f + %f * %f) * sqrt(width*width + height*height) / width)"} ' % (w_fb_cm, w_fb_cm, w_fb_cm, h_fb_cm, h_fb_cm), 0)
        t_node.add_knob('scale', '{"1 / (%f / sqrt(%f * %f + %f * %f) * sqrt(width*width + height*height) / height)"} ' % (h_fb_cm, w_fb_cm, w_fb_cm, h_fb_cm, h_fb_cm), 1)
        t_node.add_knob('center', '{{"width/2"} {"height/2"}}')
        t_node.add_knob('black_outside', 'false')
        t_node.add_knob('disable', '{{"abs(width/height - %f/%f) < 1e-4"}}' % (w_fb_cm, h_fb_cm) )
        t_node.add_knob('name', 'Transform_InverseFilmBack_' + self.sanitise_and_decode_entities(tde4.getCameraName(cam)))

        t_node.write_out(f)

    def export_fov_inverse_transform(self, f, cam):
        # xa,xb,ya,yb in unit coordinates, in this order.
        fov = tde4.getCameraFOV(cam)

        # Test if FOV is non-default
        if (fov[0] == 0.0 and fov[1] == 1.0 and fov[2] == 0.0 and fov[3] == 1.0):
            return

        # if it's not default, we need to create a Transform node after the
        # LD node.
        t_node = NukeNode('Transform')
        t_node.add_knob('translate', '{{"width/2 * (-1 + (%f + %f))"} {"height/2 * (-1 + (%f + %f))"}}' % (fov[0], fov[1], fov[2], fov[3]))
        t_node.add_knob('scale', '{{"(%f - %f)"} {"(%f - %f)" x1 1}}' % (fov[1], fov[0], fov[3], fov[2]))
        t_node.add_knob('center', '{{"width/2"} {"height/2"}}')
        t_node.add_knob('black_outside', 'false')
        t_node.add_knob('name', 'Transform_Inverse_FOV_' + self.sanitise_and_decode_entities(tde4.getCameraName(cam)))

        t_node.write_out(f)

    def doIt(self):
        '''
        ##########
        #  prev  #
        ##########
        '''

        path = tde4.getWidgetValue(self.req, 'userInput')
        self.nuke_path = path
        self.pubPath = os.path.dirname(path)

        left_camera = tde4.getWidgetValue(self.req, 'left_camera')
        camlist2 = []
        camlist2.append(self.camList[left_camera-1])

        set_attr = tde4.getCameraSequenceAttr(self.camList[left_camera-1])
        seq_width = tde4.getCameraImageWidth(self.camList[left_camera-1])
        seq_height = tde4.getCameraImageHeight(self.camList[left_camera-1])

        stereo = tde4.getWidgetValue(self.req, 'stereo')
        if stereo:
            right_camera = tde4.getWidgetValue(self.req, 'right_camera')
            if left_camera == right_camera:
                raise Exception("Left Camera and Right Camera are Equal!")
            else:
                camlist2.append(self.camList[right_camera-1])
        start_frame = int(tde4.getWidgetValue(self.req, "start_frame")) - 1
        DD_common.makeDir(self.pubPath)

        ld = "# Created by 3DEqualizer4 using DD_Setup ESxport Nuke Distortion Nodes export script\n"
        for cam in camlist2:
            ld += DD_common.exportNukeDewarpNode2(cam, start_frame)
        # write nuke script.
        f2 = open(self.nuke_path, "w")
        f2.write(ld)
        f2.close()

        # '''
        # ###########
        # # nuke 11 #
        # ###########
        # '''
        # # nukeXpubpath = ""
        # # if '_LD' in self.nuke_path:
        #     # nukeXpubpath = self.nuke11_path
        #
        # f = open(self.nuke11_path, "w")
        # f.write(
        #     '# Created by 3DEqualizer4 using Export NukeX Native LensDistortion Nodes export script\n')
        #
        # # write FOV pre-transform, if necessary
        # self.export_fov_transform(f, cam)
        #
        # # write Film back pre-transform
        # self.export_film_back_transform(f, cam)
        #
        # # write the LensDistortion node
        # self.export_lensdistortion_node(f, cam, start_frame, self.nuke11_path)
        #
        # # write Film back post-transform
        # self.export_film_back_inverse_transform(f, cam)
        #
        # # write Inverse FOV post-transform, if necessary
        # self.export_fov_inverse_transform(f, cam)
        #
        # f.close()

        tde4.postQuestionRequester("Export nuke distortion node for camera" , "Publish Success!", "Ok")

        # write json file.
        left_camera = tde4.getWidgetValue(self.req, 'left_camera')
        lens_json = dict()
        lens_json["show"] = self.show
        lens_json["seq"] = self.seq
        lens_json["shot"] = self.shot
        lens_json["3deProject"] = tde4.getProjectPath()
        lens_json["distortionNukeScript"] = self.nuke_path
        # lens_json["distortionNuke11Script"] = self.nuke11_path
        lens_json["user"] = getpass.getuser()
        lens_json["startFrame"] = tde4.getCameraSequenceAttr(self.camList[left_camera-1])[0]
        lens_json["endFrame"] = tde4.getCameraSequenceAttr(self.camList[left_camera-1])[1]
        lens_json["resWidth"] = tde4.getCameraImageWidth(self.camList[left_camera-1])
        lens_json["resHeight"] = tde4.getCameraImageHeight(self.camList[left_camera-1])
        lens_json["plateType"] = self.plateType
        lens_json["leftCamera"] = tde4.getCameraName(self.camList[left_camera-1])
        lens_json["leftCameraPlate"] = tde4.getCameraPath(self.camList[left_camera-1])
        dxUI = dxUIcommon.setOverscanWidget(self.seq, self.camList[left_camera-1])
        lens_json['overscanSize'] = dxUI.computeOverscanValue()
        if stereo:
            lens_json["rightCamera"] = tde4.getCameraName(self.camList[right_camera-1])
            lens_json["rightCameraPlate"] = tde4.getCameraPath(self.camList[right_camera-1])
            lens_json["stereo"] = True
        else:
            lens_json["rightCamera"] = ""
            lens_json["rightCameraPlate"] = ""
            lens_json["stereo"] = False

        filepath = self.nuke_path.replace(".nk", ".shotdb")
        try:
            fj = open(filepath, "w")
            fj.write(json.dumps(lens_json, sort_keys=True, indent=4, separators=(",", ":")))
            fj.close()
        except:
            raise Exception("Exporting JSON failed! It goes something wrong.")
        try:
            #print(insertDB.distortion_insert(filepath))
            insertDB.distortion_insert(filepath)
        except:
            print("db insert error")
