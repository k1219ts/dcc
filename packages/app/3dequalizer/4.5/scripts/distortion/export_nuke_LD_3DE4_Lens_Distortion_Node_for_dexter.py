# 3DE4.script.name:     7. Export Nuke LD_3DE4 Lens Distortion Node
# 3DE4.script.version:  v1.3.0
# 3DE4.script.gui:      Main Window::DD_Setup
# 3DE4.script.comment:  Creates an LD_3DE4 Lens Distortion Node for each selected camera (exports .nk script)
# 3DE4.script.comment:  The five LD_3DE4 Nodes were introduced in Lens Distortion Plugin Kit version 1.7 (2013-12-11).
# 3DE4.script.comment:  With the release of Nuke8.0 we will update and promote these plugins on linux, osx and windows,
# 3DE4.script.comment:  also for Nuke6.2 (osx and linux), Nuke6.3 and Nuke7.0.
# 3DE4.script.comment:  This script should not be mistaken for "Export Weta Nuke Distortion"
# 3DE4.script.comment:  Creates a native LensDistortion Node for each selected camera (exports .nk script)
# 3DE4.script.comment:  This allows you to use the new GPU accelerated LensDistortion nodes found in NukeX 11.0 onwards.
# 3DE4.script.comment:  All five distortion models in 3DE are compatible with the NukeX's new LensDistortion node.
# Date: 2018-04-16
# Original script: Wolgang Niedermeier (Weta)
# Author: Uwe Sassenberg (SDV)
# For 3DE4 releases r1 or higher

# Edited by jungmin.lee(RND)

import os
import DD_common
import tde4
import string
import re
import json
import TDE4_common

try:
    from dxpublish import insertDB
except:
    import sys
    sys.path.insert(0, "/netapp/backstage/dev/lib/python_lib_2.6")
#       sys.path.append("/netapp/backstage/pub/lib/python_lib")
    from dxpublish import insertDB

class CancelException(Exception):
    pass

'''
###########
# NUKE 11 #
###########
'''
class NukeNode:
    ''' Class to represent a Nuke node, and the knob names & values that
    belong to it. The idea is to make writing nodes to a script easier. '''
    def __init__(self, node_name):
        self.knob_name_value_tuples = dict()
        self.ordered_knob_names = []
        self.node_name = node_name

    def add_knob(self, name, value, index=0):
        if not name in self.knob_name_value_tuples.iterkeys():
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

# We translate our API model and parameter names into Nuke identifiers.
# The rules are:
# - The empty string maps to an underscore
# - When the names starts with 0-9 it gets an underscore
# - All non-alphanumeric characters are mapped to underscores, but sequences
#   of underscores shrink to a single underscore, looks better.
def sanitise_nuke_names(s):
    if s == "":
        return "_"
    if s[0] in "0123456789":
        t = "_"
    else:
        t = ""
    t += string.join(re.sub("[+,:; _-]+","_",s.strip()).split())

    return t

def nukify_name(s):
    # ensure param name has valid characters
    t = sanitise_nuke_names(s)

    # map TDE4 param name to Nuke's LD name
    t = map_TDE4_paramname_to_nuke(t)[0]

    return t

def nuke_para_index(s):

    # ensure param name has valid characters
    t = sanitise_nuke_names(s)

    # map TDE4 param name to Nuke's LD index
    t = map_TDE4_paramname_to_nuke(t)[1]

    return t

def map_TDE4_paramname_to_nuke(tde4param):
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
    }

    if not tde4param in _knob_mapping:
        raise Exception('     TDE4 knob name "%s" not mapped to Nuke LD node knobs.     ' % tde4param )
    else:
        return _knob_mapping[tde4param]

# Nuke interprets entities like "<" and ">".
def sanitise_and_decode_entities(s):
    s = sanitise_nuke_names(s)
    return s.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")

def get_using_old_API(lens):
    old_api = True
    try:
        # if this fails, the new python API for getLensLDAdjustableParameter will be used.
        # there was a bug until version 1.3, which lead to old_api false, always.
        for para in (get_LD_model_parameter_list(model)):
            tde4.getLensLDAdjustableParameter(lens,para,1)
    except:
        old_api = False

    #print 'oldapi?, ', old_api
    return old_api

def get_dynamic_distortion_mode(lens):
    try:
        dyndistmode  = tde4.getLensDynamicDistortionMode(lens)
    except:
        # For 3DE4 Release 1:
        if tde4.getLensDynamicDistortionFlag(lens) == 1:
            dyndistmode = "DISTORTION_DYNAMIC_FOCAL_LENGTH"
        else:
            dyndistmode = "DISTORTION_STATIC"

    return dyndistmode


def are_all_values_equal_to(cam, lens, para, value):
    # get what dynamic distortion mode (if any), as well as the
    # if we're using the old API
    num_frames   = tde4.getCameraNoFrames(cam)
    dyndistmode  = get_dynamic_distortion_mode(lens)
    old_api = get_using_old_API(lens)

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

def get_lens_type_LD_value(cam, lens):
    model = tde4.getLensLDModel(lens)
    if model == '3DE Classic LD Model':
        has_anamorphic_squeeze = not are_all_values_equal_to(cam, lens, 'Anamorphic Squeeze', 1.0)
        has_curvature_x        = not are_all_values_equal_to(cam, lens, 'Curvature X', 0.0)
        has_curvature_y        = not are_all_values_equal_to(cam, lens, 'Curvature Y', 0.0)
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

def set_model_preset_LD_values(ld_node, lens):

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

class CancelException(Exception):
  pass

def get_LD_model_parameter_list(model):
  l = []
  for p in range(tde4.getLDModelNoParameters(model)):
    l.append(tde4.getLDModelParameterName(model, p))
  return l

def export_fov_transform(f, cam):
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
  t_node.add_knob('name', 'Transform_FOV_' + sanitise_and_decode_entities(tde4.getCameraName(cam)))

  t_node.write_out(f)

def export_fov_inverse_transform(f, cam):
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
  t_node.add_knob('name', 'Transform_Inverse_FOV_' + sanitise_and_decode_entities(tde4.getCameraName(cam)))

  t_node.write_out(f)

def export_film_back_transform(f, cam):
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
  t_node.add_knob('name', 'Transform_FilmBack_' + sanitise_and_decode_entities(tde4.getCameraName(cam)))

  t_node.write_out(f)

def export_film_back_inverse_transform(f, cam):
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
  t_node.add_knob('name', 'Transform_InverseFilmBack_' + sanitise_and_decode_entities(tde4.getCameraName(cam)))

  t_node.write_out(f)

def export_lensdistortion_node(f, cam, offset, nuke_path):
  lens   = tde4.getCameraLens(cam)
  model   = tde4.getLensLDModel(lens)
  num_frames   = tde4.getCameraNoFrames(cam)
  w_fb_cm = tde4.getLensFBackWidth(lens)
  h_fb_cm = tde4.getLensFBackHeight(lens)
  lco_x_cm = tde4.getLensLensCenterX(lens)
  lco_y_cm = tde4.getLensLensCenterY(lens)
  pxa = tde4.getLensPixelAspect(lens)
# xa,xb,ya,yb in unit coordinates, in this order.
  fov = tde4.getCameraFOV(cam)

  old_api = get_using_old_API(lens)

  #print 'camera: ', tde4.getCameraName(cam)
  #print 'offset:', offset
  #print 'lens:', tde4.getLensName(lens)
  #print 'model: ', model

# Lens Distortion node object to make setting up and writing our node out
# easier.
  ld_node = NukeNode('LensDistortion2')

  ld_node.add_knob('output', 'Undistort')

# write lens type
  ld_node.add_knob('lensType', '"%s"' % get_lens_type_LD_value(cam, lens))

# write the model preset
  set_model_preset_LD_values(ld_node, lens)

# write focal length curve if dynamic
  if tde4.getCameraZoomingFlag(cam):
    #print 'dynamic focal length'
    focalStr = '{{curve '
    for frame in range(1,num_frames + 1):
      focalStr += 'x%i' % (frame+offset)
      focalStr += ' %.7f ' % tde4.getCameraFocalLength(cam,frame)
    focalStr += '}}'
    ld_node.add_knob('focal', focalStr)
# write static focal length else
  else:
    #print 'static focal length'
    ld_node.add_knob('focal', '%.7f ' % tde4.getCameraFocalLength(cam,1))

# write focus distance curve if dynamic
# Note, focus distance is a 3DE specific knob, we add it as a
# user knob for LensDistortion
  ld_node.add_duplicate_knob('addUserKnob', '{20 3DE}')
#   ld_node.add_duplicate_knob('addUserKnob', '{3 focusDistance l "Focus Distance"}')
#
#   if not old_api:
#     try:
#       if tde4.getCameraFocusMode(cam) == "FOCUS_DYNAMIC":
#         print 'dynamic focus distance'
#         focusDistanceStr = '{{curve '
#         for frame in range(1,num_frames + 1):
#           focusDistanceStr += 'x%i' % (frame+offset)
#           focusDistanceStr += ' %.7f ' % tde4.getCameraFocus(cam,frame)
#         focusDistanceStr += '}}'
#         ld_node.add_knob('focusDistance', focusDistanceStr)
#       else:
#         print 'static focus distance'
#         ld_node.add_knob('focusDistance', '%.7f' % tde4.getCameraFocus(cam,1))
#     except:
# # For 3DE4 Release 1:
#       ld_node.add_knob('focusDistance', '100,0')
# # write static focus distance else
#   else:
# # For 3DE4 Release 1:
#     ld_node.add_knob('focusDistance', '100,0')

# write camera
  ld_node.add_knob('sensorSize', '{%.7f %.7f}' % (w_fb_cm, h_fb_cm) )
  ld_node.add_knob('centre', '{{" %.7f / sqrt(%.7f*%.7f + %.7f*%.7f) * 2} {" %.7f / sqrt(%.7f*%.7f + %.7f*%.7f) * 2}}' % (lco_x_cm, w_fb_cm, w_fb_cm, h_fb_cm, h_fb_cm, lco_y_cm, w_fb_cm, w_fb_cm, h_fb_cm, h_fb_cm))

# write distortion parameters
#
# dynamic distortion

# get what dynamic distortion mode (if any), as well as the
# if we're using the old API

  dyndistmode  = get_dynamic_distortion_mode(lens)

  if old_api:
    if dyndistmode=="DISTORTION_DYNAMIC_FOCAL_LENGTH":
      #print 'dynamic lens distortion, focal length'
# dynamic focal length (zoom)
      for para in (get_LD_model_parameter_list(model)):
        paraStr = ' {{curve '
        for frame in range(1,num_frames + 1):
          focal = tde4.getCameraFocalLength(cam,frame)
          distance = tde4.getCameraFocus(cam,frame)
          paraStr += 'x%i' % (frame+offset)
          paraStr += ' %.7f '%tde4.getLensLDAdjustableParameter(lens, para, focal, distance)
        paraStr += '}}'
        ld_node.add_knob(nukify_name(para), paraStr, nuke_para_index(para))

    if dyndistmode=="DISTORTION_DYNAMIC_FOCUS_DISTANCE":
      #print 'dynamic lens distortion, focus distance'
# dynamic focus distance
      for para in (get_LD_model_parameter_list(model)):
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
        ld_node.add_knob(nukify_name(para), paraStr, nuke_para_index(para))

# static distortion
    if dyndistmode=="DISTORTION_STATIC":
      #print 'static lens distortion'
      for para in (get_LD_model_parameter_list(model)):
        ld_node.add_knob(nukify_name(para), '%.7f' % tde4.getLensLDAdjustableParameter(lens, para, 1), nuke_para_index(para))
  else:
# new API
    if dyndistmode=="DISTORTION_STATIC":
      #print 'static lens distortion'
      for para in (get_LD_model_parameter_list(model)):
        ld_node.add_knob(nukify_name(para), '%.7f' % tde4.getLensLDAdjustableParameter(lens, para, 1, 1), nuke_para_index(para))
    else:
      #print 'dynamic lens distortion,'
# dynamic
      for para in (get_LD_model_parameter_list(model)):
        # print model
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
        ld_node.add_knob(nukify_name(para), paraStr, nuke_para_index(para))

  ld_node.add_knob('name', 'LensDistortion_' + sanitise_and_decode_entities(tde4.getCameraName(cam)))

  ld_node.write_out(f)

# copied from 'export_nuke_LD_3DE4_Lens_Distortion_Node.py'
def set_initial_frame_widget(wdg,a,b):
  val = tde4.getWidgetValue(nuke_node_req,"option_menu_default_initial_frame")
  try:
    initial_frame_3de4 = tde4.getCameraFrameOffset(id_cam)
  except:
    initial_frame_3de4 = 1
# 3 means user-defined. In this case the GUI allows to enter a frame value.
  if val == 3:
    tde4.setWidgetSensitiveFlag(nuke_node_req,"text_initial_frame_nuke",True)
  elif val == 2:
    tde4.setWidgetSensitiveFlag(nuke_node_req,"text_initial_frame_nuke",False)
    tde4.setWidgetValue(nuke_node_req,"text_initial_frame_nuke",str(initial_frame_3de4))
  elif val == 1:
    tde4.setWidgetSensitiveFlag(nuke_node_req,"text_initial_frame_nuke",False)
    tde4.setWidgetValue(nuke_node_req,"text_initial_frame_nuke",str(tde4.getCameraSequenceAttr(id_cam)[0]))

'''
###########
#  prev   #
###########
'''

def _export_LD_callback(requester, widget, action):
    if widget == "stereo":
        mode = tde4.getWidgetValue(nuke_node_req, "stereo")
        if mode == 0:
            tde4.setWidgetSensitiveFlag(nuke_node_req, "right_camera", 0)
            tde4.setWidgetLabel(nuke_node_req, "left_camera", "Main Camera")
            tde4.modifyOptionMenuWidget(nuke_node_req, "right_camera", "", "")
        if mode == 1:
            tde4.setWidgetSensitiveFlag(nuke_node_req, "right_camera", 1)
            tde4.setWidgetLabel(nuke_node_req, "left_camera", "Left Camera")
            tde4.modifyOptionMenuWidget(nuke_node_req, "right_camera", "Right Camera", *camlist_name)

def write_lens_json():
    lens_json = dict()
    lens_json["show"] = j["show"]
    lens_json["seq"] = j["seq"]
    lens_json["shot"] = j["shot"]
    lens_json["3deProject"] = tde4.getProjectPath()
    lens_json["distortionNukeScript"] = nuke_path
    lens_json["distortionNuke11Script"] = nukeXpubpath
    lens_json["user"] = USER
    lens_json["startFrame"] = tde4.getCameraSequenceAttr(camlist[left_camera-1])[0]
    lens_json["endFrame"] = tde4.getCameraSequenceAttr(camlist[left_camera-1])[1]
    lens_json["resWidth"] = tde4.getCameraImageWidth(camlist[left_camera-1])
    lens_json["resHeight"] = tde4.getCameraImageHeight(camlist[left_camera-1])
    lens_json["plateType"] = j["platetype"]
    lens_json["leftCamera"] = tde4.getCameraName(camlist[left_camera-1])
    lens_json["leftCameraPlate"] = tde4.getCameraPath(camlist[left_camera-1])
    lens_json['overscanSize'] = OVERSCAN
    if stereo:
        lens_json["rightCamera"] = tde4.getCameraName(camlist[right_camera-1])
        lens_json["rightCameraPlate"] = tde4.getCameraPath(camlist[right_camera-1])
        lens_json["stereo"] = True
    else:
        lens_json["rightCamera"] = ""
        lens_json["rightCameraPlate"] = ""
        lens_json["stereo"] = False

    filepath = nuke_path.replace(".nk", ".shotdb")
    try:
        fj = open(filepath, "w")
        fj.write(json.dumps(lens_json, sort_keys=True, indent=4, separators=(",", ":")))
        fj.close()
    except:
        raise Exception("Exporting JSON failed! It goes something wrong.")
    try:
        #print insertDB.distortion_insert(filepath)
        insertDB.distortion_insert(filepath)
    except e:
        print e
        print "db insert error"


try:
    fu = open('/home/%s/.mmvuser'%os.getenv('USER'))
    USER = (fu.readline()).rstrip('\n')
    fu.close()
except:
    USER = os.getenv("USER")


# main
"""
if os.path.isfile("/tmp/.3de_shot_info"):
    f = open("/tmp/.3de_shot_info", "r")
    try:
        j = json.load(f)
        pubPath = os.path.join("/", "show", j["show"], "shot", j["seq"], j["shot"], "matchmove", "pub", "nuke")
    except:
        raise Exception("Parsing JSON failed! It goes something wrong.")
    finally:
        f.close()
"""

if os.environ.has_key('show'):
    j = {}
    try:
        for envKey in ['show', 'seq','shot','platetype']:
            j[envKey] = os.environ[envKey]
        pubPath = os.path.join("/", "show", j["show"], "shot", j["seq"], j["shot"], "matchmove", "pub", "nuke")
    except:
        raise Exception("Parsing ENVKEY failed! It goes something wrong.")

    try:
        camlist = tde4.getCameraList(1)
        if not camlist:
            raise Exception('     Only selected cameras will be exported     ')

        camlist_name = []
        for c in camlist:
            camlist_name.append(tde4.getCameraName(c))

        nuke_file = "%s_%s_matchmove"%(j["shot"], j["platetype"])
        pub_ver = DD_common.get_final_ver(pubPath, nuke_file, "*.nk")
        nuke_file += "_v%.2d.nk"%(pub_ver+1)
        nuke_path = os.path.join(pubPath, nuke_file)

        overscan_list = ["1.08", "1.1", "1.15", "1.2", "custom"]
        bbox = TDE4_common.bbdld_compute_bounding_box()

        overscan_scale = round(bbox[2] / bbox[4], 2)
        if overscan_scale < round(bbox[3] / bbox[5], 2):
            overscan_scale = round(bbox[3] / bbox[5], 2)

        #print bbox[0], bbox[1], bbox[2], bbox[3], overscan_scale

        if bbox[0] < 0.0000 or bbox[1] < 0.0000:
            if overscan_scale > 1.2:
                OVERSCAN = overscan_scale
            else:
                for i in reversed(overscan_list):
                    if i != "custom" and float(i) > overscan_scale:
                        OVERSCAN = float(i)
        else:
            OVERSCAN = 1.00
        # print OVERSCAN

        # open requester
        nuke_node_req = tde4.createCustomRequester()
        tde4.addFileWidget(nuke_node_req, 'userInput', 'Filename: ', '*.nk', nuke_path)
        tde4.addToggleWidget(nuke_node_req, "stereo", "Stereo")
        tde4.setWidgetCallbackFunction(nuke_node_req, "stereo", "_export_LD_callback")
        tde4.addOptionMenuWidget(nuke_node_req, "left_camera", "Main Camera", *camlist_name)
        tde4.addOptionMenuWidget(nuke_node_req, "right_camera", "", "")
        tde4.setWidgetSensitiveFlag(nuke_node_req, "right_camera", 0)
        tde4.addTextFieldWidget(nuke_node_req, 'start_frame', "Start Frame", str(tde4.getCameraSequenceAttr(camlist[0])[0]))

        ret     = tde4.postCustomRequester(nuke_node_req, "Export nuke distortion node for camera", 700, 0, "Ok", "Cancel")
        if ret != 1:
            raise CancelException('Cancelled')

        nuke_path = tde4.getWidgetValue(nuke_node_req, 'userInput')

        # check path and suffix
        if not nuke_path:
            raise Exception('     No path entered     ')

        if not nuke_path.endswith('.nk'):
            nuke_path = nuke_path+'.nk' #/show/xyfy/shot/DZR/DZR_0020/matchmove/pub/nuke/DZR_0020_EL1_org_matchmove_v02.nk

        # export
        if ret == 1:
            '''
            ##########
            #  prev  #
            ##########
            '''
            left_camera = tde4.getWidgetValue(nuke_node_req, "left_camera")
            camlist2 = []
            camlist2.append(camlist[left_camera-1])

            set_attr = tde4.getCameraSequenceAttr(camlist[left_camera-1])
            seq_width = tde4.getCameraImageWidth(camlist[left_camera-1])
            seq_height = tde4.getCameraImageHeight(camlist[left_camera-1])

            stereo = tde4.getWidgetValue(nuke_node_req, "stereo")
            if stereo:
                right_camera = tde4.getWidgetValue(nuke_node_req, "right_camera")
                if left_camera == right_camera:
                    raise Exception("Left Camera and Right Camera are Equal!")
                else:
                    camlist2.append(camlist[right_camera-1])
            start_frame = int(tde4.getWidgetValue(nuke_node_req, "start_frame")) - 1
            DD_common.make_dir(pubPath)

            ld = "# Created by 3DEqualizer4 using DD_Setup ESxport Nuke Distortion Nodes export script\n"
            for cam in camlist2:
                ld += DD_common.exportNukeDewarpNode2(cam, start_frame)
            # write nuke script.
            f2 = open(nuke_path, "w")
            f2.write(ld)
            f2.close()

            '''
            ###########
            # nuke 11 #
            ###########
            '''
            nukeXpubpath = ""
            if '_matchmove_v' in nuke_path:
                nukeXpubpath = nuke_path.replace('_matchmove_v','_matchmove_nuke11_v')
            try:
                f = open(nukeXpubpath, "w")
                f.write(
                    '# Created by 3DEqualizer4 using Export NukeX Native LensDistortion Nodes export script\n')

                # write FOV pre-transform, if necessary
                export_fov_transform(f, cam)

                # write Film back pre-transform
                export_film_back_transform(f, cam)

                # write the LensDistortion node
                export_lensdistortion_node(f, cam, start_frame, nukeXpubpath)

                # write Film back post-transform
                export_film_back_inverse_transform(f, cam)

                # write Inverse FOV post-transform, if necessary
                export_fov_inverse_transform(f, cam)
            except:
                pass

            finally:
                f.close()

            tde4.postQuestionRequester("Export nuke distortion node for camera" , "Publish Success!", "Ok")

            # write json file.
            write_lens_json()

    except CancelException, e:
        print e

    except Exception, e:
        print e
        tde4.postQuestionRequester('Error ', str(e), '  OK  ')

else:
#    tde4.postQuestionRequester("  Export nuke distortion node for camera  ", "There is no \".3de_shot_info.\"\nPlease open a project using \"Open Project\" script first.", "Ok")
    tde4.postQuestionRequester("  Export nuke distortion node for camera  ", "There is no \"ENVKEY\"\nPlease open a project using \"Open Project\" script first.", "Ok")
