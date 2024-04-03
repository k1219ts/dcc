##################################################
# Written by Daehwan Jang (daehwanj@gmail.com)
# Last updated: Apr 24, 2012
##################################################

#
#
# 3DE4.script.name:    Convert Dynamic Lens To Static Lens...
#
# 3DE4.script.version:    v1.0.1
#
# 3DE4.script.gui:    Main Window::Dexter
#
# 3DE4.script.comment:    Convert Dynamic Lens To Static Lens.
#
#

import os
import tde4

def getLDmodelParameterList(model):
    l = []
    for p in range(tde4.getLDModelNoParameters(model)):
        l.append(tde4.getLDModelParameterName(model, p))
    return l

#
# main
l = tde4.getLensList(1)

if l != []:
    for i in l:
        f = tde4.getLensDynamicDistortionFlag(i)
        if f == 0:
            tde4.postQuestionRequester("Convert To Static Lens...", "Error, Select a zoom lens to convert to a static lens first.", "Ok")
        else:
			focal = tde4.getLensFocalLength(i)
			focus = tde4.getLensFocus(i)
			ldm = tde4.getLensLDModel(i)
			mode = tde4.getLensDynamicDistortionMode(i)
       	for para in (getLDmodelParameterList(ldm)):
				tde4.setLensDynamicDistortionFlag(i, 1)
				tde4.setLensDynamicDistortionMode(i, mode)
				value = tde4.getLensLDAdjustableParameter(i, para, focal, focus)
				tde4.setLensDynamicDistortionFlag(i, 0)
				tde4.setLensDynamicDistortionMode(i, "DISTORTION_STATIC")
				tde4.setLensLDAdjustableParameter(i, para, focal, focus, value)
    	tde4.postQuestionRequester("Convert To Static Lens...", "Done.", "Ok")
else:
    tde4.postQuestionRequester("Convert To Static Lens...", "Error, Select a zoom lens to convert to a static lens first.", "Ok")
