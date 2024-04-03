#
#
# 3DE4.script.name:  3. Import Project Lens...
#
# 3DE4.script.version:  v1.0
#
# 3DE4.script.gui:  Main Window::DD_Setup
#
# 3DE4.script.comment:  Import Project Lens.
#
#
# Written by kwantae.kim(kkt0525@gmail.com)

import os
import DD_common

#
# functions...
def _import_lens_callback(requester, widget, action):

    if widget == "lensList":
        slLens = DD_common.find_list_item(requester, "lensList")
        lensList = DD_common.get_file_list(os.path.join(lensPath, slLens), "", "*.txt")

        tde4.removeAllListWidgetItems(requester, "res")
        for i in lensList:
            count = 0
            tde4.insertListWidgetItem(requester, "res", i, count)
            count += 1

def readCurve(f,curve):
    tde4.deleteAllCurveKeys(curve)
    string = f.readline()
    n = int(string)
    for i in range(n):
        string = f.readline()
        para = string.split()
        k = tde4.createCurveKey(curve,[float(para[0]),float(para[1])])
        tde4.setCurveKeyTangent1(curve,k,[float(para[2]),float(para[3])])
        tde4.setCurveKeyTangent2(curve,k,[float(para[4]),float(para[5])])
        tde4.setCurveKeyMode(curve,k,para[6])

#
# main...
if os.environ.has_key('show'):

    lensPath = os.path.join("/", "show", os.environ["show"], "asset", "global", "matchmove", "lens")
    lensList = DD_common.get_dir_list(lensPath)

    req = tde4.createCustomRequester()
    tde4.addTextFieldWidget(req, "show", "Show", os.environ["show"])
    tde4.setWidgetSensitiveFlag(req, "show", 0)

    tde4.addListWidget(req, "lensList", "Lens List", 0, 250)
    for i in lensList:
        if i[:1] != "_":
            count = 0
            tde4.insertListWidgetItem(req, "lensList", i, count)
            count += 1
    tde4.setWidgetCallbackFunction(req, "lensList", "_import_lens_callback")

    tde4.addListWidget(req, "res", "Resolution", 0, 100)
    tde4.insertListWidgetItem(req, "res", "Select lens first.", 0)
    tde4.setWidgetCallbackFunction(req, "res", "_import_lens_callback")

    ret = tde4.postCustomRequester(req, "Import Project Lens", 950, 0, "Import", "Cancel")

    if ret==1:
        sllLens = DD_common.find_list_item(req, "lensList")
        slRes = DD_common.find_list_item(req, "res")
        path = os.path.join("/", "show", os.environ["show"], "asset", "global", "matchmove", "lens", sllLens, slRes)
        #print path

        if path!=None:
            f    = open(path,"r")
            if not f.closed:
                l = tde4.createLens()

                name = f.readline()
                name = name.strip()
                tde4.setLensName(l,name)

                string = f.readline()
                para = string.split()
                fl = float(para[2])
                tde4.setLensFocalLength(l,fl)
                tde4.setLensFilmAspect(l,float(para[3]))
                tde4.setLensLensCenterX(l,float(para[4]))
                tde4.setLensLensCenterY(l,float(para[5]))
                tde4.setLensFBackWidth(l,float(para[0]))
                tde4.setLensPixelAspect(l,float(para[6]))
                tde4.setLensFBackHeight(l,float(para[1]))
            
                string = f.readline()
                string = string.strip()
                if string[0]=='0' or string[0]=='1': tde4.setLensDynamicDistortionFlag(l,int(string))
                else: tde4.setLensDynamicDistortionMode(l,string)
            
                model = f.readline()
                model = model.strip()
                tde4.setLensLDModel(l,model)
            
                dyndist = tde4.getLensDynamicDistortionMode(l)
                tde4.setLensDynamicDistortionMode(l,"DISTORTION_STATIC")
                para = ""
                while para!="<end_of_file>":
                    para = f.readline()
                    para = para.strip()
                
                    if para!="<end_of_file>":
                        string = f.readline()
                        string = string.strip()
                        d = float(string)
                        tde4.setLensLDAdjustableParameter(l,para,fl,100.0,d)
                        curve = tde4.getLensLDAdjustableParameterCurve(l,para)
                        readCurve(f,curve)
                tde4.setLensDynamicDistortionMode(l,dyndist)
            
                string = f.readline()
                string = string.strip()
                if string=="<begin_2dlut_samples>":
                    para = ""
                    while para!="<end_2dlut_samples>":
                        para = f.readline()
                        para = para.strip()
                        
                        if para!="<end_2dlut_samples>":
                            string = f.readline()
                            string = string.strip()
                            n = int(string)
                            for i in range(n):
                                string = f.readline()
                                string = string.split()
                                focal = float(string[0])
                                focus = float(string[1])
                                v = float(string[2])
                                tde4.addLens2DLUTSample(l,para,focal,focus,v)
            
                f.close

                cam_list = tde4.getCameraList(1)

                for cam in cam_list:
                    tde4.setCameraLens(cam, l)
            else:
                tde4.postQuestionRequester("Import Lens...","Error, couldn't open file.","Ok")    

else:
    tde4.postQuestionRequester("Import Project lens.", "There is no \"ENVKEY\"\nPlease open a project using \"Open Project\" script first.", "Ok")
