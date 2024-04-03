# 3DE4.script.name:    View OpenEXR Metadata...
# 3DE4.script.version:    v1.0
# 3DE4.script.gui.config_menus: False
# 3DE4.script.gui:    Main Window::Dexter
# Patcha Saheb(patchasaheb@gmail.com)
# 15 May 2018(Montreal)
# original Code: Patcha OpenEXR Metadata Extractor v1.0

import sys
import xml.etree.ElementTree as et


def Cursor_Update(req):
    global curve_req
    cam = tde4.getCurrentCamera()
    frame = tde4.getCurrentFrame(cam)
    tde4.setCurveAreaWidgetCursorPosition(curve_req,"curves_area",frame,1)

def Curvearea_Callback(req,widget,action):
    global curve_req
    cam = tde4.getCurrentCamera()
    if action==3 or action==2:
        f = tde4.getCurveAreaWidgetCursorPosition(curve_req,"curves_area")
        n = tde4.getCameraNoFrames(cam)
        if f < 1: f = 1
        if f > n: f = 1
        tde4.setCurrentFrame(cam,int(f))

def attributeValues():
    l = []
    null = 0
    items = tde4.getListWidgetNoItems(req,"data_list_widget")
    for item in range(0,items):
        if tde4.getListWidgetItemSelectionFlag(req,"data_list_widget",item) == 1:
            null = 1
            item_label = tde4.getListWidgetItemLabel(req,"data_list_widget",item)
            item_label = item_label.split()[0]
            l.append(item_label)
            break

    if null == 1:
        for frame in range(1,frames+1):
            path = tde4.getCameraFrameFilepath(cam,frame)
            try:
                xml = tde4.convertOpenEXRMetaDataToXML(path)
                xml_clean = "".join(c for c in xml if ord(c) < 128)
            except:
                raise error("File '" + path + "' doesn't seem to be an EXR file.")
            root = et.fromstring(xml_clean)
            for a in root.findall("attribute"):
                name = a.find("name").text
                value = a.find("value").text
                if item_label == name:
                    l.append(value.split()[0])
    return list(l)

def viewCurve(req,widget,action):
    global curve_req
    pg = tde4.getCurrentPGroup()
    cam = tde4.getCurrentCamera()
    frames = tde4.getCameraNoFrames(cam)
    frame_offset = tde4.getCameraFrameOffset(cam)
    #create curve...
    global curve
    curve = tde4.createCurve()
    curve_y_values_list = attributeValues()
    if len(curve_y_values_list) > 1:
        tde4.setWidgetValue(curve_req,"attr_name_txt",str(curve_y_values_list[0]))
        tde4.postProgressRequesterAndContinue(window_title, "Extracting Attribute values from Metadata...", frames,"Ok")
        for frame in range(1,frames+1):
            key = tde4.createCurveKey(curve,[frame,float(curve_y_values_list[frame])])
            tde4.setCurveKeyFixedXFlag(curve,key,1)
            tde4.setCurveKeyMode(curve,key,"LINEAR")
            #if frame % 10000 == 0:
            tde4.updateProgressRequester(frame,"Extracting Attribute values from Metadata...")
        tde4.detachCurveAreaWidgetAllCurves(curve_req,"curves_area")
        tde4.attachCurveAreaWidgetCurve(curve_req,"curves_area",curve,1.0,0.4,0.4,1)
        tde4.setCurveAreaWidgetXOffset(curve_req,"curves_area",frame_offset-1)
        curve_y_values_list.pop(0)
        tde4.setCurveAreaWidgetDimensions(curve_req,"curves_area",1,frames,float(min(curve_y_values_list,key=float))-1.0,float(max(curve_y_values_list,key=float))+1.0)
        tde4.setCurveAreaWidgetCursorPosition(curve_req,"curves_area",tde4.getCurrentFrame(cam),1)
    else:
        tde4.postQuestionRequester(window_title,'No attribute was selected.','OK')


def viewAll(req,widget,action):
    global curve
    key_data = []
    key_list = tde4.getCurveKeyList(curve,0)
    for key in key_list:
        pos2d = tde4.getCurveKeyPosition(curve,key)
        key_data.append(pos2d[1])
        if len(key_data) >= 1:
            tde4.setCurveAreaWidgetDimensions(curve_req,"curves_area",1.0,frames,min(key_data)-0.1,max(key_data)+0.1)
        else:
            tde4.setCurveAreaWidgetDimensions(curve_req,"curves_area",1.0,frames,-0.2,1.0)


def writeData(req,widget,action):
    global curve
    #try:
    current_curve = tde4.getFirstCurrentCurve()
    delay_value = int(tde4.getWidgetValue(curve_req,"frame_delay"))
    key_data = []
    key_list = tde4.getCurveKeyList(curve,0)
    if len(key_list) > 0:
        for key in key_list:
            pos2d = tde4.getCurveKeyPosition(curve,key)
            key_data.append(pos2d[1])
        key_data = key_data[delay_value:]

        pg = tde4.getCurrentPGroup()
        cam = tde4.getCurrentCamera()
        frames = tde4.getCameraNoFrames(cam)

        current_curve_key_list = tde4.getCurveKeyList(current_curve,0)
        tde4.deleteAllCurveKeys(current_curve)
        for j in range(len(key_data)):
            key = tde4.createCurveKey(current_curve,[j+1,key_data[j]])
            tde4.setCurveKeyMode(current_curve,key,"LINEAR")
            tde4.setCurveKeyFixedXFlag(current_curve,key,1)
            tde4.copyPGroupEditCurvesToFilteredCurves(pg,cam)


















































def displayCurveEditor(req,widget,action):
    global curve_req
    curve_req = tde4.createCustomRequester()

    #add curve are widget...
    tde4.addCurveAreaWidget(curve_req,"curves_area","")
    tde4.setWidgetAttachModes(curve_req,"curves_area","ATTACH_WINDOW","ATTACH_WINDOW","ATTACH_AS_IS","ATTACH_WINDOW")
    tde4.setWidgetOffsets(curve_req,"curves_area",10,10,30,30)

    #display attribute name text field widget...
    tde4.addTextFieldWidget(curve_req,"attr_name_txt","Attribute Name","")
    tde4.setWidgetSensitiveFlag(curve_req,"attr_name_txt",0)
    tde4.setWidgetAttachModes(curve_req,"attr_name_txt","ATTACH_WINDOW","ATTACH_WINDOW","ATTACH_WINDOW","ATTACH_AS_IS")
    tde4.setWidgetOffsets(curve_req,"attr_name_txt",120,250,5,0)

    #add show curve button...
    tde4.addButtonWidget(curve_req,"view_curve","Show Curve",120,5)
    tde4.setWidgetLinks(curve_req,"view_curve","","","curves_area","")
    tde4.setWidgetOffsets(curve_req,"view_curve",20,0,5,0)

    #add frame delay button...
    tde4.addTextFieldWidget(curve_req,"frame_delay","Frame delay in time(Offset)","1")
    tde4.setWidgetLinks(curve_req,"frame_delay","view_curve","","curves_area","")
    tde4.setWidgetOffsets(curve_req,"frame_delay",35,600,5,0)
    tde4.setWidgetSize(curve_req,"frame_delay",100,20)

    #add write button...
    tde4.addButtonWidget(curve_req,"write_btn","Write to current curve",100,5)
    tde4.setWidgetAttachModes(curve_req,"write_btn","ATTACH_WIDGET","ATTACH_WINDOW","ATTACH_OPPOSITE_WIDGET","ATTACH_NONE")
    tde4.setWidgetLinks(curve_req,"write_btn","frame_delay","","frame_delay","")
    tde4.setWidgetOffsets(curve_req,"write_btn",80,250,00,40)

    #add view all button...
    tde4.addButtonWidget(curve_req,"view_all_btn","View All",50,5)
    tde4.setWidgetAttachModes(curve_req,"view_all_btn","ATTACH_WIDGET","ATTACH_WINDOW","ATTACH_OPPOSITE_WIDGET","ATTACH_NONE")
    tde4.setWidgetLinks(curve_req,"view_all_btn","write_btn","","write_btn","")
    tde4.setWidgetOffsets(curve_req,"view_all_btn",110,10,00,40)



    tde4.setWidgetCallbackFunction(curve_req,"curves_area","Curvearea_Callback")
    tde4.setWidgetCallbackFunction(curve_req,"view_curve","viewCurve")
    tde4.setWidgetCallbackFunction(curve_req,"view_all_btn","viewAll")
    tde4.setWidgetCallbackFunction(curve_req,"write_btn","writeData")

    tde4.postCustomRequesterAndContinue(curve_req,window_title,1100,700,"Cursor_Update")
    tde4.updateGUI()
    tde4.setCurveAreaWidgetDimensions(curve_req,"curves_area",1.0,frames,-0.2,1.0)


def Filter(req,widget,action):
    pg = tde4.getCurrentPGroup()
    cam = tde4.getCurrentCamera()
    frame = tde4.getCurrentFrame(cam)
    frames = tde4.getCameraNoFrames(cam)
    filter_value = str(tde4.getWidgetValue(req,"filter_txt"))
    filter_value = filter_value.lower()
    tde4.removeAllListWidgetItems(req,"data_list_widget")
    path = tde4.getCameraFrameFilepath(cam,frame)
    try:
        xml = tde4.convertOpenEXRMetaDataToXML(path)
        xml_clean = "".join(c for c in xml if ord(c) < 128)
    except:
        raise error("File '" + path + "' doesn't seem to be an EXR file.")
    root = et.fromstring(xml_clean)
    if filter_value != "none":
        for a in root.findall("attribute"):
            name = a.find("name").text
            value = a.find("value").text
            item_label = str(name) + "  " + "==" + "  " + str(value)
            #item_label = item_label.lower()
            if filter_value in item_label.lower():
                tde4.insertListWidgetItem(req,"data_list_widget",item_label)
    if filter_value == "none":
        for a in root.findall("attribute"):
            name = a.find("name").text
            value = a.find("value").text
            tde4.insertListWidgetItem(req,"data_list_widget",(str(name) + "  " + "==" + "  " + str(value)))



pg = tde4.getCurrentPGroup()
cam = tde4.getCurrentCamera()
frame = tde4.getCurrentFrame(cam)
frames = tde4.getCameraNoFrames(cam)

window_title = "View OpenEXR Metadata"
req = tde4.createCustomRequester()

#add label widget...
tde4.addLabelWidget(req,"label","Attribute Name  ==  Attribute Value","ALIGN_LABEL_CENTER")
tde4.setWidgetAttachModes(req,"label","ATTACH_POSITION","ATTACH_POSITION","ATTACH_WINDOW","ATTACH_AS_IS")
tde4.setWidgetOffsets(req,"label",2,98,5,10)

#add list widget...
tde4.addListWidget(req,"data_list_widget", "", 1,490)
tde4.setWidgetAttachModes(req,"data_list_widget","ATTACH_POSITION","ATTACH_POSITION","ATTACH_WINDOW","ATTACH_WINDOW")
tde4.setWidgetOffsets(req,"data_list_widget",2,98,30,90)

#add filter text field widget...
tde4.addTextFieldWidget(req,"filter_txt","Filter")
tde4.setWidgetOffsets(req,"filter_txt",10,10,15,10)

#add display curve editor button widget...
"""
tde4.addButtonWidget(req,"curve_editor_btn","Display selected Attribute values in Curve Editor")
tde4.setWidgetAttachModes(req,"curve_editor_btn","ATTACH_WINDOW","ATTACH_WINDOW","ATTACH_AS_IS","ATTACH_AS_IS")
tde4.setWidgetOffsets(req,"curve_editor_btn",13,13,20,10)
"""

#for frame in range(1,frames+1):
tde4.removeAllListWidgetItems(req,"data_list_widget")
path = tde4.getCameraFrameFilepath(cam,frame)
try:
    xml = tde4.convertOpenEXRMetaDataToXML(path)
    xml_clean = "".join(c for c in xml if ord(c) < 128)
except:
    print("File '" + path + "' doesn't seem to be an EXR file.")

root = et.fromstring(xml_clean)
for a in root.findall("attribute"):
    name = a.find("name").text
    value = a.find("value").text
    tde4.insertListWidgetItem(req,"data_list_widget",(str(name) + "  " + "==" + "  " + str(value)))




tde4.setWidgetCallbackFunction(req,"filter_txt","Filter")
#tde4.setWidgetCallbackFunction(req,"curve_editor_btn","displayCurveEditor")



tde4.postCustomRequesterAndContinue(req,window_title,490,700)
