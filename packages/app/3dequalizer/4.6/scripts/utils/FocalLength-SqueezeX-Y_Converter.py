# 3DE4.script.name:		FocalLength-SqueezeX/Y Converter...
# 3DE4.script.version:	v1.53
# 3DE4.script.gui:Curve Editor::Edit
# 3DE4.script.comment:
# Patcha Saheb(patchasaheb@gmail.com) / Michael Karp (mckarp@aol.com)
#August 05 2015.
#
#pseudo code...
#reciprocal of x is 1/x
#
#Convert SqX to SqY and focal
##Multiply reciprocal of raw SqX with raw SqY
##Multiply reciprocal of raw SqX with raw SqX [sets to 1]
##Multiply instantaneous reciprocal of SqX with focal
#
#Convert SqY to SqX and focal
##Multiply reciprocal of raw SqY with raw SqX
##Multiply reciprocal of raw SqY with raw SqY [sets to 1]
##Multiply instantaneous reciprocal of SqY with focal
#
#Convert focal to sqx&sqy
##if focus curve is non-linear bake all distortion curve,else bake only 3channels
##Multiply reciprocal of Focal  * (SqX)
##Multiply reciprocal of Focal  * (SqY)
##Multiply reciprocal of Focal  * (Focal) [sets to 1]
##Multiply new sqx * (first frame's Focal value)
##Multiply new sqy * (first frame's Focal value)
##make focal curve static to first frame's focal value

#3channels = SqX & SqY & Focal

from vl_sdv import*
import os,sys
import math

def Bake_3_Channels():
	cam	= tde4.getCurrentCamera()
	current_frame = tde4.getCurrentFrame(cam)
	frames = tde4.getCameraNoFrames(cam)
	if cam!= None:
		f	= tde4.getCurrentFrame(cam)
		lens	= tde4.getCameraLens(cam)
	else:	lens	= None
	if lens!=None: 
		model = tde4.getLensLDModel(lens)
	else: model = None
	if model!=None:
		#bake only 3 channels...
		sq_x_list = []
		sq_y_list = []
		focal_list = []
		new_sq_x_list = []
		new_sq_y_list = []
		n	= tde4.getLDModelNoParameters(model)
		count = 0
		#get sqx curve and instantaneous sqx keys list...
		for i0 in range(n):
			count = count + 1
			if count == 12:
				para	= tde4.getLDModelParameterName(model,i0)
				sq_x_curve = tde4.getLensLDAdjustableParameterCurve(lens,para)
				for f in range(1,frames+1):	
					para	= tde4.getLDModelParameterName(model,i0)		
					focal = tde4.getCameraFocalLength(cam,f) * 10.0
					focus = tde4.getCameraFocus(cam,f)
					if tde4.getLDModelParameterType(model,para)=="LDP_DOUBLE_ADJUST":
						v	= tde4.getLensLDAdjustableParameter(lens,para,focal,focus)
						sq_x_list.append(v)
		#get sqy curve and instantaneous sqy keys list......
		count = 0	
		for i0 in range(n):
			count = count + 1
			if count == 13:
				para	= tde4.getLDModelParameterName(model,i0)
				sq_y_curve = tde4.getLensLDAdjustableParameterCurve(lens,para)
				for f in range(1,frames+1):	
					para	= tde4.getLDModelParameterName(model,i0)		
					focal = tde4.getCameraFocalLength(cam,f) * 10.0
					focus = tde4.getCameraFocus(cam,f)
					if tde4.getLDModelParameterType(model,para)=="LDP_DOUBLE_ADJUST":
						v	= tde4.getLensLDAdjustableParameter(lens,para,focal,focus)
						sq_y_list.append(v)
		#get focal data...
		for i in range(1,frames+1):
			focal = tde4.getCameraFocalLength(cam,i) * 10.0
			focal_list.append(1/focal)

		#get first frame's focal value
		focal = tde4.getCameraFocalLength(cam,1) * 10.0				

		for j in range(0,len(focal_list)):
			#Multiply reciprocal of focal with instantaneous sqx
			new_sq_x = focal_list[j] * sq_x_list[j]
			#Multiply new sqx with first frame's Focal value
			new_sq_x = new_sq_x * focal
			new_sq_x_list.append(new_sq_x)
			#Multiply reciprocal of focal with instantaneous sqy
			new_sq_y = focal_list[j] * sq_y_list[j]
			#Multiply new sqy with first frame's Focal value
			new_sq_y = new_sq_y * focal
			new_sq_y_list.append(new_sq_y)
		#put new focal values...
		for m in range(1,frames+1):
			tde4.setCameraFocalLengthMode(cam,"FOCAL_DYNAMIC")
			tde4.setCameraFocalLength(cam,m,(focal/10.0))
		#put new sqx and sqy values...
		count1 = 0
		for i1 in range(n):
			count1 = count1 + 1
			if count1 == 12:
				para	= tde4.getLDModelParameterName(model,i1)
				curve = tde4.getLensLDAdjustableParameterCurve(lens,para)
				kl = tde4.getCurveKeyList(curve,0)
				for key in kl:
					tde4.deleteCurveKey(curve,key)
				for f in range(1,frames+1):
					para	= tde4.getLDModelParameterName(model,i1)
					curve = tde4.getLensLDAdjustableParameterCurve(lens,para)
					k = tde4.createCurveKey(curve,[f,float(new_sq_x_list[f-1])])
					tde4.setCurveKeyMode(curve,k,"LINEAR")
			if count1 == 13:
				para	= tde4.getLDModelParameterName(model,i1)
				curve = tde4.getLensLDAdjustableParameterCurve(lens,para)
				kl = tde4.getCurveKeyList(curve,0)
				for key in kl:
					tde4.deleteCurveKey(curve,key)
				for f in range(1,frames+1):
					para	= tde4.getLDModelParameterName(model,i1)
					curve = tde4.getLensLDAdjustableParameterCurve(lens,para)
					k = tde4.createCurveKey(curve,[f,new_sq_y_list[f-1]])
					tde4.setCurveKeyMode(curve,k,"LINEAR")	

def Pre_Bake(req,widget,action):
	cam	= tde4.getCurrentCamera()
	current_frame = tde4.getCurrentFrame(cam)
	frames = tde4.getCameraNoFrames(cam)
	if cam!= None:
		f	= tde4.getCurrentFrame(cam)
		lens	= tde4.getCameraLens(cam)
	else:	lens	= None
	if lens!=None: 
		model = tde4.getLensLDModel(lens)
	else: model = None
	if model!=None:
		#check whether focus curve is linear or not...
		##if focus curve is linear bake only 3channels if not bake all curves and set focus curve to 1:1
		count = 0
		for frame in range(1,frames+1):
			focus = tde4.getCameraFocus(cam,frame)
			if int(focus+0.001) == frame:
				count = count + 1	
		if count == frames:
			Bake_3_Channels()
		else:
			#bake all curves, every frame...
			n	= tde4.getLDModelNoParameters(model)
			para_array = []
			for i0 in range(n):
				for frame in range(1,frames+1):
					tde4.setCurrentFrame(cam,frame)
					focal	= tde4.getCameraFocalLength(cam,tde4.getCurrentFrame(cam))
					focus = tde4.getCameraFocus(cam,tde4.getCurrentFrame(cam))
					para	= tde4.getLDModelParameterName(model,i0)
					if tde4.getLDModelParameterType(model,para)=="LDP_DOUBLE_ADJUST":
						v	= tde4.getLensLDAdjustableParameter(lens,para,focal,focus)
						para_array.append(v)
				#delete all curve keys...
				curve = tde4.getLensLDAdjustableParameterCurve(lens,para)
				tde4.deleteAllCurveKeys(curve)
			#create new curve keys...
			new_paralist = para_array
			for i1 in range(n):
				i = 1
				for n in new_paralist:
					tde4.setCurrentFrame(cam,i)
					para1	= tde4.getLDModelParameterName(model,i1)
					curve = tde4.getLensLDAdjustableParameterCurve(lens,para1)
					tde4.createCurveKey(curve,[i,n])
					i = i + 1
					if i == frames + 1:
						del new_paralist[0:frames]
						break
					#put all keys mode to linear...
					para2	= tde4.getLDModelParameterName(model,i1)
					curve = tde4.getLensLDAdjustableParameterCurve(lens,para2)
					kl	= tde4.getCurveKeyList(curve,0)
					for k in kl:
						tde4.setCurveKeyMode(curve,k,"LINEAR")
			#reset focus curve to 1:1...
			focus_curve = tde4.getCameraFocusCurve(cam)
			tde4.deleteAllCurveKeys(focus_curve)
			tde4.createCurveKey(focus_curve,[1,1])
			tde4.createCurveKey(focus_curve,[frames,frames])
			focus_kl = tde4.getCurveKeyList(focus_curve,0)
			for focus_k in focus_kl:
				tde4.setCurveKeyMode(focus_curve,focus_k,"LINEAR")	
			tde4.setCurrentFrame(cam,current_frame)
			#bake 3 channels...
			Bake_3_Channels()

def Pickup():
	pg = tde4.getCurrentPGroup()
	cam = tde4.getCurrentCamera()
	frames = tde4.getCameraNoFrames(cam)
	if cam!=None:
		frame	= tde4.getCurrentFrame(cam)
		lens	= tde4.getCameraLens(cam)
	else:	lens	= None
	if lens!=None: model = tde4.getLensLDModel(lens)
	else: model = None
	if model!=None:	
		#create emphty lists...
		focal_list = []
		raw_sq_x_ext_keys_x_list = []
		raw_sq_y_ext_keys_x_list = []
		#filling lists...
		n	= tde4.getLDModelNoParameters(model)
		count = 0
		#get sqx curve and instantaneous sqx keys list...
		for i0 in range(n):
			count = count + 1
			if count == 12:
				para	= tde4.getLDModelParameterName(model,i0)
				sq_x_curve = tde4.getLensLDAdjustableParameterCurve(lens,para)
		#get sqy curve and instantaneous sqy keys list......
		count = 0	
		for i0 in range(n):
			count = count + 1
			if count == 13:
				para	= tde4.getLDModelParameterName(model,i0)
				sq_y_curve = tde4.getLensLDAdjustableParameterCurve(lens,para)
		#get raw sqx existing keys...		
		kl = tde4.getCurveKeyList(sq_x_curve,0)
		for k in kl:
			key = tde4.getCurveKeyPosition(sq_x_curve,k)
			raw_sq_x_ext_keys_x_list.append(key[0])
		#get raw sqy existing keys...		
		kl = tde4.getCurveKeyList(sq_y_curve,0)
		for k in kl:
			key = tde4.getCurveKeyPosition(sq_y_curve,k)
			raw_sq_y_ext_keys_x_list.append(key[0])
		#get focal data...
		for i in range(1,frames+1):
			focal = tde4.getCameraFocalLength(cam,i) * 10.0
			focal_list.append(focal)
		#pickups on sqy curve...
		for j in raw_sq_x_ext_keys_x_list:
			k = tde4.evaluateCurve(sq_y_curve,j)
			tde4.createCurveKey(sq_y_curve,[j,k])
		#pickups on sqx curve...
		for j in raw_sq_y_ext_keys_x_list:
			k = tde4.evaluateCurve(sq_x_curve,j)
			tde4.createCurveKey(sq_x_curve,[j,k])
		#pickup in between integer keys on sqx & sqy curves...
		##update raw sqx & raw sqy keys data...
		###remove previous keys data to update new existing keys data...
		raw_sq_x_ext_keys_x_list = []
		raw_sq_y_ext_keys_x_list = []
		#sqx curve existing keys data...
		kl = tde4.getCurveKeyList(sq_x_curve,0)
		for k in kl:
			key = tde4.getCurveKeyPosition(sq_x_curve,k)
			raw_sq_x_ext_keys_x_list.append(key[0])
		#sqy curve existing keys data...
		kl = tde4.getCurveKeyList(sq_y_curve,0)
		for k in kl:
			key = tde4.getCurveKeyPosition(sq_y_curve,k)
			raw_sq_y_ext_keys_x_list.append(key[0])
		#integer pickups...
		tde4.postProgressRequesterAndContinue(window_title,"Pickingup keys...Step 1/5",int(raw_sq_x_ext_keys_x_list[-1]),"Cancel")
		null = 0	
		count = 0	
		for i in range(len(raw_sq_x_ext_keys_x_list)):
			null = null + count
			count = count + 1
			if count != len(raw_sq_x_ext_keys_x_list):
				k = count
				for j in range(int(raw_sq_x_ext_keys_x_list[i]),int(raw_sq_x_ext_keys_x_list[i+1])):
					y_pos = tde4.evaluateCurve(sq_x_curve,j)
					key = tde4.createCurveKey(sq_x_curve,[j,y_pos])
					tde4.setCurveKeyMode(sq_x_curve,key,"LINEAR")
					y_pos = tde4.evaluateCurve(sq_y_curve,j)
					key = tde4.createCurveKey(sq_y_curve,[j,y_pos])
					tde4.setCurveKeyMode(sq_y_curve,key,"LINEAR")
					null = null + 1
					if null % 10 == 0:
						tde4.updateProgressRequester(null,"Pickingup keys...Step 1/5")
		tde4.unpostProgressRequester()
	
def Convert_SqX_And_SqY_to_Focal(req,widget,action):
	pg = tde4.getCurrentPGroup()
	cam = tde4.getCurrentCamera()
	frames = tde4.getCameraNoFrames(cam)
	if cam!=None:
		frame	= tde4.getCurrentFrame(cam)
		lens	= tde4.getCameraLens(cam)
	else:	lens	= None
	if lens!=None: model = tde4.getLensLDModel(lens)
	else: model = None
	if model!=None:	
		Pickup()
		new_sq_x_list = []
		new_sq_y_list = []
		focal_list = []
		baked_sq_x_keys_list = []
		baked_sq_y_keys_list = []
		n	= tde4.getLDModelNoParameters(model)
		count = 0
		#get sqx curve instantaneous sqx keys list...
		for i0 in range(n):
			count = count + 1
			if count == 12:
				para	= tde4.getLDModelParameterName(model,i0)
				sq_x_curve = tde4.getLensLDAdjustableParameterCurve(lens,para)
				for f in range(1,frames+1):	
					para	= tde4.getLDModelParameterName(model,i0)		
					focal = tde4.getCameraFocalLength(cam,f) * 10.0
					focus = tde4.getCameraFocus(cam,f)
					if tde4.getLDModelParameterType(model,para)=="LDP_DOUBLE_ADJUST":
						v	= tde4.getLensLDAdjustableParameter(lens,para,focal,focus)
						baked_sq_x_keys_list.append(v)
		#get sqy curve and instantaneous sqy keys list......
		count = 0	
		for i0 in range(n):
			count = count + 1
			if count == 13:
				para	= tde4.getLDModelParameterName(model,i0)
				sq_y_curve = tde4.getLensLDAdjustableParameterCurve(lens,para)
				for f in range(1,frames+1):	
					para	= tde4.getLDModelParameterName(model,i0)		
					focal = tde4.getCameraFocalLength(cam,f) * 10.0
					focus = tde4.getCameraFocus(cam,f)
					if tde4.getLDModelParameterType(model,para)=="LDP_DOUBLE_ADJUST":
						v	= tde4.getLensLDAdjustableParameter(lens,para,focal,focus)
						baked_sq_y_keys_list.append(v)
		#get focal data...
		for i in range(1,frames+1):
			focal = tde4.getCameraFocalLength(cam,i) * 10.0
			focal_list.append(focal)

		if widget == "sqx_to_focal":			
			#multiply reciprocal of raw sqx with raw sqy...
			sq_x_keys_list = tde4.getCurveKeyList(sq_x_curve,0)
			sq_y_keys_list = tde4.getCurveKeyList(sq_y_curve,0)
			tde4.postProgressRequesterAndContinue(window_title,"Calculating keys data...Step 2/5",len(sq_y_keys_list),"Cancel")
			null = 1 
			for i in sq_x_keys_list:
				if null % 10 == 0:
					tde4.updateProgressRequester(null,"Calculating keys data...Step 2/5")
				sq_x_key = tde4.getCurveKeyPosition(sq_x_curve,i)
				sq_y_key = tde4.evaluateCurve(sq_y_curve,sq_x_key[0])
				new_sq_y = (1/sq_x_key[1]) * sq_y_key
				new_sq_y_list.append(new_sq_y)
				null = null + 1
			tde4.unpostProgressRequester()	
			#write new sqy values...
			tde4.postProgressRequesterAndContinue(window_title,"Writing new Squeeze Y keys...Step 3/5",len(sq_y_keys_list),"Cancel")
			count = 0
			null = 1
			for i in sq_y_keys_list:
				key = tde4.getCurveKeyPosition(sq_y_curve,i)
				tde4.setCurveKeyPosition(sq_y_curve,i,[key[0],new_sq_y_list[count]])
				count = count + 1
				if null % 10 == 0:
					tde4.updateProgressRequester(null,"Writing new Squeeze Y keys...Step 3/5")	
				null = null + 1		
			tde4.unpostProgressRequester()
			#set sqx to 1...
			tde4.postProgressRequesterAndContinue(window_title,"Writing new Squeeze X keys...Step 4/5",len(sq_x_keys_list),"Cancel")
			null = 1
			for i in sq_x_keys_list:
				key = tde4.getCurveKeyPosition(sq_x_curve,i)
				tde4.setCurveKeyPosition(sq_x_curve,i,[key[0],1])
				if null % 10 == 0:
					tde4.updateProgressRequester(null,"Writing new Squeeze X keys...Step 4/5")
				null = null + 1
			tde4.unpostProgressRequester()
			#multiply instantaneous reciprocal of sqx with focal...
			tde4.postProgressRequesterAndContinue(window_title,"Writing new Focal keys...Step 5/5",len(baked_sq_x_keys_list),"Cancel")
			null = 1
			for i in range(len(baked_sq_x_keys_list)):
				new_focal = (1/baked_sq_x_keys_list[i]) * focal_list[i]
				tde4.setCameraFocalLengthMode(cam,"FOCAL_DYNAMIC")
				tde4.setCameraFocalLength(cam,i+1,(new_focal/10.0))
				if null % 10 == 0:
					tde4.updateProgressRequester(null,"Writing new Focal keys...Step 5/5")
				null = null + 1
			tde4.unpostProgressRequester()

	if widget == "sqy_to_focal":
			#multiply reciprocal of raw sqy with raw sqx...
			sq_x_keys_list = tde4.getCurveKeyList(sq_x_curve,0)
			sq_y_keys_list = tde4.getCurveKeyList(sq_y_curve,0)
			tde4.postProgressRequesterAndContinue(window_title,"Calculating keys data...Step 2/5",len(sq_x_keys_list),"Cancel")
			null = 1 
			for i in sq_x_keys_list:
				if null % 10 == 0:
					tde4.updateProgressRequester(null,"Calculating keys data...Step 2/5")
				sq_x_key = tde4.getCurveKeyPosition(sq_x_curve,i)
				sq_y_key = tde4.evaluateCurve(sq_y_curve,sq_x_key[0])
				new_sq_x = (1/sq_y_key) * sq_x_key[1]
				new_sq_x_list.append(new_sq_x)
				null = null + 1
			tde4.unpostProgressRequester()
			#write new sqx values...
			tde4.postProgressRequesterAndContinue(window_title,"Writing new Squeeze X keys...Step 3/5",len(sq_x_keys_list),"Cancel")
			count = 0
			null = 1
			for i in sq_x_keys_list:
				key = tde4.getCurveKeyPosition(sq_x_curve,i)
				tde4.setCurveKeyPosition(sq_x_curve,i,[key[0],new_sq_x_list[count]])
				count = count + 1
				if null % 10 == 0:
					tde4.updateProgressRequester(null,"Writing new Squeeze X keys...Step 3/5")	
				null = null + 1		
			tde4.unpostProgressRequester()
			#set sqy to 1...
			tde4.postProgressRequesterAndContinue(window_title,"Writing new Squeeze Y keys...Step 4/5",len(sq_y_keys_list),"Cancel")
			null = 1
			for i in sq_y_keys_list:
				key = tde4.getCurveKeyPosition(sq_y_curve,i)
				tde4.setCurveKeyPosition(sq_y_curve,i,[key[0],1])
				if null % 10 == 0:
					tde4.updateProgressRequester(null,"Writing new Squeeze X keys...Step 4/5")
				null = null + 1
			tde4.unpostProgressRequester()
			#multiply instantaneous reciprocal of sqy with focal...
			tde4.postProgressRequesterAndContinue(window_title,"Writing new Focal keys...Step 5/5",len(baked_sq_y_keys_list),"Cancel")
			null = 1
			for i in range(len(baked_sq_y_keys_list)):
				new_focal = (1/baked_sq_y_keys_list[i]) * focal_list[i]
				tde4.setCameraFocalLengthMode(cam,"FOCAL_DYNAMIC")
				tde4.setCameraFocalLength(cam,i+1,(new_focal/10.0))
				if null % 10 == 0:
					tde4.updateProgressRequester(null,"Writing new Focal keys...Step 5/5")
				null = null + 1
			tde4.unpostProgressRequester()
					
def Transform_Curves(req,widget,action):
	pg = tde4.getCurrentPGroup()
	cam = tde4.getCurrentCamera()
	frames = tde4.getCameraNoFrames(cam)
	if cam!=None:
		frame	= tde4.getCurrentFrame(cam)
		lens	= tde4.getCameraLens(cam)
	else:	lens	= None
	if lens!=None: model = tde4.getLensLDModel(lens)
	else: model = None
	if model!=None:	
		if widget == "factor_slider":		
			factor_slider = tde4.getWidgetValue(req,"factor_slider")
			tde4.setWidgetValue(req,"factor_text",str(factor_slider))
			
		if widget == "factor_text":				
			factor_text = tde4.getWidgetValue(req,"factor_text")
			factor_slider = tde4.getWidgetValue(req,"factor_slider")
			if float(factor_text) >= 0.001:
				if float(factor_text) > 2.0:
					tde4.setWidgetSensitiveFlag(req,"factor_slider",0)
				else:
					tde4.setWidgetValue(req,"factor_slider",str(factor_text))
					tde4.setWidgetSensitiveFlag(req,"factor_slider",1)
			else:
				tde4.setWidgetValue(req,"factor_text","1.1")
				tde4.setWidgetValue(req,"factor_slider","1.1")
				tde4.postQuestionRequester(window_title,"Error, minimum value is 0.001","OK")
				
		v = tde4.getWidgetValue(req,"factor_text")		
		if widget == "apply":
			#multiply factor value with focal curve...
			focal_mode = tde4.getCameraFocalLengthMode(cam)
			if not focal_mode == "FOCAL_DYNAMIC":
				lens_focal = tde4.getLensFocalLength(lens)
				new_focal = lens_focal * float(v)
				tde4.setLensFocalLength(lens,new_focal)
			if focal_mode == "FOCAL_DYNAMIC":
				zoom_curve = tde4.getCameraZoomCurve(cam)
				key_list = tde4.getCurveKeyList(zoom_curve)
				for key in key_list:
					key_pos = tde4.getCurveKeyPosition(zoom_curve,key)
					new_focal = key_pos[1] * float(v)
					tde4.setCurveKeyPosition(zoom_curve,key,[key_pos[0],new_focal])
			n	= tde4.getLDModelNoParameters(model)
			count = 0
			for i0 in range(n):
				count = count + 1
				#multiply raw sqx keys with factor value...
				if count == 12:
					para	= tde4.getLDModelParameterName(model,i0)
					curve = tde4.getLensLDAdjustableParameterCurve(lens,para)
					if tde4.getLDModelParameterType(model,para)=="LDP_DOUBLE_ADJUST":
						kl = tde4.getCurveKeyList(curve,0)
						for key in kl:
							pos = tde4.getCurveKeyPosition(curve,key)
							f = float(pos[1]) * float(v)
							tde4.setCurveKeyPosition(curve,key,[pos[0],f])
				#multiply raw sqy keys with factor value...
				if count == 13:
					para	= tde4.getLDModelParameterName(model,i0)
					curve = tde4.getLensLDAdjustableParameterCurve(lens,para)
					if tde4.getLDModelParameterType(model,para)=="LDP_DOUBLE_ADJUST":
						kl = tde4.getCurveKeyList(curve,0)
						for key in kl:
							pos = tde4.getCurveKeyPosition(curve,key)
							f = float(pos[1]) * float(v)
							tde4.setCurveKeyPosition(curve,key,[pos[0],f])

#build GUI...
window_title = "Patcha Focal-SqueezeX/Y Converter v1.53"
req	= tde4.createCustomRequester()
#add convert sqx to focal button...
tde4.addButtonWidget(req,"sqx_to_focal","Convert SqueezeX to FocalLength & SqueezeY",70,10)
tde4.setWidgetAttachModes(req,"sqx_to_focal","ATTACH_POSITION","ATTACH_POSITION","ATTACH_WINDOW","ATTACH_AS_IS")
tde4.setWidgetOffsets(req,"sqx_to_focal",7,93,10,0)
#add convert sqy to focal button...
tde4.addButtonWidget(req,"sqy_to_focal","Convert SqueezeY to FocalLength & SqueezeX",70,10)
tde4.setWidgetAttachModes(req,"sqy_to_focal","ATTACH_POSITION","ATTACH_POSITION","ATTACH_WINDOW","ATTACH_AS_IS")
tde4.setWidgetOffsets(req,"sqy_to_focal",7,93,40,0)
#add convert focal to sqx & sqy button...
tde4.addButtonWidget(req,"focal_to_sqxy","Convert FocalLength to SqueezeX & SqueezeY",70,10)
tde4.setWidgetAttachModes(req,"focal_to_sqxy","ATTACH_POSITION","ATTACH_POSITION","ATTACH_WINDOW","ATTACH_AS_IS")
tde4.setWidgetOffsets(req,"focal_to_sqxy",7,93,70,0)
#add separator...
tde4.addSeparatorWidget(req,"sep1")
tde4.setWidgetAttachModes(req,"sep1","ATTACH_POSITION","ATTACH_POSITION","ATTACH_WINDOW","ATTACH_AS_IS")
tde4.setWidgetOffsets(req,"sep1",2,98,90,0)
#add apply button widget...
tde4.addButtonWidget(req,"apply","Transform SqX/Y and Focal curves",70,10)
tde4.setWidgetAttachModes(req,"apply","ATTACH_POSITION","ATTACH_POSITION","ATTACH_WINDOW","ATTACH_AS_IS")
tde4.setWidgetOffsets(req,"apply",15,85,140,0)
#add factor slider widget...
tde4.addScaleWidget(req,"factor_slider","factor","DOUBLE",0.001,2.0,1.1)
tde4.setWidgetAttachModes(req,"factor_slider","ATTACH_POSITION","ATTACH_POSITION","ATTACH_WINDOW","ATTACH_AS_IS")
tde4.setWidgetOffsets(req,"factor_slider",15,80,110,0)
#add factor text fiels widget...
tde4.addTextFieldWidget(req,"factor_text"," ","1.1")
tde4.setWidgetAttachModes(req,"factor_text","ATTACH_POSITION","ATTACH_POSITION","ATTACH_WINDOW","ATTACH_AS_IS")
tde4.setWidgetOffsets(req,"factor_text",82,98,110,0)
#callbacks...
tde4.setWidgetCallbackFunction(req,"sqx_to_focal","Convert_SqX_And_SqY_to_Focal")
tde4.setWidgetCallbackFunction(req,"sqy_to_focal","Convert_SqX_And_SqY_to_Focal")
tde4.setWidgetCallbackFunction(req,"focal_to_sqxy","Pre_Bake")
tde4.setWidgetCallbackFunction(req,"factor_slider","Transform_Curves")
tde4.setWidgetCallbackFunction(req,"factor_text","Transform_Curves")
tde4.setWidgetCallbackFunction(req,"apply","Transform_Curves")
tde4.postCustomRequesterAndContinue(req,window_title,420,170)