#
#
# 3DE4.script.name:	Save out Rendered Frames for Dexter...
#
# 3DE4.script.version:	v1.2.1
#
# 3DE4.script.gui:	Overview Controls::Render
#
# 3DE4.script.comment:	Renders out a sequence of frames according to the current view settings.
#
#

# v1.2.1 - overscan 1.08 and check even number
# functions...

import os
import DD_common

def prepareImagePath(path,frame):
	path	= path.replace("\\","/")
	i	= 0
	n	= 0
	i0	= -1
	while(i<len(path)):
		if path[i]=='#': n += 1
		if n==1 and i0==-1: i0 = i
		i	+= 1
	if i0!=-1:
		fstring		= "%%s%%0%dd%%s"%(n)
		path2		= fstring%(path[0:i0],frame,path[i0+n:len(path)])
		path		= path2
	return path
	

def _renderOutCallback(requester,widget,action):
	c	= tde4.getCurrentCamera()
	
	if widget=="res_mode" or widget=="overscan":
		w0		= tde4.getCameraImageWidth(c)
		h0		= tde4.getCameraImageHeight(c)
		mode		= tde4.getWidgetValue(requester,"res_mode")
		overscan	= tde4.getWidgetValue(requester,"overscan")
		if mode==1:
			tde4.setWidgetSensitiveFlag(requester,"width",0)
			w	= w0
			h	= h0
		if mode==2:
			tde4.setWidgetSensitiveFlag(requester,"width",0)
			w	= w0/2
			h	= h0/2
		if mode==3:
			tde4.setWidgetSensitiveFlag(requester,"width",0)
			w	= w0/4
			h	= h0/4
		if overscan==1:
			w	= int(float(w)*1.08)
			h	= int(float(h)*1.08)
			if w%2==1:
				w += 1
			if h%2==1:
				h += 1
		tde4.setWidgetValue(requester,"width",str(w))
		tde4.setWidgetValue(requester,"height",str(h))

	if widget=="width":
		w0	= tde4.getCameraImageWidth(c)
		h0	= tde4.getCameraImageHeight(c)
		w	= int(tde4.getWidgetValue(requester,"width"))
		h	= int(float(w)*float(h0)/float(w0))
		tde4.setWidgetValue(requester,"height",str(h))

proj_path = os.path.split(tde4.getProjectPath())[0]
proj_file = os.path.split(tde4.getProjectPath())[1]
proj_file = os.path.splitext(proj_file)[0]

c	= tde4.getCurrentCamera()

if c!=None:
	splited_path = proj_path.split("/")	#	['', 'show', 'tisf', 'shot', 'DLH', 'DLH_0570', 'matchmove', 'dev', '3de']
	tmp_path = splited_path[:-1]	#	['', 'show', 'tisf', 'shot', 'DLH', 'DLH_0570', 'matchmove', 'dev']
	tmp_path = "/".join(tmp_path)
	new_path = os.path.join(tmp_path, "preview", proj_file, tde4.getCameraName(c)+".####.jpg")

	_render_out_dexter_req		= tde4.createCustomRequester()
	tde4.addFileWidget(_render_out_dexter_req,"file","Output Path","*",new_path)
	tde4.addToggleWidget(_render_out_dexter_req,"delete_cache","Clear Render Cache",1)
	tde4.addOptionMenuWidget(_render_out_dexter_req,"res_mode","Render Resolution","Full 1:1","Half 1:2","Quarter 1:4")
	tde4.setWidgetCallbackFunction(_render_out_dexter_req,"res_mode","_renderOutCallback")
	tde4.setWidgetAttachModes(_render_out_dexter_req,"res_mode","ATTACH_AS_IS","ATTACH_NONE","ATTACH_AS_IS","ATTACH_AS_IS")
	tde4.setWidgetSize(_render_out_dexter_req, "res_mode", 150,20)
	tde4.setWidgetValue(_render_out_dexter_req,"res_mode","1")
	tde4.addTextFieldWidget(_render_out_dexter_req,"width","","")
	tde4.setWidgetCallbackFunction(_render_out_dexter_req,"width","_renderOutCallback")
	tde4.setWidgetSensitiveFlag(_render_out_dexter_req,"width",0)
	tde4.setWidgetOffsets(_render_out_dexter_req,"width",10,0,0,0)
	tde4.setWidgetLinks(_render_out_dexter_req,"width","res_mode","", "res_mode", "")
	tde4.setWidgetAttachModes(_render_out_dexter_req,"width","ATTACH_WIDGET","ATTACH_NONE","ATTACH_OPPOSITE_WIDGET","ATTACH_NONE")
	tde4.setWidgetSize(_render_out_dexter_req, "width", 80,20)
	tde4.addTextFieldWidget(_render_out_dexter_req,"height","X","")
	tde4.setWidgetSensitiveFlag(_render_out_dexter_req,"height",0)
	tde4.setWidgetOffsets(_render_out_dexter_req,"height",25,0,0,0)
	tde4.setWidgetLinks(_render_out_dexter_req,"height","width","", "width", "")
	tde4.setWidgetAttachModes(_render_out_dexter_req,"height","ATTACH_WIDGET","ATTACH_NONE","ATTACH_OPPOSITE_WIDGET","ATTACH_NONE")
	tde4.setWidgetSize(_render_out_dexter_req, "height", 80,20)
	tde4.addToggleWidget(_render_out_dexter_req,"overscan","Overscan (x1.08)",1)
	tde4.setWidgetCallbackFunction(_render_out_dexter_req,"overscan","_renderOutCallback")
	tde4.addOptionMenuWidget(_render_out_dexter_req,"format_mode","Image Format","Jpeg","Tiff","PNG")
	tde4.setWidgetAttachModes(_render_out_dexter_req,"format_mode","ATTACH_AS_IS","ATTACH_NONE","ATTACH_AS_IS","ATTACH_AS_IS")
	tde4.setWidgetSize(_render_out_dexter_req, "format_mode", 150,20)
	tde4.addTextFieldWidget(_render_out_dexter_req,"frame_range","Frame Range","")
		
	w0		= tde4.getCameraImageWidth(c)
	h0		= tde4.getCameraImageHeight(c)
	mode		= tde4.getWidgetValue(_render_out_dexter_req,"res_mode")
	overscan	= tde4.getWidgetValue(_render_out_dexter_req,"overscan")
	
	if mode==1:
		w	= w0
		h	= h0
	if mode==2:
		w	= w0/2
		h	= h0/2
	if mode==3:
		w	= w0/4
		h	= h0/4
	if overscan==1:
		w	= int(float(w)*1.08)
		h	= int(float(h)*1.08)
		if w%2==1:
				w += 1
		if h%2==1:
				h += 1
	tde4.setWidgetValue(_render_out_dexter_req,"width",str(w))
	tde4.setWidgetValue(_render_out_dexter_req,"height",str(h))

	n	= tde4.getCameraNoFrames(c)
	offset	= tde4.getCameraFrameOffset(c)
	pbr	= tde4.getCameraPlaybackRange(c)
	tde4.setWidgetValue(_render_out_dexter_req,"frame_range","%d %d"%(offset+pbr[0]-1,offset+pbr[1]-1))
	
	ret	= tde4.postCustomRequester(_render_out_dexter_req, "Save out Rendered Frames...", 800, 0, "Render...","Cancel")
	if ret==1:
		path		= tde4.getWidgetValue(_render_out_dexter_req,"file")
		DD_common.make_dir(os.path.split(path)[0])
		w0		= tde4.getCameraImageWidth(c)
		h0		= tde4.getCameraImageHeight(c)
		mode		= tde4.getWidgetValue(_render_out_dexter_req,"res_mode")
		overscan	= tde4.getWidgetValue(_render_out_dexter_req,"overscan")
		render_w		= int(tde4.getWidgetValue(_render_out_dexter_req,"width"))
		render_h		= int(tde4.getWidgetValue(_render_out_dexter_req,"height"))
		
		pg		= tde4.getCurrentPGroup()
		pl		= tde4.getPointList(pg,1)

		if mode==1: scale = 1.0
		if mode==2: scale = 0.5
		if mode==3: scale = 0.25
		clear	= tde4.getWidgetValue(_render_out_dexter_req,"delete_cache")
		r	= (tde4.getWidgetValue(_render_out_dexter_req,"frame_range")).split()
		mode	= tde4.getWidgetValue(_render_out_dexter_req,"format_mode")
		if mode==1: format = "IMAGE_JPEG"
		if mode==2: format = "IMAGE_TIFF"
		if mode==3: format = "IMAGE_PNG"
		
		if clear==1: tde4.clearRenderCache()
		
		tde4.postProgressRequesterAndContinue("Save out Rendered Frames...", "...", int(r[1])-int(r[0])+1, "Cancel")
		i	= 0
		for f in range(int(r[0]),int(r[1])+1):
			i		= i+1
			cont		= tde4.updateProgressRequester(i,"Saving Frame %d..."%f)
			if not cont: break
			filename	= prepareImagePath(path,f)
			x = int(float(w0)*1.4*scale/2)
			y = int(float(h0)*1.4*scale/2)
			x2 = int(float(render_w)/2)
			y2 = int(float(render_h)/2)

			if overscan==1:
				ok	= tde4.saveRenderCacheFrame(c,f-offset+1,filename,format,scale,overscan,0,x-x2,y-y2,render_w,render_h)
			else:
				ok	= tde4.saveRenderCacheFrame(c,f-offset+1,filename,format,scale,overscan,0)
			if not ok: print "Error saving file: \"%s\"."%filename
		
		tde4.unpostProgressRequester()
else:
	tde4.postQuestionRequester("Save out Rendered Frames...","There is no current Camera.","Ok")
