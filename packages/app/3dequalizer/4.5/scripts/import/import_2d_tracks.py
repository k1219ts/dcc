#
#
# 3DE4.script.name:	Import 2D Tracks for Dexter...
#
# 3DE4.script.version:	v1.0
#
# 3DE4.script.gui:	Main Window::Dexter::Import Data
#
# 3DE4.script.comment:	Imports 2D tracking curves from an Ascii file.
#
#

# added 'if' statement to ignore unvalid data

#
# main script...

c	= tde4.getCurrentCamera()
pg	= tde4.getCurrentPGroup()
if c!=None and pg!=None:
	frames	= tde4.getCameraNoFrames(c)
	width	= tde4.getCameraImageWidth(c)
	height	= tde4.getCameraImageHeight(c)
	
	req	= tde4.createCustomRequester()
	tde4.addFileWidget(req,"file_browser","Filename...","*.txt")
	tde4.addOptionMenuWidget(req,"mode_menu","","Always Create New Points","Replace Existing Points If Possible")
	
	ret	= tde4.postCustomRequester(req,"Import 2D Tracks...",500,120,"Ok","Cancel")
	if ret==1:
		create_new = tde4.getWidgetValue(req,"mode_menu")
		path	= tde4.getWidgetValue(req,"file_browser")
		if path!=None:
			#
			# main block...
			
			f	= open(path,"r")
			if not f.closed:
				string	= f.readline()
				n	= int(string)
				for i in range(n):
					name	= f.readline()
					name	= name.rstrip("\n")
					p	= tde4.findPointByName(pg,name)
					if create_new==1 or p==None: p = tde4.createPoint(pg)
					tde4.setPointName(pg,p,name)
					
					string	= f.readline()
					color	= int(string)
					tde4.setPointColor2D(pg,p,color)
					
					l	= []
					for j in range(frames): l.append([-1.0,-1.0])
					string	= f.readline()
					n0	= int(string)
					for j in range(n0):
						string	= f.readline()
						line	= string.split()
						if 0.0 <= float(line[1]) <= width and 0.0 <= float(line[2]) <= height:
							l[int(line[0])-1] = [float(line[1])/width,float(line[2])/height]
					tde4.setPointPosition2DBlock(pg,p,c,1,l)
				f.close()
			else:
				tde4.postQuestionRequester("Import 2D Tracks...","Error, couldn't open file.","Ok")
			
			# end main block...
			#
else:
	tde4.postQuestionRequester("Import 2D Tracks...","There is no current Point Group or Camera.","Ok")




