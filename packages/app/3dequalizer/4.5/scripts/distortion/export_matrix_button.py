#
#
# 3DE4.script.name:	Export Matrix for Crop...
#
# 3DE4.script.version:	v1.1
#
# 3DE4.script.gui.button:	Distortion Edit Controls::Export Matix, align-bottom-right, 70, 20
#
# 3DE4.script.comment:	Writes the matrix of the currently selected camera to an ascii file.
#
#

#
# main script...

cam	= tde4.getCurrentCamera()
if cam!=None:
	path	= tde4.postFileRequester("Export Matrix...","*.txt")
	if path!=None:
		#
		# main block...
		
		if path.find(".txt",len(path)-4)==-1: path += ".txt"
		f	= open(path,"w")
		if not f.closed:
			# header...
			f.write("# 3DE4 Matrix Data v1.0...\n")
			f.write("# %s\n\n"%(tde4.get3DEVersion()))
			# resoultion...
			w		= tde4.getCameraImageWidth(cam)
			h      = tde4.getCameraImageHeight(cam)
			f.write("%d %d\n\n"%(w,h))
			# dimensions...
			dim	= tde4.getCameraMatrixDimensions(cam)
			f.write("%d %d %d %d\n"%(dim[0],dim[1],dim[2],dim[3]))
			
			for x in range(dim[0],dim[1]+1):
				for y in range(dim[2],dim[3]+1):
					p2d	= tde4.getCameraMatrixPointPos(cam,x,y)
					enabled	= tde4.getCameraMatrixPointValidFlag(cam,x,y)
					status	= tde4.getCameraMatrixPointStatus(cam,x,y)
					f.write("%d %d %.15f %.15f %d %s\n"%(x,y,p2d[0],p2d[1],enabled,status))
			
			f.close()
		else:
			tde4.postQuestionRequester("Export Matrix...","Error, couldn't open file.","Ok")
		
		# end main block...
		#
else:
	tde4.postQuestionRequester("Export Matrix...","There is no current Camera.","Ok")




