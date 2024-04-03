#
#
# 3DE4.script.name:	Import Matrix for Crop...
#
# 3DE4.script.version:	v1.1
#
# 3DE4.script.gui.button:	Distortion Edit Controls::Import Matix, align-bottom-right, 70, 20
#
# 3DE4.script.comment:	Reads in the matrix of the currently selected camera from an ascii file.
#
#

#
# main script...

cam	= tde4.getCurrentCamera()
if cam!=None:
	path	= tde4.postFileRequester("Import Matrix...","*.txt")
	if path!=None:
		#
		# main block...
		
		if path.find(".txt",len(path)-4)==-1: path += ".txt"
		f	= open(path,"r")
		if not f.closed:
			# read header...
			string	= f.readline()
			if string=="# 3DE4 Matrix Data v1.0...\n":
				string	= f.readline()
				string	= f.readline()

				# read original resolution
				string	= f.readline()
				orgres = string.split()				
				rx = float(orgres[0])
				ry = float(orgres[1])
				string	= f.readline()

				# read dimensions
				string	= f.readline()
				dim	= string.split()
				xa	= int(dim[0])
				xb	= int(dim[1])
				ya	= int(dim[2])
				yb	= int(dim[3])
				tde4.setCameraMatrixDimensions(cam,xa,xb,ya,yb)
				
				# read points
				string	= f.readline()
				data	= string.split()
				while len(data)==6:
					x	= int(data[0])
					y	= int(data[1])
					p2d	= [float(data[2]),float(data[3])]
					enabled	= int(data[4])
					tde4.setCameraMatrixPointPos(cam,x,y,p2d)
					tde4.setCameraMatrixPointValidFlag(cam,x,y,enabled)
					tde4.setCameraMatrixPointStatus(cam,x,y,data[5])
					
					string	= f.readline()
					data	= string.split()
				
				f.close()
				
			else:
				tde4.postQuestionRequester("Import Matrix...","Error, wrong file version or format.","Ok")
				
		else:
			tde4.postQuestionRequester("Import Matrix...","Error, couldn't open file.","Ok")
						
		#covert resolution from original res. to current res. 
		w = tde4.getCameraImageWidth(cam)
		h = tde4.getCameraImageHeight(cam)
		fx = rx/w
		fy = ry/h
		ox = (w-rx)*0.5/w
		oy = (h-ry)*0.5/h

		dm = tde4.getCameraMatrixDimensions(cam)
		dmx = dm[0]
		while dmx<=dm[1]:
			dmy	= dm[2]
			while dmy<=dm[3]:
				dp2d     = tde4.getCameraMatrixPointPos(cam,dmx,dmy)
				dp2d[0]	= (dp2d[0]*fx)+ox
				dp2d[1]	= (dp2d[1]*fy)+oy
				tde4.setCameraMatrixPointPos(cam,dmx,dmy,dp2d)
				dmy       += 1
			dmx       += 1
		
		# end main block...
		#
else:
	tde4.postQuestionRequester("Import Matrix...","There is no current Camera.","Ok")




