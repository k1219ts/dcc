#
#
# 3DE4.script.name:	Export 2D Inverse Tracks...
#
# 3DE4.script.version:	v1.0
#
# 3DE4.script.gui:	Main Window::Dexter::Export Data
#
# 3DE4.script.comment:	Writes the 2D inverse tracking curves of all selected points to an Ascii file.
#
#

#
# main script...

c	= tde4.getCurrentCamera()
pg	= tde4.getCurrentPGroup()
if c!=None and pg!=None:
	n	= tde4.getCameraNoFrames(c)
	width	= tde4.getCameraImageWidth(c)
	height	= tde4.getCameraImageHeight(c)
	seqattr = tde4.getCameraSequenceAttr(c)
	
	p	= tde4.getContextMenuObject()			# check if context menu has been used, and retrieve point...
	if p!=None:
		pg	= tde4.getContextMenuParentObject()	# retrieve point's parent pgroup (not necessarily being the current one!)...
		l	= tde4.getPointList(pg,1)
	else:
		l	= tde4.getPointList(pg,1)		# otherwise use regular selection...
	if len(l)>0:
		req	= tde4.createCustomRequester()
		tde4.addFileWidget(req,"file_browser","Filename...","*.txt")	
		ret	= tde4.postCustomRequester(req,"Export 2D Tracks...",500,0,"Ok","Cancel")
		if ret==1:
			path	= tde4.getWidgetValue(req,"file_browser")
			if path!=None:
				#
				# main block...
				
				if path.find(".txt",len(path)-4)==-1: path += ".txt"
				f	= open(path,"w")
				if not f.closed:
					f.write("%d\n"%(len(l)))
					for point in l:
						endframe = seqattr[1] - seqattr[0] + 1
						name	= tde4.getPointName(pg,point)
						f.write(name); f.write("\n")
						color	= tde4.getPointColor2D(pg,point)
						f.write("%d\n"%(color))
						c2d	= tde4.getPointPosition2DBlock(pg,point,c,1,n)
						n0	= 0
						for v in c2d:
							if v[0]!=-1.0 and v[1]!=-1.0: n0 += 1
						f.write("%d\n"%(n0))
						frame	= 1
						for v in c2d:
							if v[0]!=-1.0 and v[1]!=-1.0: f.write("%d %.15f %.15f\n"%(endframe,v[0]*width,v[1]*height))
							endframe -= 1
							frame	+= 1
					f.close()
				else:
					tde4.postQuestionRequester("Export 2D Tracks...","Error, couldn't open file.","Ok")
				
				# end main block...
				#
				
	else:
		tde4.postQuestionRequester("Export 2D Tracks...","There are no selected points.","Ok")
else:
        tde4.postQuestionRequester("Export 2D Tracks...","There is no current Point Group or Camera.","Ok")




