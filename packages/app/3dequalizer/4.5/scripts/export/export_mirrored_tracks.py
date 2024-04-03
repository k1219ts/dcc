##################################################
# Description: Writes mirrored 2D tracking curves of all selected points to an Ascii file.
# Modified by Daehwan Jang (daehwanj@gmail.com)
# Last updated: May 18, 2011
##################################################

#
#
# 3DE4.script.name:	Export Mirrored 2D Tracks...
#
# 3DE4.script.version:	v1.0
#
# 3DE4.script.gui:	Main Window::Dexter::Export Data
#
# 3DE4.script.comment:	Writes mirrored 2D tracking curves of all selected points to an Ascii file.
#
#

#
# main script...

c	= tde4.getCurrentCamera()
pg	= tde4.getCurrentPGroup()
if c!=None and pg!=None:
	try:
		req	= _export_mirrored_tracks
	except (ValueError,NameError,TypeError):
		_export_mirrored_tracks	= tde4.createCustomRequester()
		req				= _export_mirrored_tracks
		tde4.addToggleWidget(req,"MirrorWidth","Mirror Width",0)	
		tde4.addToggleWidget(req,"MirrorHeight","Mirror Height",0)	

	ret	= tde4.postCustomRequester(req,"Export Mirrored 2D Tracks...",600,0,"Ok","Cancel")
	MirrorWidthData	= tde4.getWidgetValue(req,"MirrorWidth")
	MirrorHeightData	= tde4.getWidgetValue(req,"MirrorHeight")
	n	= tde4.getCameraNoFrames(c)

	width	= tde4.getCameraImageWidth(c) * -1
	height	= tde4.getCameraImageHeight(c) * -1

	if MirrorWidthData != 0:
		width	= width * -1
	if MirrorHeightData != 0:
		height	= height * -1

	l	= tde4.getPointList(pg,1)
	if len(l)>0:
		path	= tde4.postFileRequester("Export Mirrored 2D Tracks...","*.txt")
		if path!=None:
			#
			# main block...
			
			if path.find(".txt",len(path)-4)==-1: path += ".txt"
			f	= open(path,"w")
			if not f.closed:
				f.write("%d\n"%(len(l)))
				for point in l:
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
						if v[0]!=-1.0 and v[1]!=-1.0: f.write("%d %.15f %.15f\n"%(frame,(MirrorWidthData-v[0])*width, (MirrorHeightData-v[1])*height))
						frame	+= 1
				f.close()
			else:
				tde4.postQuestionRequester("Export Mirrored 2D Tracks...","Error, couldn't open file.","Ok")
			
			# end main block...
			#
else:
	tde4.postQuestionRequester("Export Mirrored 2D Tracks...","There is no current Point Group or Camera.","Ok")
