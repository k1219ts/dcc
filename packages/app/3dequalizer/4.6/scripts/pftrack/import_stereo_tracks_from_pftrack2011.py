##################################################
# Description: Import 2D stereo Tracks from PFTrack 2011.
# Modified by Dongho Cha (izerax.ch@gmail.com)
# Last updated: 2011.12.2(v1.0)
##################################################

# 3DE4.script.name:	Import 2D stereo Tracks from PFTrack 2011...
#
# 3DE4.script.version:	v1.0
#
# 3DE4.script.gui:	Main Window::Dexter::Import Data
#
# 3DE4.script.comment:	Imports 2D stereo tracking curves from an Ascii file, which is exported from PFTrack 2011.
#
#

#
# main script...

c	= tde4.getCurrentCamera()
cl	= tde4.getCameraList()
pg	= tde4.getCurrentPGroup()
if c!=None and pg!=None:
	frames	= tde4.getCameraNoFrames(c)
	width	= tde4.getCameraImageWidth(c)
	height	= tde4.getCameraImageHeight(c)

	#
	# get Camera Sequence
	# get timeOffset
	seqInfo = tde4.getCameraSequenceAttr(c)
	timeOffset = seqInfo[1] - tde4.getCameraNoFrames(c) + 1
	
	req	= tde4.createCustomRequester()
	tde4.addFileWidget(req,"file_browser","Filename...","*.txt")
	tde4.addOptionMenuWidget(req,"mode_menu","","Always Create New Points","Replace Existing Points If Possible")
	
	ret	= tde4.postCustomRequester(req,"Import 2D stereo Tracks from PFTrack 2011...",500,120,"Ok","Cancel")
	if ret==1:
		create_new = tde4.getWidgetValue(req,"mode_menu")
		path	= tde4.getWidgetValue(req,"file_browser")
		if path!=None:
			#
			# main block...
			
			f	= open(path,"r")
			if not f.closed:
				string	= f.readline()

				#
				# pass comments in file...
				while string[0] == '#':
					string = f.readline()

				while string == "\n":
					string	= f.readline()
					if string == "":
						break
					name	= string.rstrip("\n")
					name	= name.strip('"')
					p	= tde4.findPointByName(pg,name)
					
					string = f.readline()	# "clipNumber"
					cn	= int(string)
					if (create_new==1 or p==None) and cn==1: p = tde4.createPoint(pg)
					tde4.setPointName(pg,p,name)
					tde4.setPointColor2D(pg,p,1)
					l	= []
					for j in range(frames): l.append([-1.0,-1.0])
					string	= f.readline()
					n0	= int(string)
					for j in range(n0):
						string	= f.readline()
						line	= string.split()
						l[int(line[0])-timeOffset] = [float(line[1])/width,float(line[2])/height]
				
					tde4.setPointPosition2DBlock(pg,p,cl[cn-1],1,l)
					string = f.readline()

				f.close()
			else:
				tde4.postQuestionRequester("Import 2D Tracks from PFTrack 2011...","Error, couldn't open file.","Ok")
			
			# end main block...
			#
else:
	tde4.postQuestionRequester("Import 2D Tracks from PFTrack 2011...","There is no current Point Group or Camera.","Ok")
