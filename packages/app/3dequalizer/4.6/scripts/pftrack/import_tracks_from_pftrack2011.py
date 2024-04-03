##################################################
# Description: Import 2D Tracks from PFTrack 2012.
# Modified by Daehwan Jang (daehwanj@gmail.com)
# Last updated: 2013.03.22
##################################################

# 3DE4.script.name:	Import 2D Tracks from PFTrack 2012...
#
# 3DE4.script.version:	v1.2
#
# 3DE4.script.gui:	Main Window::Dexter::Import Data
#
# 3DE4.script.comment:	Imports 2D tracking curves from an Ascii file, which is exported from PFTrack 2012.
#
#

# change log
# v1.2: Add offset frame

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
	tde4.addTextFieldWidget(req, "frame_offset", "Frame Offset", "0")

	ret	= tde4.postCustomRequester(req,"Import 2D Tracks from PFTrack 2012...",500,0,"Ok","Cancel")
	if ret==1:
		create_new = tde4.getWidgetValue(req,"mode_menu")
		path	= tde4.getWidgetValue(req,"file_browser")
		frame_offset = int( tde4.getWidgetValue(req,"frame_offset") )
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
					if create_new==1 or p==None: p = tde4.createPoint(pg)
					tde4.setPointName(pg,p,name)
					tde4.setPointColor2D(pg,p,0)

					string = f.readline()	# pass "clipNumber"
					l	= []
					for j in range(frames): l.append([-1.0,-1.0])
					string	= f.readline()
					n0	= int(string)
					for j in range(n0):
						string	= f.readline()
						line	= string.split()
						l[int(line[0])+frame_offset] = [float(line[1])/width,float(line[2])/height]
					tde4.setPointPosition2DBlock(pg,p,c,1,l)
					string = f.readline()

				f.close()
			else:
				tde4.postQuestionRequester("Import 2D Tracks from PFTrack 2012...","Error, couldn't open file.","Ok")
			
			# end main block...
			#
else:
	tde4.postQuestionRequester("Import 2D Tracks from PFTrack 2012...","There is no current Point Group or Camera.","Ok")
