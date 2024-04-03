##################################################
# Description: Export Stereo 2D Tracks to PFTrack 2011.
# Modified by Daehwan Jang (daehwanj@gmail.com)
# Last updated: 2011.04.26(v1.0)
##################################################

# 3DE4.script.name:	Export Stereo 2D Tracks To PFTrack 2011...
#
# 3DE4.script.version:	v1.0
#
# 3DE4.script.gui:	Main Window::Dexter::Export Data
#
# 3DE4.script.comment:	Writes the Stereo 2D tracking data of all selected points to an Ascii file, which can be imported into PFTrack 2011.
#
#

#
# main script...

c	= tde4.getCameraList()
pg	= tde4.getCurrentPGroup()

if c!=None and pg!=None:
	s3dCamList = ['Primary', 'Secondary']
	for ac in c:
		if tde4.getCameraStereoMode(ac) == "STEREO_PRIMARY":
			s3dCamList[0] = ac
		elif tde4.getCameraStereoMode(ac) == "STEREO_SECONDARY":
			s3dCamList[1] = ac
	
	if s3dCamList[0] != 'Primary' and s3dCamList[1] != 'Secondary':
		n	= tde4.getCameraNoFrames(s3dCamList[0])
		width	= tde4.getCameraImageWidth(s3dCamList[0])
		height	= tde4.getCameraImageHeight(s3dCamList[0])
		l	= tde4.getPointList(pg,1)
		
		startFrame = tde4.getCameraSequenceAttr(s3dCamList[0])
		
		if len(l)>0:
			path	= tde4.postFileRequester("Export Stereo 2D Tracks to PFTrack 2011...","*.txt")
			if path!=None:
				#
				# main block...
			
				if path.find(".txt",len(path)-4)==-1: path += ".txt"
				f	= open(path,"w")

				f.write("# Stereo 2D feature tracks generated by 3DEquazlier V4\n") 
				f.write("# export_stereo_tracks_to_pftrack2011.py v0.1\n") 
				f.write("#\n")
				f.write("# \"Name\"\n")
				f.write("# clipNumber\n")
				f.write("# frameCount\n")
				f.write("# frame, xpos, ypos, similarity\n")
				f.write("#\n\n")

				if not f.closed:
					for point in l:
						name	= tde4.getPointName(pg,point)

						for s3dCam in s3dCamList:
							f.write("\"exported_"+ name +"\"\n")	# a name of point
							if s3dCam == s3dCamList[1]:
								f.write("2\n")		# which of camera(primary, secondary)
							else:
								f.write("1\n")		# which of camera(primary, secondary)

							c2d	= tde4.getPointPosition2DBlock(pg,point,s3dCam,1,n)
							n0	= 0
							for v in c2d:
								if v[0]!=-1.0 and v[1]!=-1.0: n0 += 1
							f.write("%d\n"%(n0))		# length of frames that be tracked

							frame	= startFrame[0]			# start frame

							for v in c2d:
								if v[0]!=-1.0 and v[1]!=-1.0:
									f.write("%d %.3f %.3f "%(frame,v[0]*width,v[1]*height))
									f.write("1.000\n")	# tracked points error. just leaves one)
								frame += 1
					
						f.write("\n")
					f.close()
				else:
					tde4.postQuestionRequester("Export 2D Tracks to PFTrack 2011...","Error, couldn't open file.","Ok")
				# end main block...
				#
			tde4.postQuestionRequester("Export 2D Tracks to PFTrack 2011...","Exporting done! If you export tracks from linux to windows, There will be newline problem. Then use unix2dos command in terminal.","Ok")
		else:
			tde4.postQuestionRequester("Export 2D Tracks to PFTrack 2011...","Select Points which you want to export first","Ok")
	else:
		tde4.postQuestionRequester("Export 2D Tracks to PFTrack 2011...","There is no the Primary or Secondary Camera. Check Stereo Mode in Attribute Editor.","Ok")
else:
	tde4.postQuestionRequester("Export 2D Tracks to PFTrack 2011...","There is no current Point Group or Camera.","Ok")
