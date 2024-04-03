##################################################
# Description: Export Survey Textfile for PFTrack 2011.
# Modified by Dongho Cha (izerax.ch@gmail.com)
# Last updated: 2012.5.23(v1.0)
##################################################
#
# 3DE4.script.name:	Export Survey Textfile for PFTrack 2011
#
# 3DE4.script.version:	v1.0
#
# 3DE4.script.gui:	Main Window::Dexter::Export Data
#
# 3DE4.script.comment:	Writes the 3D positions of all selected points to an Ascii file for PFTrack 2011.
#
#

#
# main script...

pg	= tde4.getCurrentPGroup()
if pg!=None:
	l	= tde4.getPointList(pg,1)
	if len(l)>0:
		path	= tde4.postFileRequester("Export Survey Textfile for PFTrack 2011..","*.txt")
		if path!=None:
			#
			# main block...
			
			if path.find(".txt",len(path)-4)==-1: path += ".txt"
			f	= open(path,"w")
			if not f.closed:
				f.write("# Name\t\tSurveyX\tSurveyY\tSurveyZ\tUncertainty\n")
				for point in l:
					if tde4.isPointCalculated3D(pg,point):
						name	= tde4.getPointName(pg,point)
						p3d	= tde4.getPointCalcPosition3D(pg,point)
						f.write("\"exported_%s\"\t%.6f\t%.6f\t%.6f\t0.000000\n"%(name,p3d[0],p3d[1],p3d[2]))
				f.close()
			else:
				tde4.postQuestionRequester("Export Survey Textfile...","Error, couldn't open file.","Ok")
			
			# end main block...
			#
	else:
		tde4.postQuestionRequester("Export Survey Textfile...","Error, there are no Points selected.","Ok")
else:
	tde4.postQuestionRequester("Export Survey Textfile...","There is no current Point Group.","Ok")




