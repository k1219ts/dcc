#
#
# 3DE4.script.name:	Export 2D Tracks To Nuke...
#
# 3DE4.script.version:	v1.0
#
# 3DE4.script.gui:	Main Window::3DE4::File::Export
# 3DE4.script.gui:	Object Browser::Context Menu Point
# 3DE4.script.gui:	Object Browser::Context Menu Points
# 3DE4.script.gui:	Object Browser::Context Menu PGroup
#
# 3DE4.script.comment:	Exports all selected tracking points or all tracking points of current point group if nothing was selected.
# 3DE4.script.comment:	Creates a Nuke file that can be imported in Nuke as a Tracker4 node containing all exported points.
#
#
# Based on: export_tracks_to_Nuke
# Author: Pushkarev Aleksandr

import os

startStr = '''set cut_paste_input [stack 0]
push $cut_paste_input
Tracker4 {
tracks { { 1 31 %d }
{ { 5 1 20 enable e 1 }
{ 3 1 75 name name 1 }
{ 2 1 58 track_x track_x 1 }
{ 2 1 58 track_y track_y 1 }
{ 2 1 63 offset_x offset_x 1 }
{ 2 1 63 offset_y offset_y 1 }
{ 4 1 27 T T 1 }
{ 4 1 27 R R 1 }
{ 4 1 27 S S 1 }
{ 2 0 45 error error 1 }
{ 1 1 0 error_min error_min 1 }
{ 1 1 0 error_max error_max 1 }
{ 1 1 0 pattern_x pattern_x 1 }
{ 1 1 0 pattern_y pattern_y 1 }
{ 1 1 0 pattern_r pattern_r 1 }
{ 1 1 0 pattern_t pattern_t 1 }
{ 1 1 0 search_x search_x 1 }
{ 1 1 0 search_y search_y 1 }
{ 1 1 0 search_r search_r 1 }
{ 1 1 0 search_t search_t 1 }
{ 2 1 0 key_track key_track 1 }
{ 2 1 0 key_search_x key_search_x 1 }
{ 2 1 0 key_search_y key_search_y 1 }
{ 2 1 0 key_search_r key_search_r 1 }
{ 2 1 0 key_search_t key_search_t 1 }
{ 2 1 0 key_track_x key_track_x 1 }
{ 2 1 0 key_track_y key_track_y 1 }
{ 2 1 0 key_track_r key_track_r 1 }
{ 2 1 0 key_track_t key_track_t 1 }
{ 2 1 0 key_centre_offset_x key_centre_offset_x 1 }
{ 2 1 0 key_centre_offset_y key_centre_offset_y 1 }
}
{
'''
endStr = '''}
}
name Tracker1
}'''


c = tde4.getCurrentCamera()
pg = tde4.getCurrentPGroup()
if c!=None and pg!=None:
	n = tde4.getCameraNoFrames(c)
	width = tde4.getCameraImageWidth(c)
	height = tde4.getCameraImageHeight(c)
	offset = tde4.getCameraFrameOffset(c)-1
	pl = tde4.getPointList(pg,1)
	if len(pl)==0:
		pl = tde4.getPointList(pg)
	if len(pl)>0 and n>0:
		path = tde4.postFileRequester("Export 2D Tracks To Nuke...","*.nk")
		if path!=None:
			if not os.path.isdir(path):
				if not path.endswith('.nk'):
					path+='.nk'
				f = open(path,"w")
				if not f.closed:
					f.write(startStr%(len(pl)))
					for point in pl:
						c2d	= tde4.getPointPosition2DBlock(pg,point,c,1,n)
						xCurve = ''
						yCurve = ''
						inAnim = False
						ff = -1
						frame = 1
						count = 0
						isPrevFirstKey = False
						for v in c2d:
							if tde4.isPointPos2DValid(pg,point,c,frame):
								if ff==-1:
									ff=frame+offset
								if not inAnim:
									xCurve+=' x'+str(frame+offset)+' '+str(v[0]*width)
									yCurve+=' x'+str(frame+offset)+' '+str(v[1]*height)
									inAnim = True
									count+=1
									isPrevFirstKey = True
								else:
									if count>1 and isPrevFirstKey:
										xCurve+=' x'+str(frame+offset)+' '+str(v[0]*width)
										yCurve+=' x'+str(frame+offset)+' '+str(v[1]*height)
									else:
										xCurve+=' '+str(v[0]*width)
										yCurve+=' '+str(v[1]*height)
									isPrevFirstKey = False
							else:
								inAnim = False
								isPrevFirstKey = False
							frame+=1
						ff = str(ff)
						f.write(' { {curve K x'+ff+' 1} "track '+tde4.getPointName(pg,point)+'" {curve'+xCurve+'} {curve'+yCurve+'} {curve K x'+ff+' 0} {curve K x'+ff+' 0} 0 0 0 {curve x'+ff+' 0} 1 0 -32 -32 32 32 -22 -22 22 22 {} {}  {}  {}  {}  {}  {}  {}  {}  {}  {}   } \n')
					f.write(endStr)
					f.close()
				else:
					tde4.postQuestionRequester("Export 2D Tracks To Nuke...","Error, couldn't open file.","Ok")
			else:
				tde4.postQuestionRequester("Export 2D Tracks To Nuke...","Filename is empty.","Ok")
	else:
		tde4.postQuestionRequester("Export 2D Tracks To Nuke...","Point group or Camera is empty.","Ok")
else:
	tde4.postQuestionRequester("Export 2D Tracks To Nuke...","There is no current Point Group or Camera.","Ok")