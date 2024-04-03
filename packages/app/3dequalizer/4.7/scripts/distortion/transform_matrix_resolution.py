#
# 3DE4.script.name:	Transform Matrix Points to Current Crop Chart
#
# 3DE4.script.version:	v1.0
#
# 3DE4.script.gui:	Distortion Edit Controls::Edit
# 
# 3DE4.script.comment:	Applies crop factor to all matrix points of the current camera
#

cam	= tde4.getCurrentCamera()
if cam!=None:
	try:
		mode	= tde4.getWidgetValue(_trans_matrix_req,"ores")
	except (ValueError,NameError,TypeError):
		_trans_matrix_req = tde4.createCustomRequester()
		tde4.addTextFieldWidget(_trans_matrix_req,"ores","Org.Resoultion(px)","0.0 0.0")

	button = tde4.postCustomRequester(_trans_matrix_req,"Transform Matrix Points from Original...",500,0,"Ok","Cancel")
	if button == 1:
		# Extract transformation parameters
		string		= (tde4.getWidgetValue(_trans_matrix_req,"ores")).split()
		rx		= float(string[0])
		ry		= float(string[1])
		w		= tde4.getCameraImageWidth(cam)
		h      = tde4.getCameraImageHeight(cam)

		fx		= rx/w
		fy		= ry/h
		ox     = (w-rx)*0.5/w
		oy     = (h-ry)*0.5/h

		
		d       = tde4.getCameraMatrixDimensions(cam)
		x       = d[0]
		while x<=d[1]:
			y       = d[2]
			while y<=d[3]:
				p2d     = tde4.getCameraMatrixPointPos(cam,x,y)
				p2d[0]	= (p2d[0]*fx)+ox
				p2d[1]	= (p2d[1]*fy)+oy
				tde4.setCameraMatrixPointPos(cam,x,y,p2d)
				y       += 1
			x       += 1
else:
	tde4.postQuestionRequester("Transform Matrix Points...","There is no current Camera.","Ok")
