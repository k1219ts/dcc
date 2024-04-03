#
#
# 3DE4.script.name:	Photoscan...
#
# 3DE4.script.version:	v1.0
#
# 3DE4.script.gui:	Main Window::3DE4::Export Project
#
# 3DE4.script.comment:	Creates a py script file that contains all project data, which can be imported into Photoscan to build Point Cloud.
#
# Giovanni Di Grezia 2016
# http://www.xgiovio.com
#
# Some functions are from 3de export scripts


from vl_sdv import *

def convertToAngles(r3d):
	rot	= rot3d(mat3d(r3d)).angles(VL_APPLY_ZXY)
	rx	= (rot[0]*180.0)/3.141592654
	ry	= (rot[1]*180.0)/3.141592654
	rz	= (rot[2]*180.0)/3.141592654
	return(rx,ry,rz)

#maybe this function is not correct
def rot_matrix_for_photoscan (r3d):
	rot3d_old = rot3d(mat3d(r3d)).angles(VL_APPLY_ZXY)
	rot3d_new = rot3d(rot3d_old[1],rot3d_old[0],rot3d_old[2],VL_APPLY_ZXY)
	return rot3d_new.mat()



def prepareImagePath(path,startframe):
	path	= path.replace("\\","/")
	i	= 0
	n	= 0
	i0	= -1
	while(i<len(path)):
		if path[i]=='#': n += 1
		if n==1: i0 = i
		i	+= 1
	if i0!=-1:
		fstring		= "%%s%%0%dd%%s"%(n)
		path2		= fstring%(path[0:i0],startframe,path[i0+n:len(path)])
		path		= path2
	return path
	
def callback (req,label,event):
	if label == "match_align":
		if tde4.getWidgetValue(req,"match_align") == 1:
			tde4.setWidgetSensitiveFlag(req,"build_dense_cloud",1)
			tde4.setWidgetSensitiveFlag(req,"match_align_keypoint_limit",1)
			tde4.setWidgetSensitiveFlag(req,"match_align_tiepoint_limit",1)
		else:
			tde4.setWidgetSensitiveFlag(req,"build_dense_cloud",0)
			tde4.setWidgetSensitiveFlag(req,"match_align_keypoint_limit",0)
			tde4.setWidgetSensitiveFlag(req,"match_align_tiepoint_limit",0)

#
# search for camera point group...

campg	= None
pgl	= tde4.getPGroupList()
for pg in pgl:
	if tde4.getPGroupType(pg)=="CAMERA": campg = pg
if campg==None:
	tde4.postQuestionRequester("Export Photoscan..","Error, there is no camera point group.","Ok")
else :
	cams_undistorted_path = {}
	req	= tde4.createCustomRequester()
	tde4.addFileWidget(req,"file_browser","Exportfile...","*.py")
	tde4.addTextFieldWidget(req,"step","Frames Step","10")
	tde4.addSeparatorWidget(req,"separator")
	cl	= tde4.getCameraList()
	for cam in cl:
		tde4.addTextFieldWidget(req,cam,tde4.getCameraName(cam) + " undistorted",tde4.getCameraPath(cam))
	tde4.addSeparatorWidget(req,"separator2")
	tde4.addTextFieldWidget(req,"accuracy_position","Position Accuracy in Units","0.1")
	tde4.addTextFieldWidget(req,"accuracy_rotation","Rotation Accuracy in Degrees","2.0")	
	tde4.addSeparatorWidget(req,"separator3")
	tde4.addToggleWidget(req,"match_align","Match Photos and Align",1)
	tde4.addTextFieldWidget(req,"match_align_keypoint_limit","KeyPoint Limit","40000")
	tde4.addTextFieldWidget(req,"match_align_tiepoint_limit","TiePoint Limit","4000")
	tde4.addSeparatorWidget(req,"separator4")
	tde4.addToggleWidget(req,"build_dense_cloud","Build Dense Cloud",1)
	tde4.setWidgetCallbackFunction(req,"match_align","callback")
	ret	= tde4.postCustomRequester(req,"Export Photoscan (py-Script)...",800,0,"Ok","Cancel")
	if ret==1:
		path	= tde4.getWidgetValue(req,"file_browser")
		image_stepping	= int(tde4.getWidgetValue(req,"step"))
		for cam in cl:
			cams_undistorted_path[cam] = tde4.getWidgetValue(req,cam)
		if path!=None:
			if not path.endswith('.py'): path = path+'.py' 
			f	= open(path,"w")
			if not f.closed:

				f.write("import PhotoScan\n")
				f.write("doc = PhotoScan.app.document\n")
				f.write("chunk = PhotoScan.app.document.addChunk()\n")

				# write cameras...
				
				cl	= tde4.getCameraList()
				for cam in cl:
					camType		= tde4.getCameraType(cam)
					noframes	= tde4.getCameraNoFrames(cam)
					lens		= tde4.getCameraLens(cam)
					image_width = tde4.getCameraImageWidth(cam)
					image_height = tde4.getCameraImageHeight(cam)
					offset_timeline	= tde4.getCameraFrameOffset(cam)
					source_start = tde4.getCameraSequenceAttr(cam)[0]

					if lens!=None:
						fback_w_mm	= tde4.getLensFBackWidth(lens) * 10 #mm
						fback_h_mm	= tde4.getLensFBackHeight(lens) * 10 #mm
								
						frame = offset_timeline
						while frame<=noframes + offset_timeline - 1:
							if (frame - offset_timeline) % image_stepping == 0 :
								path	= cams_undistorted_path[cam]
								path	= prepareImagePath(path,source_start + frame - offset_timeline)
								f.write("chunk.addPhotos([\"" + path +"\"])\n")
								focal_mm= tde4.getCameraFocalLength(cam,frame) * 10 #mm

								f.write("for cam in chunk.cameras:\n")
								f.write("\tif (cam.photo.path.split(\"/\")[-1]) == " +  "\"" +  path.split("/")[-1] + "\":\n")
								f.write("\t\tcamera = cam\n")
								f.write("\t\tbreak\n")
								f.write("camera.sensor = chunk.addSensor()\n" )
								f.write("camera.sensor.type = camera.sensor.Type.Frame\n")
								f.write("camera.sensor.fixed = True\n")
								f.write("camera.sensor.width = " + str(image_width) + "\n")
								f.write("camera.sensor.height = " +  str(image_height) + "\n")
								f.write("calibration = PhotoScan.Calibration()\n" )
								f.write("calibration.f = ( " + str(focal_mm) + " / " + str(fback_w_mm) + ") * " +  str(image_width) + "\n")
								f.write("camera.sensor.user_calib = calibration\n")


								position_3vector = tde4.getPGroupPosition3D(campg,cam,frame)
								f.write("camera.reference.location = PhotoScan.Vector ( (" + str(position_3vector[0]) + "," + str(position_3vector[1]) + "," + str(position_3vector[2]) + "))\n" )

								rotation_3matrix = tde4.getPGroupRotation3D(campg,cam,frame)
								rotation_angles = convertToAngles (rotation_3matrix)
								#in photoscan yaw is 3de rotation y - pitch is 3de rotation x - roll is 3de rotation z
								f.write("camera.reference.rotation = PhotoScan.Vector ( (" + str(rotation_angles[1]) + "," + str(rotation_angles[0]) + "," + str(rotation_angles[2]) + "))\n" )


								#### 3de rotation matrix is different in axis from photoscan. If you want to set camera.transform = PhotoScan.Matrix you should create a new rotation matrix
								'''
								position_4matrix = mat4d( vec4d( 1.0,0,0,position_3vector[0]), vec4d( 0,1.0,0,position_3vector[1]) ,vec4d( 0,0,1.0,position_3vector[2]),vec4d( 0,0,0,1.0))
								rotation_3matrix = rot_matrix_for_photoscan (rotation_3matrix)
								rotation_4matrix = mat4d( vec4d( rotation_3matrix[0][0],rotation_3matrix[0][1],rotation_3matrix[0][2],0),vec4d( rotation_3matrix[1][0],rotation_3matrix[1][1],rotation_3matrix[1][2],0),vec4d( rotation_3matrix[2][0],rotation_3matrix[2][1],rotation_3matrix[2][2],0),vec4d( 0,0,0,1))
								transform_matrix = position_4matrix * rotation_4matrix
								f.write("camera.transform = PhotoScan.Matrix ([[" + str(transform_matrix[0][0]) + "," + str(transform_matrix[0][1]) + "," + str(transform_matrix[0][2]) + "," + str(transform_matrix[0][3]) +"], [" + str(transform_matrix[1][0]) + "," + str(transform_matrix[1][1]) + "," + str(transform_matrix[1][2]) + "," + str(transform_matrix[1][3]) + "], [" + str(transform_matrix[2][0]) + "," + str(transform_matrix[2][1]) + "," + str(transform_matrix[2][2]) + "," + str(transform_matrix[2][3]) + "], [" + str(transform_matrix[3][0]) + "," + str(transform_matrix[3][1]) + "," + str(transform_matrix[3][2]) + "," + str(transform_matrix[3][3]) + "]])\n" )
								'''
							frame += 1

				#### 3de markers not exported. Maybe coulb useful in the future if api allow to set the 2d tracking position on the frames using markers
				'''
				l	= tde4.getPointList(campg)
				for p in l:
					if tde4.isPointCalculated3D(campg,p):
						name = tde4.getPointName(campg,p)
						p3d	= tde4.getPointCalcPosition3D(campg,p)
						f.write("marker = chunk.addMarker()\n")
						f.write("marker.reference.location = PhotoScan.Vector ( (" + str(p3d[0]) + "," + str(p3d[1]) + "," + str(p3d[2]) + "))\n" )
						f.write("marker.label = \"" + name + "\"\n")
				'''

				f.write("chunk.camera_location_accuracy = PhotoScan.Vector( (" + tde4.getWidgetValue(req,"accuracy_position") +"," + tde4.getWidgetValue(req,"accuracy_position") + "," + tde4.getWidgetValue(req,"accuracy_position") +") )\n")
				f.write("chunk.camera_rotation_accuracy = PhotoScan.Vector( (" + tde4.getWidgetValue(req,"accuracy_rotation") +"," + tde4.getWidgetValue(req,"accuracy_rotation") + "," + tde4.getWidgetValue(req,"accuracy_rotation") +") )\n")
				
				if tde4.getWidgetValue(req,"match_align") == 1:
					f.write("chunk.matchPhotos(accuracy=PhotoScan.Accuracy.HighestAccuracy, preselection=PhotoScan.Preselection.NoPreselection, filter_mask=False, keypoint_limit= " + tde4.getWidgetValue(req,"match_align_keypoint_limit") + ", tiepoint_limit=" + tde4.getWidgetValue(req,"match_align_tiepoint_limit") + " )\n")
					f.write("chunk.alignCameras()\n")
					if tde4.getWidgetValue(req,"build_dense_cloud") == 1:
						f.write("chunk.buildDenseCloud(quality=PhotoScan.Quality.UltraQuality, filter=PhotoScan.FilterMode.NoFiltering ,keep_depth=False, reuse_depth=False)\n")
						
				f.close()
				tde4.postQuestionRequester("Export Photoscan...","Project successfully exported.","Ok")
			else:
				tde4.postQuestionRequester("Export Photoscan...","Error, couldn't open file.","Ok")		
