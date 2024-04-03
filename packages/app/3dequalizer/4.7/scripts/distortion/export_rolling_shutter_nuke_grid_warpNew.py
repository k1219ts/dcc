#
#
# 3DE4.script.name:	Export Nuke Grid Warp Rolling Shutter New...
#
# 3DE4.script.version:	v1.4
#
# 3DE4.script.gui:	Main Window::3DE4::File::Export
#
# 3DE4.script.comment:	This script exports a Nuke Grid Warp node for rough rolling shutter compensation.
# 3DE4.script.comment:	In case of rolling shutter artifacts created due to camera rotations, Z-Depth has no influence.
# 3DE4.script.comment:	If rolling shuttfer artifacts are created due to camera translations, Z-Depth defines main distance
# 3DE4.script.comment:	between camera and scene.
#
#

#
# import sdv's python vector lib...

import sys
from vl_sdv import *

def lin_int_grid_warp(t, v1,v2):
	return (1.0-t)*v1 + t*v2
def apply_rs_correction_grid_warp(dt,q_minus,q_center,q_plus):
	a = q_center
	b = (q_plus - q_minus) / 2.0
	c = -q_center + (q_plus + q_minus) / 2.0
	return a + dt * b + dt * dt * c

def removeRollingShutter_grid_warp(campg, fbw, fbh, lcox, lcoy, xa, xb, ya, yb, rsvalue,  v0, distance, frame,cam, use_distortion): # &v)
	p2d = [0,0]
	p2dcm = vec2d(0,0)
	p = vec2d(0,0)
	focal = 0.0
	r3d = mat3d;
	d = vec3d(0,0,0)
	p3d = vec3d(0,0,0)
	p3d0 = vec3d(0,0,0)
	p3d1 = vec3d(0,0,0)
	p3d2 = vec3d(0,0,0)
	n = tde4.getCameraNoFrames(cam)

	if (n==1):
		 return v0

#	// frame-1...
	if (frame>1):
		focal = tde4.getCameraFocalLength(cam,frame-1)
		r3d= mat3d(tde4.getPGroupRotation3D(campg,cam,frame-1))
		p3d = vec3d(tde4.getPGroupPosition3D(campg,cam,frame-1))

		p2d[0]		= (v0[0]-xa)/(xb-xa)
		p2d[1]		= (v0[1]-ya)/(yb-ya)
		#if (use_distortion == 1):
		p2d = tde4.removeDistortion2D(cam,frame,p2d)

		p2dcm		= vec2d((p2d[0]-0.5)*fbw-lcox,(p2d[1]-0.5)*fbh-lcoy)
		d		= r3d*vec3d(p2dcm[0],p2dcm[1],-focal).unit()
		p3d0		= d*distance+p3d

#	// frame+1...
	if (frame<n):
		focal = tde4.getCameraFocalLength(cam,frame+1)
		r3d= mat3d(tde4.getPGroupRotation3D(campg,cam,frame+1))
		p3d = vec3d(tde4.getPGroupPosition3D(campg,cam,frame+1))

		p2d[0]		= (v0[0]-xa)/(xb-xa)
		p2d[1]		= (v0[1]-ya)/(yb-ya)
		#if (use_distortion == 1):
		p2d = tde4.removeDistortion2D(cam,frame,p2d)

		p2dcm		= vec2d((p2d[0]-0.5)*fbw-lcox,(p2d[1]-0.5)*fbh-lcoy);
		d		= r3d*vec3d(p2dcm[0],p2dcm[1],-focal).unit();
		p3d2		= d*distance+p3d;

#	// frame...
	focal = tde4.getCameraFocalLength(cam,frame)
	r3d= mat3d(tde4.getPGroupRotation3D(campg,cam,frame))
	p3d = vec3d(tde4.getPGroupPosition3D(campg,cam,frame))

	p2d[0]		= (v0[0]-xa)/(xb-xa)
	p2d[1]		= (v0[1]-ya)/(yb-ya)
	#if (use_distortion == 1):
	p2d = tde4.removeDistortion2D(cam,frame,p2d)

	p2dcm		= vec2d((p2d[0]-0.5)*fbw-lcox,(p2d[1]-0.5)*fbh-lcoy)
	d		= r3d*vec3d(p2dcm[0],p2dcm[1],-focal).unit();
	p3d1		= d*distance+p3d;

#	// schlau splinen...
	if (frame==1) :
		p3d0 = p3d1+(p3d1-p3d2)
	if (frame==n):
		 p3d2 = p3d1+(p3d1-p3d0)
	t		= rsvalue*(1.0-v0[1])
	p3d1		= apply_rs_correction_grid_warp(-t,p3d0,p3d1,p3d2)
#	// backprojection...
	d		= r3d.trans()*(p3d1-p3d)
	p2d[0]		= (d[0]*focal/(-d[2]*fbw))+(lcox/fbw)+0.5
	p2d[1]		= (d[1]*focal/(-d[2]*fbh))+(lcoy/fbh)+0.5
	p = tde4.applyDistortion2D(cam,frame,p2d)
	p		= vec2d((p[0]*(xb-xa))+xa,(p[1]*(yb-ya))+ya);
	v		= (v0+(v0-p)).list()#			// (1.0-t)*v0+t*p;
#	print(p3d0,p3d1,p3d2)
	return v




def _export_nuke_grid_warp_rs_ld_grid_warp(path, dst_width, dst_height, grid_res_x, grid_res_y,startframe,dist, use_distortion):
	cam = tde4.getCurrentCamera()
	pg = tde4.getCurrentPGroup()
	f = open(path,"w")

	width = tde4.getCameraImageWidth(cam)
	height= tde4.getCameraImageHeight(cam)

	lens = tde4.getCameraLens(cam)
	pixel_aspect = tde4.getLensPixelAspect(lens)
	[xa,xb,ya,yb] = tde4.getCameraFOV(cam)
	dst_x_offset = xa*dst_width
	dst_y_offset = ya*dst_height

	fb_width = tde4.getLensFBackWidth(lens)
	fb_height = tde4.getLensFBackHeight(lens)
	lco_x = tde4.getLensLensCenterX(lens);
	lco_y = tde4.getLensLensCenterY(lens);
	rsvalue = tde4.getCameraRollingShutterTimeShift(cam)*tde4.getCameraFPS(cam)

	#calc tile size (image border is 2 px for clamping)
	tiles_w = (width+5)/float(grid_res_x)
	tiles_h = (height+5)/float(grid_res_y)
	frames = tde4.getCameraNoFrames(cam)

	grid = []
	tde4.postProgressRequesterAndContinue("Export Nukev7 GridWarp...", "...", (grid_res_y+1)*(grid_res_x+1) + frames, "Cancel")
	pf = 0
	#create 2D array with entrys [posx,posy,undist x, undist y] all in unitspace
	for frame in range(1, frames+1):
		fr = []
		for y in range(0,grid_res_y+1):
			row = []
			for x in range(0,grid_res_x+1):
				#remind image border resize for correct clamping
				delta = 1e-4
				p1 = [(x*tiles_w + 0.5 - 3.0)/width,(y*tiles_h + 0.5 - 3.0)/height]
				#print("Frame: " , frame)
				p2 = removeRollingShutter_grid_warp(pg, fb_width, fb_height, lco_x, lco_y,xa ,xb ,ya ,yb , rsvalue,  vec2d(p1), dist, frame,cam,use_distortion)
				pxplus = removeRollingShutter_grid_warp(pg, fb_width, fb_height, lco_x, lco_y,xa ,xb ,ya ,yb , rsvalue,  vec2d(p1[0]+delta,p1[1]), dist, frame,cam,use_distortion)
				pxminus = removeRollingShutter_grid_warp(pg, fb_width, fb_height, lco_x, lco_y,xa ,xb ,ya ,yb , rsvalue,  vec2d(p1[0]-delta,p1[1]), dist, frame,cam,use_distortion)
				pyplus = removeRollingShutter_grid_warp(pg, fb_width, fb_height, lco_x, lco_y,xa ,xb ,ya ,yb , rsvalue,  vec2d(p1[0],p1[1]+delta), dist, frame,cam,use_distortion)
				pyminus = removeRollingShutter_grid_warp(pg, fb_width, fb_height, lco_x, lco_y,xa ,xb ,ya ,yb , rsvalue,  vec2d(p1[0],p1[1]-delta), dist, frame,cam,use_distortion)

				delta2 = 2.0*delta
				px = [(pxplus[0]-pxminus[0])/delta2,(pxplus[1]-pxminus[1])/delta2]
				py = [(pyplus[0]-pyminus[0])/delta2,(pyplus[1]-pyminus[1])/delta2]
				jmat = mat2d(px[0],px[1],py[0],py[1]).trans()

				if(use_distortion):
					jmat = mat2d(tde4.getDistortion2DJacobianMatrix(cam,frame,[p2[0],p2[1]]))*jmat
					p2 = tde4.removeDistortion2D(cam,frame,p2)
				row.append([p1[0],p1[1],p2[0],p2[1],jmat])
			fr.append(row)
		pf = pf + 1
		cont		= tde4.updateProgressRequester(pf,"Prepare Nukev7 GridWarp...")
		if not cont: break
		grid.append(fr)

	f.write('set cut_paste_input [stack 0]\n')
	f.write('version 7.0 v5\n')
	f.write('push $cut_paste_input\n')


	f.write('GridWarp3 {\n')
	f.write(' toolbar_visibility_src false\n')

	t_width = 1.0 / float(grid_res_x) / 3.0
	t_height = 1.0 / float(grid_res_y) / 3.0

	f.write('  source_grid_col  {\n')
	f.write('    1 %i %i 4 1 0\n'%(grid_res_x+1,grid_res_y+1))
	f.write('    {default }\n')
	f.write('    {\n')
	for y in range(0,grid_res_y+1):
		for x in range(0,grid_res_x+1):
			#for frame in range(0, frames):
			r = [t_width,0]
			l = [-t_width,0.0]
			o = [0.0,t_height]
			u=[0.0,-t_height]

			f.write('      { {2 %.15f %.15f} { {2 %.15f %.15f}  {2 %.15f %.15f}  {2 %.15f %.15f}  {2 %.15f %.15f} }  }\n'%(grid[0][y][x][0]*dst_width*(xb-xa) + dst_x_offset,grid[0][y][x][1]*dst_height*(yb-ya) + dst_y_offset,o[0]*dst_width*(xb-xa),o[1]*dst_height*(yb-ya),u[0]*dst_width*(xb-xa),u[1]*dst_height*(yb-ya),r[0]*dst_width*(xb-xa),r[1]*dst_height*(yb-ya),l[0]*dst_width*(xb-xa),l[1]*dst_height*(yb-ya)))
	f.write('    }\n')
	f.write('  }\n')


	f.write('  destination_grid_col  {\n')
	f.write('    1 %i %i 4 1 0\n'%(grid_res_x+1,grid_res_y+1))
	f.write('    {default }\n')
	f.write('    {\n')
	for y in range(0,grid_res_y+1):
		for x in range(0,grid_res_x+1):

			pointx = "{curve L "
			pointy = "{curve L "
			ta_lx = "{curve L "
			ta_ly = "{curve L "
			ta_rx = "{curve L "
			ta_ry = "{curve L "
			ta_ox = "{curve L "
			ta_oy = "{curve L "
			ta_ux = "{curve L "
			ta_uy = "{curve L "


			for frame in range(0, frames):
				r = [t_width,0]
				l = [-t_width,0.0]
				o = [0.0,t_height]
				u=[0.0,-t_height]

	 			jmat = mat2d(grid[frame][y][x][4])

				o = jmat*vec2d(o)
				u = jmat*vec2d(u)
				r = jmat*vec2d(r)
				l = jmat*vec2d(l)

				pointx = "%s x%i %.15f"%(pointx, frame  + startframe, grid[frame][y][x][2]*dst_width*(xb-xa)  + dst_x_offset)
				pointy = "%s x%i %.15f"%(pointy, frame  + startframe, grid[frame][y][x][3]*dst_height*(yb-ya) + dst_y_offset)
				ta_lx = "%s x%i %.15f"%(ta_lx, frame  + startframe, l[0]*dst_width*(xb-xa))
				ta_ly = "%s x%i %.15f"%(ta_ly, frame  + startframe, l[1]*dst_height*(yb-ya))
				ta_rx = "%s x%i %.15f"%(ta_rx, frame  + startframe, r[0]*dst_width*(xb-xa))
				ta_ry = "%s x%i %.15f"%(ta_ry, frame  + startframe, r[1]*dst_height*(yb-ya))
				ta_ox = "%s x%i %.15f"%(ta_ox, frame  + startframe, o[0]*dst_width*(xb-xa))
				ta_oy = "%s x%i %.15f"%(ta_oy, frame  + startframe, o[1]*dst_height*(yb-ya))
				ta_ux = "%s x%i %.15f"%(ta_ux, frame  + startframe, u[0]*dst_width*(xb-xa))
				ta_uy = "%s x%i %.15f"%(ta_uy, frame  + startframe, u[1]*dst_height*(yb-ya))

			pointx = "%s }"%(pointx)
			pointy = "%s }"%(pointy)
			ta_lx = "%s }"%(ta_lx)
			ta_ly = "%s }"%(ta_ly)
			ta_rx = "%s }"%(ta_rx)
			ta_ry = "%s }"%(ta_ry)
			ta_ox = "%s }"%(ta_ox)
			ta_oy = "%s }"%(ta_oy)
			ta_ux = "%s }"%(ta_ux)
			ta_uy = "%s }"%(ta_uy)

			pf = pf + 1
			cont		= tde4.updateProgressRequester(pf,"Write Nukev7 GridWarp...")
			if not cont: break

			f.write('	{{1 %s %s} { {1 %s %s} {1 %s %s} {1 %s %s} {1 %s %s} }}\n'%(pointx, pointy, ta_ox, ta_oy,ta_ux, ta_uy, ta_rx, ta_ry, ta_lx,ta_ly))
 	f.write('    }\n')
	f.write('  }\n')





	f.write(' grids_manually_moved true\n')
 	f.write(' background "on black"\n')
	f.write(' destination_grid_visible false\n')
	f.write(' source_grid_transform_center {%f %f}\n'%(width*0.5, height*0.5))
	f.write(' destination_grid_transform_center {%f %f}\n'%(width*0.5, height*0.5))
	f.write(' name GridWarp3_3DE_RS\n')
	f.write('}\n')

	f.close()


#
# open requester...
def exportRollingShutterNukeGridWarpGUI():
	cam	= tde4.getCurrentCamera()
	if cam ==0:
		tde4.postQuestionRequester("Export Nuke Grid Warp Rolling Shutter...","Error, no camera is selected.","Ok")
		return

	if 1 != tde4.getCameraRollingShutterEnabledFlag(cam):
		tde4.postQuestionRequester("Export Nuke Grid Warp Rolling Shutter...","Error: Rolling Shutter is disabled for current camera. ","Ok")
		return

	pgList = tde4.getPGroupList(0);
	pg = None #tde4.getCurrentPGroup()
	for pg_id in pgList:
		pg_type = tde4.getPGroupType(pg_id)
		if pg_type == "CAMERA":
			if pg == None:
				pg = pg_id
	if pg == None:
		tde4.postQuestionRequester("Export Nuke Grid Warp Rolling Shutter...","Error: There is no camera point group in this project.","Ok")
		return
	global _rolling_shutter_export_nuke_grid_warp
	try:

		req	= _rolling_shutter_export_nuke_grid_warp
	except (ValueError,NameError,TypeError):
		req = tde4.createCustomRequester()

		req	= tde4.createCustomRequester()
		tde4.addFileWidget	(req,"file_browser","Exportfile...","*.nk")
		tde4.addTextFieldWidget	(req, "depth", "Content Distance [cm]", "100")
		tde4.addToggleWidget	(req,"use_distortion","Apply Distortion",1)
		tde4.addTextFieldWidget	(req, "startframe_field", "Startframe", "1")
		tde4.addTextFieldWidget	(req,"img_width","Resolution")
		tde4.setWidgetLinks	(req,"img_width","startframe_field","", "startframe_field", "")
		tde4.setWidgetOffsets	(req,"img_width",30,0,5,0)
		tde4.setWidgetAttachModes(req,"img_width","ATTACH_AS_IS","ATTACH_NONE","ATTACH_AS_IS","ATTACH_AS_IS")
		tde4.setWidgetSize	(req, "img_width", 80,20)
		tde4.addTextFieldWidget	(req,"img_height","X")
		tde4.setWidgetOffsets	(req,"img_height",25,0,0,0)
		tde4.setWidgetLinks	(req,"img_height","img_width","", "img_width", "")
		tde4.setWidgetAttachModes(req,"img_height","ATTACH_WIDGET","ATTACH_NONE","ATTACH_OPPOSITE_WIDGET","ATTACH_NONE")
		tde4.setWidgetSize	(req, "img_height", 80,20)

		tde4.addOptionMenuWidget(req,"grid_width","Grid Warp Resolution","7","11","13","19")
		tde4.setWidgetLinks	(req,"grid_width","img_height","", "img_height", "")
		tde4.setWidgetOffsets	(req,"grid_width",30,0,5,0)
		tde4.setWidgetAttachModes(req,"grid_width","ATTACH_AS_IS","ATTACH_NONE","ATTACH_AS_IS","ATTACH_AS_IS")
		tde4.setWidgetSize	(req, "grid_width", 80,20)
		tde4.addOptionMenuWidget(req,"grid_height","X","7","11","13","19")
		tde4.setWidgetOffsets	(req,"grid_height",25,0,0,0)
		tde4.setWidgetLinks	(req,"grid_height","grid_width","", "grid_width", "")
		tde4.setWidgetAttachModes(req,"grid_height","ATTACH_WIDGET","ATTACH_NONE","ATTACH_OPPOSITE_WIDGET","ATTACH_NONE")
		tde4.setWidgetSize	(req, "grid_height", 80,20)

		_rolling_shutter_export_nuke_grid_warp = req

	offset	= tde4.getCameraFrameOffset(cam)
	tde4.setWidgetValue(req,"img_width",str(int(tde4.getCameraImageWidth(tde4.getCurrentCamera()))))
	tde4.setWidgetValue(req,"img_height",str(int(tde4.getCameraImageHeight(tde4.getCurrentCamera()))))
	tde4.setWidgetValue(req,"startframe_field",str(offset))
	tde4.setWidgetValue(req, "grid_width", "2")
	tde4.setWidgetValue(req, "grid_height", "2")

	ret	= tde4.postCustomRequester(req,"Export Nuke Grid Warp Rolling Shutter...",600,220,"Ok","Cancel")
	if ret!=1:
		return

	dst_width = float(tde4.getWidgetValue(req,"img_width"))
	dst_height= float(tde4.getWidgetValue(req,"img_height"))
	grid_res = {1 : 6, 2 : 10, 3 : 12, 4 : 18}
	grid_res_x = tde4.getWidgetValue(req,"grid_width")
	grid_res_y = tde4.getWidgetValue(req,"grid_height")

	frame0	= float(tde4.getWidgetValue(req,"startframe_field"))

	use_distortion = 0
	if tde4.getWidgetValue(req,"use_distortion") == 1:
		use_distortion = 1

	path	= tde4.getWidgetValue(req,"file_browser")
	dist = float(tde4.getWidgetValue(req,"depth"))
	lens   = tde4.getCameraLens(cam)
	_export_nuke_grid_warp_rs_ld_grid_warp(path, dst_width, dst_height, grid_res[grid_res_x], grid_res[grid_res_y],frame0,dist, use_distortion)



	tde4.postQuestionRequester("Export Nuke Grid Warp Rolling Shutter...","Rolling Shutter Grid Warp successfully exported.","Ok")

exportRollingShutterNukeGridWarpGUI()
