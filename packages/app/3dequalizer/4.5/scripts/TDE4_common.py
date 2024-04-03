import tde4
import string
import re
import sys
import math

class bbdld_bounding_box:
	def __init__(self):
		self._x_min = float("inf")
		self._x_max = float("-inf")
		self._y_min = float("inf")
		self._y_max = float("-inf")

# Extend so that it is symmetric around (cx,cy)
	def symmetrize(self,cx,cy):
		if cx - self._x_min > self._x_max - cx:
			self._x_max = 2.0 * cx - self._x_min
		else:
			self._x_min = 2.0 * cx - self._x_max

		if cy - self._y_min > self._y_max - cy:
			self._y_max = 2.0 * cy - self._y_min
		else:
			self._y_min = 2.0 * cy - self._y_max

# Extend the bounding box so that it contains (x,y)
	def extend(self,x,y):
		self._x_min = min(self._x_min,float(x))
		self._x_max = max(self._x_max,float(x))
		self._y_min = min(self._y_min,float(y))
		self._y_max = max(self._y_max,float(y))

# Symmetric extension around (cx,cy)
	def extend_symm(self,x,y,cx,cy):
		self.extend(x,y)
		self.symmetrize(cx,cy)

# Scale, multiply x and y by some positiv number
	def scale(self,sx,sy):
		self._x_min *= sx
		self._x_max *= sx
		self._y_min *= sy
		self._y_max *= sy

# Convenient for pixel coordinates (ignore float artefacs, therefore 1e-12 thingees)
	def extend_to_integer(self):
		self._x_min = math.floor(self._x_min + 1e-12)
		self._x_max = math.ceil(self._x_max - 1e-12)
		self._y_min = math.floor(self._y_min + 1e-12)
		self._y_max = math.ceil(self._y_max - 1e-12)

# Properties
	def dx(self):
		return self._x_max - self._x_min
	def dy(self):
		return self._y_max - self._y_min
	def x_min(self):
		return self._x_min
	def x_max(self):
		return self._x_max
	def y_min(self):
		return self._y_min
	def y_max(self):
		return self._y_max

	def __str__(self):
		return "[" + str(self._x_min) + "," + str(self._x_max) + "," + str(self._y_min) + "," + str(self._y_max) + "]"


def bbdld_compute_bounding_box():
# List of selected cameras
	cameras = tde4.getCameraList(True)

# We genererate a number of samples around the image in normalized coordinates.
# These samples are later unwarped, and the unwarped points
# will be used to create a bounding box. In general, it is *not* sufficient to
# undistort only the corners, because distortion might be moustache-shaped.
# This is our list of samples:
	warped = []
	for i in range(10):
		warped.append([i / 10.0,0.0])
		warped.append([(i + 1) / 10.0,1.0])
		warped.append([0.0,i / 10.0])
		warped.append([1.0,(i + 1) / 10.0])

# Run through sequence cameras
	for id_cam in cameras:
		name = tde4.getCameraName(id_cam)
# The lens of this sequence
		id_lens = tde4.getCameraLens(id_cam)
# Lens center offset as given in GUI
		lco = [tde4.getLensLensCenterX(id_lens),tde4.getLensLensCenterY(id_lens)]
# The lens center is by definition the fixed point of the distortion mapping.
		elc = [0.5 + lco[0],0.5 + lco[1]] 
# Image size
		w_px = tde4.getCameraImageWidth(id_cam)
		h_px = tde4.getCameraImageHeight(id_cam)

# The bounding boxes for non-symmetrized and symmetrized cased.
		bb_nonsymm = bbdld_bounding_box()
		bb_symm = bbdld_bounding_box()


# Run through the frames of this camera
		n_frames = tde4.getCameraNoFrames(id_cam)
		for i_frame in range(n_frames):
# 3DE4 counts from 1.
			frame = i_frame + 1
# Now we undistort all edge points for the given
# camera and frame and extend the bounding boxes.
			for p in warped:
				p_unwarped = tde4.removeDistortion2D(id_cam,frame,p)
# Accumulate bounding boxes
				bb_nonsymm.extend(p_unwarped[0],p_unwarped[1])
				bb_symm.extend_symm(p_unwarped[0],p_unwarped[1],elc[0],elc[1])

# Scale to pixel coordinates and extend to pixel-aligned values
		bb_nonsymm.scale(w_px,h_px)
		bb_nonsymm.extend_to_integer()
# Image width and height for the non-symmetrized case
		w_nonsymm_px = bb_nonsymm.dx()
		h_nonsymm_px = bb_nonsymm.dy()
# Lower left corner for the symmetrized case. This tells us
# how the undistorted image is related to the distorted image.
		x_nonsymm_px = bb_nonsymm.x_min()
		y_nonsymm_px = bb_nonsymm.y_min()

# Scale to pixel coordinates and extend to pixel-aligned values
		bb_symm.scale(w_px,h_px)
		bb_symm.extend_to_integer()
# Image width and height for the symmetrized case
		w_symm_px = bb_symm.dx()
		h_symm_px = bb_symm.dy()
# Lower left corner for the symmetrized case. This tells us
# how the undistorted image is related to the distorted image.
		x_symm_px = bb_symm.x_min()
		y_symm_px = bb_symm.y_min()

		#print "----- Camera: " + name + " -----------------"
		#print "lens center in pixel:"
		#print elc[0] * w_px,elc[1] * h_px
		#print "non-symmetrized bounding box, pixel-aligned (x,y,w,h):"
		#print x_nonsymm_px,y_nonsymm_px,w_nonsymm_px,h_nonsymm_px
		#print "symmetrized bounding box, pixel-aligned (x,y,w,h):"
		return x_symm_px,y_symm_px,w_symm_px,h_symm_px, w_px, h_px

# We translate our API model and parameter names into Nuke identifiers.
# The rules are:
# - The empty string maps to an underscore
# - When the names starts with 0-9 it gets an underscore
# - All non-alphanumeric characters are mapped to underscores, but sequences
#   of underscores shrink to a single underscore, looks better.
def nukify_name(s):
	if s == "":
		return "_"
	if s[0] in "0123456789":
		t = "_"
	else:
		t = ""
	t += string.join(re.sub("[+,:; _-]+","_",s.strip()).split())
	return t

# Nuke interprets entities like "<" and ">".
def decode_entities(s):
	return s.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")


class CancelException(Exception):	
	pass


def getLDmodelParameterList(model):
	l = []
	for p in range(tde4.getLDModelNoParameters(model)):
		l.append(tde4.getLDModelParameterName(model, p))
	return l


def valid_name(name):
	name = name.replace('\\', '/')
	name = name.replace('//', '/')
	name = name.replace(' ', '_')
	name = name.replace('\n', '')
	name = name.replace('\r', '')
	return name


def exportNukeDewarpNode(cam, offset, nuke_path):
	lens 	= tde4.getCameraLens(cam)
	model 	= tde4.getLensLDModel(lens)
	num_frames 	= tde4.getCameraNoFrames(cam)
	w_fb_cm = tde4.getLensFBackWidth(lens)
	h_fb_cm = tde4.getLensFBackHeight(lens)
	lco_x_cm = tde4.getLensLensCenterX(lens)
	lco_y_cm = tde4.getLensLensCenterY(lens)
	pxa = tde4.getLensPixelAspect(lens)
	# xa,xb,ya,yb in unit coordinates, in this order.
	fov = tde4.getCameraFOV(cam)

	print 'camera: ', tde4.getCameraName(cam)
	print 'offset:', offset
	print 'lens:', tde4.getLensName(lens)
	print 'model: ', model

	f = open(nuke_path,"w")
	try:
		f.write('# Created by 3DEqualizer4 using Export Nuke Distortion Nodes export script\n')
		f.write("LD" + nukify_name(model) + ' {\n')
		f.write(' direction undistort\n')

		# write focal length curve if dynamic
		if tde4.getCameraZoomingFlag(cam):
			print 'dynamic focal length'
			f.write(' tde4_focal_length_cm {{curve ')
			for frame in range(1,num_frames + 1):
				f.write ('x%i' % (frame+offset))
				f.write(' %.7f ' % tde4.getCameraFocalLength(cam,frame))
			f.write('}}\n')
		# write static focal length else
		else:
			print 'static focal length'
			f.write(' tde4_focal_length_cm %.7f \n' % tde4.getCameraFocalLength(cam,1))
		# write focus distance curve if dynamic
		try:
			if tde4.getCameraFocusMode(cam) == "FOCUS_DYNAMIC":
				print 'dynamic focus distance'
				f.write(' tde4_custom_focus_distance_cm {{curve ')	
				for frame in range(1,num_frames + 1):
					f.write ('x%i' % (frame+offset))
					f.write(' %.7f ' % tde4.getCameraFocus(cam,frame))
				f.write('}}\n')
		except:
			# For 3DE4 Release 1:
			pass
		# write static focus distance else
		else:
			print 'static focus distance'
			try:
				f.write(' tde4_custom_focus_distance_cm %.7f \n' % tde4.getCameraFocus(cam,1))
			except:
				# For 3DE4 Release 1:
				f.write(' tde4_custom_focus_distance_cm 100.0 \n')
		# write camera
		f.write(' tde4_filmback_width_cm %.7f \n' % w_fb_cm)
		f.write(' tde4_filmback_height_cm %.7f \n' % h_fb_cm)
		f.write(' tde4_lens_center_offset_x_cm %.7f \n' % lco_x_cm)
		f.write(' tde4_lens_center_offset_y_cm %.7f \n' % lco_y_cm)
		f.write(' tde4_pixel_aspect %.7f \n' % pxa)
		f.write(' field_of_view_xa_unit %.7f \n' % fov[0])
		f.write(' field_of_view_ya_unit %.7f \n' % fov[2])
		f.write(' field_of_view_xb_unit %.7f \n' % fov[1])
		f.write(' field_of_view_yb_unit %.7f \n' % fov[3])
		
# write distortion parameters
#
# dynamic distortion
		try:
			dyndistmode	= tde4.getLensDynamicDistortionMode(lens)
		except:
# For 3DE4 Release 1:
			if tde4.getLensDynamicDistortionFlag(lens) == 1:
				dyndistmode = "DISTORTION_DYNAMIC_FOCAL_LENGTH"
			else:
				dyndistmode = "DISTORTION_STATIC"

		old_api = True
		try:
# If this fails, the new python API for getLensLDAdjustableParameter will be used.
# There was a bug until version 1.3, which lead to old_api false, always.
			for para in (getLDmodelParameterList(model)):
				tde4.getLensLDAdjustableParameter(lens,para,1)
		except:
			old_api = False
		
		if old_api:
			if dyndistmode=="DISTORTION_DYNAMIC_FOCAL_LENGTH":
				print 'dynamic lens distortion, focal length'
# dynamic focal length (zoom)
				for para in (getLDmodelParameterList(model)):
					f.write(' ' + nukify_name(para) + ' {{curve ')	
					for frame in range(1,num_frames + 1):
						focal = tde4.getCameraFocalLength(cam,frame)
						f.write ('x%i' % (frame+offset))
						f.write(' %.7f '%tde4.getLensLDAdjustableParameter(lens, para, focal))	
					f.write('}}\n')

			if dyndistmode=="DISTORTION_DYNAMIC_FOCUS_DISTANCE":
				print 'dynamic lens distortion, focus distance'
# dynamic focus distance
				for para in (getLDmodelParameterList(model)):
					f.write(' ' + nukify_name(para) + ' {{curve ')	
					for frame in range(1,num_frames + 1):
# Older Releases do not have Focus-methods.
						try:
							focus = tde4.getCameraFocus(cam,frame)
						except:
							focus = 100.0
						f.write ('x%i' % (frame+offset))
						f.write(' %.7f '%tde4.getLensLDAdjustableParameter(lens, para, focus))	
					f.write('}}\n')

# static distortion
			if dyndistmode=="DISTORTION_STATIC":
				print 'static lens distortion'
				for para in (getLDmodelParameterList(model)):
					f.write(' ' + nukify_name(para) + ' %.7f \n'%tde4.getLensLDAdjustableParameter(lens, para, 1))
		else:
# new API
			if dyndistmode=="DISTORTION_STATIC":
				print 'static lens distortion'
				for para in (getLDmodelParameterList(model)):
					f.write(' ' + nukify_name(para) + ' %.7f \n'%tde4.getLensLDAdjustableParameter(lens, para, 1, 1))
			else:
				print 'dynamic lens distortion,'
# dynamic
				for para in (getLDmodelParameterList(model)):
					f.write(' ' + nukify_name(para) + ' {{curve ')	
					for frame in range(1,num_frames + 1):
						focal = tde4.getCameraFocalLength(cam,frame)
						focus = tde4.getCameraFocus(cam,frame)
						f.write ('x%i' % (frame+offset))
						f.write(' %.7f '%tde4.getLensLDAdjustableParameter(lens, para, focal, focus))	
					f.write('}}\n')


		
		f.write(' name LD_3DE4_' + decode_entities(tde4.getCameraName(cam)) + '\n')
		f.write('}\n')

	finally:	
		f.close()
