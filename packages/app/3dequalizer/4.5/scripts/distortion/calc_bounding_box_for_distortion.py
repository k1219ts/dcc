#
#
# 3DE4.script.name:	Compute Bounding Box For Lens Distortion
#
# 3DE4.script.version:	v1.0
#
# 3DE4.script.gui:	Distortion Edit Controls::Calc
# 3DE4.script.gui:	Main Window::Python
#
# 3DE4.script.comment:	Compute a bounding box in pixel for an undistorted image.
#
#

# Internal comment, Versions
# Internal comment, v1.0 initial

import string
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

		print "----- Camera: " + name + " -----------------"
		print "lens center in pixel:"
		print elc[0] * w_px,elc[1] * h_px
		print "non-symmetrized bounding box, pixel-aligned (x,y,w,h):"
		print x_nonsymm_px,y_nonsymm_px,w_nonsymm_px,h_nonsymm_px
		print "symmetrized bounding box, pixel-aligned (x,y,w,h):"
		print x_symm_px,y_symm_px,w_symm_px,h_symm_px

# Run the script
bbdld_compute_bounding_box()


