#
# 3DE4.script.name:	Extract Focal Length from OpenEXR Metadata
#
# 3DE4.script.version:	v1.3
#
# 3DE4.script.gui:	Curve Editor::Edit
#
# 3DE4.script.comment:	Extracts focal length from metadata of exr images
# 3DE4.script.comment:	into the focal length curve of the current camera.
#

# Internal comment: Versions
# Internal comment: v1.3 Python 2 and 3
# Internal comment: v1.2 more robust against non-ascii data
# Internal comment: v1.0 initial

import sys
import xml.etree.ElementTree as et

class exflmeta_script:
# Error classes
	class error(Exception):
		def __init__(self,value):
			self.value = value
		def __str__(self):
			return repr(self.value)
# Script name
	def name(self):
			return "Extract Focal Length from EXR Metadata"
# Main entry point, called from top-level.
	def main(self):
		try:
			self.build_dialog()
		except self.error as e:
# Error class defined for this script
			tde4.postQuestionRequester("Error",str(e),"Close")
			return
		except:
# Any other error class
			tde4.postQuestionRequester("Error","An error has occurred, see python console window","Close")
			raise
# Check Buttons of dislog
		if tde4.postCustomRequester(self.id_req,self.name(),0,0,"Ok","Cancel") == 1:
# Evaluate widgets of dialog here
			i_unit = tde4.getWidgetValue(self.id_req,"option_menu_widget_unit")
# millimeter, centimeter, inch
			if i_unit == 1:
				self.unit_factor = 1.0 / 10.0
			if i_unit == 2:
				self.unit_factor = 1.0
			if i_unit == 3:
				self.unit_factor = 2.54
			self.field = tde4.getWidgetValue(self.id_req,"text_field_widget_field")
# Empty field name not good.
			if self.field == None:
				raise self.error("Empty metadata field name")
# At this point, all data are gathered and prepared.
			try:
				self.run()
			except self.error as e:
				tde4.postQuestionRequester("Error",str(e),"Close")

# Method called when all required data are prepared.
	def run(self):
		id_cam = tde4.getCurrentCamera()
		num_frames = tde4.getCameraNoFrames(id_cam)
# Clear focal length curve
		id_curve = tde4.getCameraZoomCurve(id_cam)
		tde4.deleteAllCurveKeys(id_curve)
# Iterate over frames and build filename
		for i_frame in range(num_frames):
			frame = i_frame + 1
			path = tde4.getCameraFrameFilepath(id_cam,frame)
# Extract user-defined metadata field. We have already done a test, if path is an exr-file, so let's go ahead.
			xml = tde4.convertOpenEXRMetaDataToXML(path)
# Make sure the xml-string doesn't contain strange characters.
			xml_clean = "".join(c for c in xml if ord(c) < 128)

			root = et.fromstring(xml_clean)
			found = False
			for a in root.findall("attribute"):
				name = a.find("name")
				if name.text == self.field:
					fl_cm = float(a.find("value").text) * self.unit_factor
					found = True
			if not found:
				raise self.error("No field named '" + self.field + "' found in frame %d" % frame)
# We create a key for a piecewise linear curve, x-position locked.
			id_key = tde4.createCurveKey(id_curve,[frame,fl_cm])
			tde4.setCurveKeyMode(id_curve,id_key,"LINEAR")
			tde4.setCurveKeyFixedXFlag(id_curve,id_key,1)

	def build_dialog(self):
# Create dialog for this script if not already exists.
		try:
			self.id_req
		except:
# Create requester
			self.id_req = tde4.createCustomRequester()
# We cannot be sure about the name of the focal length field. Let's make some educated guess.
			path = tde4.getCameraFrameFilepath(tde4.getCurrentCamera(),1)
# Extract user-defined metadata field.
			try:
				xml = tde4.convertOpenEXRMetaDataToXML(path)
			except:
				raise self.error("File '" + path + "' doesn't seem to be an EXR file.")
# Make sure the xml-string doesn't contain strange characters.
			xml_clean = "".join(c for c in xml if ord(c) < 128)

			root = et.fromstring(xml_clean)
			found = False
			for a in root.findall("attribute"):
				if "focal" in a.find("name").text.lower():
# Smells like focal length...
					tde4.addTextFieldWidget(self.id_req,"text_field_widget_field","Metadata field to extract",a.find("name").text)
					found = True
					break
			if not found:
				tde4.addTextFieldWidget(self.id_req,"text_field_widget_field","Metadata field to extract","no idea")
# Usually, focal length values are given in mm, so lets choose mm as default.
			tde4.addOptionMenuWidget(self.id_req,"option_menu_widget_unit","Metadata field has unit","mm","cm","inch")

try:
	the_exflmeta_script
except:
	the_exflmeta_script = exflmeta_script()

# del-commands during development ensure that
# the script is re-loaded again within 3DE4
# after editing. When in use, objects are not deleted
# so that 3DE4 remembers entries in the dialog from one
# script call to the next (which is useful).
try:
	the_exflmeta_script.main()
except:
# During development:
	del the_exflmeta_script
	del exflmeta_script
#end
	raise
else:
# During development:
	del the_exflmeta_script
	del exflmeta_script
#end
	pass
