#
# 3DE4.script.name:  Render Out Reference Frames...
#
# 3DE4.script.version:  v1.1
#
# 3DE4.script.gui:  Main Window::Dexter::Export Data
#
# 3DE4.script.comment:  Render Out Reference Frames in Overview...
#
#

cl = tde4.getCameraList()
i = 1
req = tde4.createCustomRequester()
tde4.addFileWidget(req, "file", "File", "*", "")

ret = tde4.postCustomRequester(req, "Render Out Reference Frames", 700, 0, "Ok", "Cancel" )
if ret == 1:
	fp = tde4.getWidgetValue(req, "file")

	for c in cl:
		filename = "%s.%04d.jpg"%(fp, i)
		tde4.saveRenderCacheFrame(c, 1, filename, "IMAGE_JPEG", 0.5, 0)
		i = i+1