#----------------------------------------------------------------------------------------------------------
#
# AUTOMATICALLY GENERATED FILE TO BE USED BY W_HOTBOX
#
# NAME: Copy Class
#
#----------------------------------------------------------------------------------------------------------

from PySide2 import QtWidgets
import nuke

nodeClasses = ' '.join(sorted([i.Class() for i in nuke.selectedNodes()]))

QtWidgets.QApplication.clipboard().setText(nodeClasses)

	
