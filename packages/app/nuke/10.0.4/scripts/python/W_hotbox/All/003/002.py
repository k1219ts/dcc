#----------------------------------------------------------------------------------------------------------
#
# AUTOMATICALLY GENERATED FILE TO BE USED BY W_HOTBOX
#
# NAME: Copy Tile Color
#
#----------------------------------------------------------------------------------------------------------

from pymodule.Qt import QtWidgets

colorsList = []
for i in nuke.selectedNodes():

    interfaceColor = i.knob('tile_color').value()

    if interfaceColor == 0:
        interfaceColor = nuke.defaultNodeColor(i.Class())

	colorsList.append(str(interfaceColor))

nodeColorsString = ' '.join(sorted(colorsList))

QtWidgets.QApplication.clipboard().setText(nodeColorsString)
