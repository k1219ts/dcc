'''
Date : 2019. 01. 10.
Name : Hyunjun Cheong
Info : Forced Manual Mode is set when performing Houdini-FX
	+ Hotkey ( AutoUpdate & Manual Mode )
'''

# import hou library
import hou

# Get Current Mode
mode = hou.updateModeSetting().name()

# Force Manual Mode
if mode != 'Manual':
    hou.setUpdateMode(hou.updateMode.Manual)

# Set AutoUpdate/Manual Hotkeys

# Added Except Part ( This hot key class doesn't not work under 16.0 )
# if hou.applicationVersionString() != '16.0.633':
if float(hou.applicationVersionString().split('.')[0] + '.' + hou.applicationVersionString().split('.')[1]) >= 16.5:
    if hasattr(hou, 'hotkeys'):
        hou.hotkeys.addAssignment("h.update_mode_always", "F6")
        hou.hotkeys.addAssignment("h.update_mode_never", "F7")

'''
When using keymap.overrides example
h.update_mode_always	"Update Mode 'Always'"	"Update mode 'Always'"	 F11
h.update_mode_never	"Update Mode 'Never'"	"Update mode 'Never'"	 F12
'''

# print '123.py'
