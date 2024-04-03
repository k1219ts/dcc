'''

        Dexter Studios Nuke Setup

        @ author    :   daeseok.chae @ CGSupervisor in Dexter Studios
                        kwantae.kim @ CGSupervisor in Dexter Studios

        @ date      :   2020.09.03
'''

import nuke
import os
# import nukescripts
# import sys
# import subprocess
# import getpass

nuke.tprint('initialize dcc nuke')

nuke.pluginAddPath('./icons')
nuke.pluginAddPath('./python')
nuke.pluginAddPath('./gizmos')
nuke.pluginAddPath('./plugins')

nuke.knobDefault('Root.colorManagement', 'OCIO')
# nuke.knobDefault('Viewer.viewerProcess.key1', 'SHOT')
nuke.knobDefault('OCIODisplay.key1', 'Hello')
print '# !!!!! #', nuke.root().knob('colorManagement').value()
