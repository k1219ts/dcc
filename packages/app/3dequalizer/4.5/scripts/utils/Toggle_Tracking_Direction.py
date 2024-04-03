# 3DE4.script.name:    Toggle Tracking Direction
# 3DE4.script.version: v1.0    
# 3DE4.script.gui:    Main Window::Dexter
# 3DE4.script.comment: Toggle's tracking direction

from tde4 import *

pg = getCurrentPGroup()
cam = getCurrentCamera()
nfr = getCameraNoFrames(cam)
points = getPointList(pg, 1)

if cam!=None and pg!=None:
    for point in points:
        if getPointTrackingDirection(pg, point)=='TRACKING_FW':
            setPointTrackingDirection(pg, point, 'TRACKING_BW')
        elif getPointTrackingDirection(pg, point)=='TRACKING_BW':
            setPointTrackingDirection(pg, point, 'TRACKING_FW_BW')
        elif getPointTrackingDirection(pg, point)=='TRACKING_FW_BW':
            setPointTrackingDirection(pg, point, 'TRACKING_FW')
