from utils import activeCam

node = kwargs['node']

cam = activeCam.get()

if cam:
    node.parm('camera').set(cam)