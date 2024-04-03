from utils import activeCam

node = kwargs['node']

cam = activeCam.get()

if cam:
    node.parm('HO_renderCamera').set(cam)
    node.parm('HO_iprCamera').set(cam)