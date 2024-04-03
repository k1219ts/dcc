import hou

def createdSetup(node):
    show = hou.expandString('$SHOW')
    if show:
        node.parm('SHOW').set(show)
    seq = hou.expandString('$SEQ')
    if seq:
        node.parm('SEQ').set(seq)
    shot = hou.expandString('$SHOT')
    if shot:
        node.parm('SHOT').set(shot)
