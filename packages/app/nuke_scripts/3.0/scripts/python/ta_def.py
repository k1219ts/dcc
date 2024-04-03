#Find_ID by SEOBC
import nuke
def findId():
    for rd in nuke.selectedNodes():

# Read File
        fe = rd['file'].value()
        xp = rd['xpos'].value()
        yp = rd['ypos'].value()
        fs = fe.split('/')
        ub = fe.split('_')
        pt = ub[-1].split('.')

# StickyNote
        sk = nuke.nodes.StickyNote()
        sk.knob('label').setValue(pt[-3])
        sk.knob('note_font_size').setValue(30)
        sk.knob('note_font_color').setValue(4294967295)
        sk.setInput(0,rd)
        sk['xpos'].setValue(xp-0)
        sk['ypos'].setValue(yp-41)
        if fs.count('pub') == 1 :
            sk['tile_color'].setValue(4194303)
        else :
            sk['tile_color'].setValue(4278190335)

