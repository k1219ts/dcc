import nuke

def convert():
    sto = nuke.selectedNode()
    xp = sto['xpos'].value()
    yp = sto['ypos'].value()

    tw = nuke.nodes.TimeWarp()
    tw['xpos'].setValue(xp+90)
    tw['ypos'].setValue(yp-0)
    tw['lookup'].setAnimated()

    sf=nuke.frame()

    for f in range(int(nuke.root()['first_frame'].value()),int(nuke.root()['last_frame'].value()+1)) :
        nuke.frame(f)
        lu = f-int(sto['time_offset'].valueAt(f))
        tw['lookup'].setValueAt(lu,f)

