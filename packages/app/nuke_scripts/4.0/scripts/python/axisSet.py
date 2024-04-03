import nuke, nukescripts
import os



def axisSet00():

    a = nuke.selectedNodes()
    tem = []
    for i in a:
        tem.append(i)

    setXpos = ((int(a[-1].xpos()) + int(a[0].xpos()+100))/2)

    pos = 0

    sl = []

    rl = []

    tl = []

    for node in tem:
        b = nuke.createNode('loc_sphere')
        b.setInput(0,node)
        b.setXYpos(node.xpos(),node.ypos() + 150)
        sl.append(b)

    nukescripts.clear_selection_recursive()

    sm = nuke.createNode('Scene')

    for num in range(len(sl)):

        slo = sl[num]

        sm.setXYpos(int(setXpos) , slo.ypos() + 100)

        sm.setInput(num,slo)

    nukescripts.clear_selection_recursive()

    dot2 = nuke.createNode('Dot')
    dot2.knob('tile_color').setValue(927764223)
    dot2.setXYpos(sm.xpos() - 200, sm.ypos() + 200)
    dot2.knob('label').setValue('camera')
    dot2.knob('note_font_size').setValue(100)
    nukescripts.clear_selection_recursive()
    sm['selected'].setValue(True)


    ren = nuke.nodes.ScanlineRender()

    ren.setXYpos(sm.xpos()-10 , sm.ypos() + 400)

    ren.setInput(1,sm)
    ren.setInput(2,dot2)



def axisSet01():

    a = nuke.selectedNodes()
    setXpos = (int(a[-1].xpos()) + int(a[0].xpos()+100))/2


    pos = 0

    sl = []

    rl = []

    tl = []

    for node in a:

        b = nuke.nodes.Sphere()
        b['hide_input'].setValue('True')

        b.setXYpos(node.xpos()+ 50, node.ypos() + 200)

        b['uniform_scale'].setValue(1)

        sl.append(b)

        a = nuke.nodes.TransformGeo()

        a.setXYpos(node.xpos()-10, node.ypos() + 250)

        a.setInput(1,node)

        a.setInput(0,b)

        tl.append(a)




    nukescripts.clear_selection_recursive()


    c = nuke.createNode('Constant')



    c['selected'].setValue(False)

    c.knob('color').setValue([0.2,0.1,0.2,1])


    cm = nuke.createNode('Dot')

    cm.setXYpos(int(setXpos), node.ypos() - 250)

    cm.setInput(0,c)

    c.setXYpos(int(setXpos - 16), cm.ypos()-100)


    for node in sl:

        node.setInput(0,cm)


    nukescripts.clear_selection_recursive()


    sm = nuke.createNode('Scene')


    for num in range(len(tl)):

        tlo = tl[num]

        sm.setXYpos(int(setXpos) , tlo.ypos() + 100)

        sm.setInput(num,tlo)

    nukescripts.clear_selection_recursive()

    dot2 = nuke.createNode('Dot')
    dot2.knob('tile_color').setValue(927764223)
    dot2.setXYpos(sm.xpos() - 200, sm.ypos() + 200)
    dot2.knob('label').setValue('camera')
    dot2.knob('note_font_size').setValue(100)
    nukescripts.clear_selection_recursive()
    sm['selected'].setValue(True)


    ren = nuke.nodes.ScanlineRender()

    ren.setXYpos(sm.xpos()  , sm.ypos() + 400)

    ren.setInput(1,sm)
    ren.setInput(2,dot2)



def axisSet02():


    a = nuke.selectedNodes()
    setXpos = (int(a[-1].xpos()) + int(a[0].xpos()+100))/2

    pos = 0

    sl = []

    rl = []

    for node in a:

        b = nuke.nodes.Sphere()

        b['hide_input'].setValue('True')

        b.setXYpos(node.xpos()+ 50, node.ypos() + 200)

        b['uniform_scale'].setValue(1)

        sl.append(b)

        a = nuke.nodes.TransformGeo()

        a.setXYpos(node.xpos()-10, node.ypos() + 250)

        a.setInput(1,node)

        a.setInput(0,b)

        ren = nuke.nodes.ScanlineRender()

        ren.setXYpos(node.xpos()-16, node.ypos() + 400)

        ren.setInput(1,a)

        rl.append(ren)


    nukescripts.clear_selection_recursive()


    c = nuke.createNode('Constant')



    c['selected'].setValue(False)

    c.knob('color').setValue([0.2,0.1,0.2,1])


    cm = nuke.createNode('Dot')

    cm.setXYpos(int(setXpos), node.ypos() - 250)

    cm.setInput(0,c)

    c.setXYpos(int(setXpos) - 16, cm.ypos()-100)

    for node in sl:

        node.setInput(0,cm)


    nukescripts.clear_selection_recursive()


    dot2 = nuke.createNode('Dot')

    dot2.knob('tile_color').setValue(927764223)

    dot2.knob('label').setValue('camera')
    dot2.knob('note_font_size').setValue(100)

    for node in rl:

        dot2.setXYpos(cm.xpos() + 150 , cm.ypos())

        node.setInput(2,dot2)






def createAxisSet():
    cn = nuke.createNode('NoOp')
    cn.setName('axisSet')
#    ck0 = nuke.PyScript_Knob('00','set00')
    ck1 = nuke.PyScript_Knob('01','set01')
    ck2 = nuke.PyScript_Knob('02','set02')
#    cn.addKnob(ck0)
    cn.addKnob(ck1)
    cn.addKnob(ck2)
#    setVal0 = cn.knob('00')
#    setVal0.setValue('axisSet.axisSet00()')
    setVal1 = cn.knob('01')
    setVal1.setValue('axisSet.axisSet01()')
    setVal2 = cn.knob('02')
    setVal2.setValue('axisSet.axisSet02()')
    cn.knob('tile_color').setValue(847014143)
    cn.knob('label').setValue('-\nmade by sb')
    cn.knob('note_font').setValue('Bitstream Vera Sans Italic')
