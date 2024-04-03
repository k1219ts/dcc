import hou

def sprite(n):
    with hou.undos.group("Change to Sprite"):
        connected = [o for o in n.outputs() if o.type().name() == 'redshift::Material']
        pos = n.position()
        file = n.evalParm('tex0')
        name = n.name()
        cd = n.color()
        
        s = n.parent().createNode('redshift::Sprite',name)
        s.parm('tex0').set(file)
        s.setPosition(pos)
        s.setColor(cd)
        n.destroy()
        
        if connected:
            c = connected[0]
            output = c.outputs()
            
            s.setNextInput(c)
            
            if output:
                output[0].setInput(0,s,0)
                
            s.moveToGoodPosition(1,0,1,0)

def normal(n):
    with hou.undos.group("Change to Bump"):
        connected = n.outputConnections()
        pos = n.position()
        file = n.evalParm('tex0')
        scale = n.evalParm('scale')
        name = n.name()
        cd = n.color()
        
        b = n.parent().createNode('redshift::BumpMap', None)
        b.parm('inputType').set('1')
        b.parm('scale').set(scale)
        b.setPosition(pos)
        n.destroy()
        
        t = b.createInputNode(0,'redshift::TextureSampler',name)
        t.parm('tex0').set(file)
        t.setPosition(pos)
        t.setColor(cd)
        t.move((-3,0))
        
        if connected:
            c = connected[0]
            ind = c.inputIndex()
            
            c.outputNode().setInput(ind,b)            