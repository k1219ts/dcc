import hou

def takeSecond(elem):
    return elem[1]

def arrage(node):
    inserts = node.inputs()
    if not inserts: return

    sortpos = []
    for i in inserts:
        node.setInput(0,None,0)
        sortpos.append((i,i.position().x()))
        
    sortpos.sort(key=takeSecond)
    
    [node.setNextInput(n[0]) for n in sortpos]