########################################################
# make_Reder_Job_set
#
# Produced by : Jihoon Han
########################################################
# Last Update : 2018/12/07
# - Multiful Create Job Node set  //2018/04/05
# - Fixed Created Position //2018/12/07
#

#-----------------------------------------------------------

import hou

def createParmNode(pos):
    base = hou.selectedNodes()[0]
    rop_path = base.parent()
    summit = rop_path.createNode("DXC_submitter")
    summit.move(pos+hou.Vector2(0,-2))
    
    
    config = rop_path.createNode("DXC_config")
    config.move(pos+hou.Vector2(-2,-1))
    
    node = hou.selectedNodes()
    
    i = 0
    for nodes in node:
        summit.setInput(i,nodes)
        config.setInput(i,nodes)
        i += 1

def main():
    node = hou.selectedNodes()
    array_x = []
    array_y = []
    array_pos = []
    i = 0
    for nodes in node:
        pos = nodes.position()
        pos_x = pos[0]
        pos_y = pos[1]
        array_x.append(pos_x)
        array_y.append(pos_y)
   
    x = sorted(array_x)[0]
    y = sorted(array_y)[0]
    pos = hou.Vector2(x,y)

    try:    
        createParmNode(pos)
    except:
        hou.ui.displayMessage("Only ues RopNet")
