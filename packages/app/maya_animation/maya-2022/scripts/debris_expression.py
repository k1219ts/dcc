import maya.cmds as cmds
import random

def debris_expression(sel):
    debris_CON = cmds.circle(n='debris_CON')[0]
    cmds.addAttr(debris_CON, ln='offsetRX', at='double', k=1, dv=random.uniform(0, 100))
    cmds.addAttr(debris_CON, ln='offsetRY', at='double', k=1, dv=random.uniform(0, 100))
    cmds.addAttr(debris_CON, ln='offsetRZ', at='double', k=1, dv=random.uniform(0, 100))
    cmds.addAttr(debris_CON, ln='offsetTX', at='double', k=1, dv=0)
    cmds.addAttr(debris_CON, ln='offsetTY', at='double', k=1, dv=random.uniform(-1, -10))
    cmds.addAttr(debris_CON, ln='offsetTZ', at='double', k=1, dv=0)
    cmds.addAttr(debris_CON, ln='r_speed', at='double', k=1, dv=random.uniform(0, 1))
    cmds.addAttr(debris_CON, ln='t_speed', at='double', k=1, dv=random.uniform(0, 1))

    exprStr = 'float $ctime = `playbackOptions -q -min`;\n'
    exprStr += '{locator_r}.rotateX = (frame - $ctime + ({locator}.offsetRX + {object}.offsetRX)) * ({locator}.r_speed + {object}.r_speed);\n'
    exprStr += '{locator_r}.rotateY = (frame - $ctime + ({locator}.offsetRY + {object}.offsetRY)) * ({locator}.r_speed + {object}.r_speed);\n'
    exprStr += '{locator_r}.rotateZ = (frame - $ctime + ({locator}.offsetRZ + {object}.offsetRZ)) * ({locator}.r_speed + {object}.r_speed);\n'
    exprStr += '{locator_t}.tx = (frame - $ctime) * ({locator}.offsetTX + {object}.offsetTX) * ({locator}.t_speed + {object}.t_speed);\n'
    exprStr += '{locator_t}.ty = (frame - $ctime) * ({locator}.offsetTY + {object}.offsetTY) * ({locator}.t_speed + {object}.t_speed);\n'
    exprStr += '{locator_t}.tz = (frame - $ctime) * ({locator}.offsetTZ + {object}.offsetTZ) * ({locator}.t_speed + {object}.t_speed);\n'

    for object in sel:
        locator_r = cmds.spaceLocator(n=object + "_LOC_r")[0]
        locator_t = cmds.spaceLocator(n=object + "_LOC_t")[0]
        cmds.parent(locator_r, locator_t)
        cmds.parent(locator_t, debris_CON)
        objectTrans = cmds.xform(object, q=True, rp=True, ws=True)
        cmds.xform(locator_r, t=(objectTrans[0], objectTrans[1], objectTrans[2]))
        cmds.parentConstraint(locator_r, object, mo=True, w=1)

        cmds.addAttr(object, ln='offsetRX', at='double', k=1, dv=0)
        cmds.addAttr(object, ln='offsetRY', at='double', k=1, dv=0)
        cmds.addAttr(object, ln='offsetRZ', at='double', k=1, dv=0)
        cmds.addAttr(object, ln='offsetTX', at='double', k=1, dv=0)
        cmds.addAttr(object, ln='offsetTY', at='double', k=1, dv=-1)
        cmds.addAttr(object, ln='offsetTZ', at='double', k=1, dv=0)
        cmds.addAttr(object, ln='r_speed', at='double', k=1, dv=1)
        cmds.addAttr(object, ln='t_speed', at='double', k=1, dv=1)

        exprNewStr = exprStr.format(locator=debris_CON, locator_r=locator_r, object=object, locator_t=locator_t)
        cmds.expression(n=object + '_EXP', o=object, s=exprNewStr)

