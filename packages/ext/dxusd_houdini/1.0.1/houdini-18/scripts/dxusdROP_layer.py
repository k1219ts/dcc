import hou

def ResolveOP(this, op, type):
    if type == 'inst':
        # set ns layer and dprim
        if not this.parm('nslyrtgl').evalAsInt():
            this.parm('nslyr').set(op.name())
        if not this.parm('dprimtgl').evalAsInt():
            this.parm('dprim').set('World')

        # find sop path
        prt = ''
        pnt = ''
        for c in op.children():
            if c.type().name() == 'dxusdSOP_importPrototypes':
                prt = c.path()
            elif c.type().name() == 'dxusdSOP_instancingPoints':
                pnt = c.path()

        this.parm('inst_scatterprototypespath').set(prt)
        this.parm('inst_scatterpointspath').set(pnt)

        # set selected op
        this.parm('selectedop').set(pnt)

    elif type == 'feather':
        # set ns layer and dprim
        if not this.parm('nslyrtgl').evalAsInt():
            this.parm('nslyr').set(op.name())
        if not this.parm('dprimtgl').evalAsInt():
            this.parm('dprim').set('Feather')

        # set selected op
        exptype = this.parm('feather_exporttype').evalAsInt()
