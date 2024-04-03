import hou,ast,os

debug = 0

last_valid_selection = []
allowed = ['geo','instance','arnold_volume','arnold_procedural','material']
matnetworks = ['matnet','mat','shopnet','shop']
flashnets = {'Shop': ['vopmaterial', 'redshift_vopnet', 'arnold_vopnet', 'octane_vopnet'], 'Vop': ['arnold_materialbuilder', 'redshift_vopnet', 'octane_vopnet', 'materialbuilder']}

def getpref():

    pref = hou.expandString('$HOUDINI_USER_PREF_DIR/ring.pref')
    if os.path.isfile(pref):
        f = file(pref,'r')
        with open(pref, 'r') as f:
            recent = ast.literal_eval(f.read())
            
        return recent
    else:
        None

def setmatflags(nodes):

    ## Get visible material pane tabs

    tabs = hou.ui.paneTabs()
    editors = [t for t in tabs 
        if t.type().name() == 'NetworkEditor' and
            t.pwd().type().name() in matnetworks]

    flash = [t for t in tabs 
        if t.type().name() == 'NetworkEditor' and
            t.pwd().type().category().name() in flashnets and 
            t.pwd().type().name() in flashnets.get(t.pwd().type().category().name())]
    
    if not flash and not editors: return
    
    ## Set node flags

    flagtype = hou.nodeFlag.Footprint

    visiblemats = [child for e in editors for child in e.pwd().children()]

    with hou.undos.disabler():
        [n.setGenericFlag(flagtype, 1) if n in nodes else n.setGenericFlag(flagtype, 0) 
            for n in visiblemats]

    ## Set footprints

    if editors:
        [e.setFootprints(
            [hou.NetworkFootprint(flagtype,
                hou.ui.colorFromName('GraphOutputHighlight'), 1, False)]) 
                    for e in editors]

    ## Set flash messages
            
    if flash:
        if nodes:
            amt = len(nodes)
            for e in flash:
                curnode = e.pwd()
                if curnode in nodes:
                    icon = 'TOP_status_cooked'

                    if amt > 1:
                        msg = 'Current and %i Other'%(amt-1)
                    else:
                        msg = 'Current'

                else:
                    icon = 'TOP_status_error'

                    msg = "'%s' "%nodes[0].path()
                    if amt > 1:
                        msg += 'and %i Other'%(amt-1)

                if amt > 2:
                    msg += 's'

                e.flashMessage(icon,msg,3)
                
        else:
            [e.flashMessage(None,None,0) for e in flash]

def highlightmaterials():
    if debug:
        print 'Highlighting Materials.'

    global last_valid_selection

    pane = getpane()

    ## Filter 

    geonodes = [n for n in hou.selectedNodes() if validnode(n)]
    for n in geonodes:
        if n.type().category().name() == 'Object':
            if n.type().name() == 'instance':
                insnode = n.node(n.evalParm('instancepath'))

                if insnode:
                    geonodes.append(insnode)

    ## Remember Selection

    if geonodes:
        last_valid_selection = geonodes
    elif pane:
        if pane.pwd().type().name() in matnetworks:
            geonodes = last_valid_selection

    ## Get materials

    matlist,inslist = buildmatlist(geonodes,0)
    if inslist:
        matlist += buildmatlist(inslist,1)[0]

    if debug:
        print 'mat list',matlist
        print 'instance list',inslist

    setmatflags(matlist)

def buildmatlist(nodes,mode):
    matlist = []
    inslist = []

    for n in nodes:

        ## Add parameter callback

        try:
            addparmcallback(n)
        except:
            continue
            
        ## Get obj materials and children

        if n.type().category().name() == 'Object':

            geomat = n.node(n.evalParm('shop_materialpath'))
            if geomat:
                matlist.append(geomat)

            ## Catch material child sops

            if pref:
                if pref and pref[1]:
                    if debug:
                        print 'Finding child material SOPs.'

                    children = n.allSubChildren(1,0)

                    ## Look for material SOPs

                    if not mode:
                        matnodes = [c for c in children if c.type().name() == 'material']

                        for node in matnodes:
                            matlist = materialSOP(node,matlist)

                    ## Look for material attributes and instancest

                    try:
                        if pref[3]:
                            if debug:
                                print 'Finding child sop attributes(OBJ selection).'

                            matnodes = [c for c in children 
                                if [f for f in [hou.nodeFlag.Render,hou.nodeFlag.Display] 
                                    if c.isGenericFlagSet(f)]]
                            if matnodes:
                                for n in matnodes:
                                    matlist,inslist = getattribs(n,matlist,inslist)
                    except: pass

            continue

        ## Get SOP attributes

        elif pref:
            try:
                if pref[3]:
                    if debug:
                        print 'Finding sop attributes(SOP selection).'
                    matlist,inslist = getattribs(n,matlist,inslist)
                    #continue
            except:
                pass

        ## Else its a material SOP

        if n.type().name() == 'material' and not mode:
            matlist = materialSOP(n,matlist)

    return matlist,inslist

def materialSOP(node,matlist):
    num = node.evalParm('num_materials')
    if num:
        for p in range(num):
            p = 'shop_materialpath%i'%(p+1)
            parmmat = node.node(node.evalParm(p))
            if parmmat:
                matlist.append(parmmat)

    return matlist

def getattribs(n,matlist,inslist):
    prim = n.geometry().findPrimAttrib('shop_materialpath')
    if prim:
        matlist += [n.node(p) for p in prim.strings() if n.node(p)]
        
    point = n.geometry().findPointAttrib('shop_materialpath')
    if point:
        matlist += [n.node(p) for p in point.strings() if n.node(p)]

    point = n.geometry().findPointAttrib('instance')
    if point:
        inslist += [n.node(p) for p in point.strings() if n.node(p)]
        
    detail = n.geometry().findGlobalAttrib('shop_materialpath')
    if detail:
        matlist += [n.node(p) for p in detail.strings() if n.node(p)]

    return matlist,inslist

def fixpath(p):
    if p.startswith('op:'):
        p = p[3:]

    return p

def validnode(n):
    if n.type().category().name() == 'Object' and n.type().name() in allowed:
        return True
    elif n.type().category().name() == 'Sop':
        if n.type().name() == 'material':
            return True
        elif pref:
            try:
                #if pref[3] and [f for f in [hou.nodeFlag.Render,hou.nodeFlag.Display] if n.isGenericFlagSet(f)]:
                #    return True
                if pref[3]:
                    return True
            except: pass

    return False

def getpane():
    try:
        curdesk = hou.ui.curDesktop()
        activepane = curdesk.paneTabUnderCursor()
    except:
        return None

    if activepane:
        return activepane
    else:
        return None

## Callbacks

def addparmcallback(n):
    if not [e[1] for e in n.eventCallbacks() if e[1].func_name.startswith('footprint_')]:
        n.addEventCallback((hou.nodeEventType.ParmTupleChanged,), footprint_parmChange)

def footprint_parmChange(**kwargs):
    try:
        p = kwargs["parm_tuple"]
        n = kwargs["node"]

        if p.name().startswith('shop_materialpath'):
            highlightmaterials()
        elif n.type().name() == 'instance' and n.type().category().name() == 'Object' and p.name() == 'instancepath':
            highlightmaterials()
        elif n.type().name() == 'object_merge' and n.type().category().name() == 'Sop' and p.name().startswith('objpath'):
            highlightmaterials()
        elif n.type().name() == 'attribwrangle' and n.type().category().name() == 'Sop' and p.name() == 'snippet':
            highlightmaterials()
            
    except:
        return

def selectioncallback(selection):
    try:
        if pref and not pref[0]:
            return
    except: pass

    highlightmaterials()

## Camera Footprints

class viewportcallbacks:
    def __init__(self):
        self.cam = None
        self.curport = None

        if pref:
            try:
                if pref and not pref[2]:
                    return
            except: pass

        if debug:
            print 'Adding camera event.'
        hou.hipFile.addEventCallback(self.get_viewers_on_load)
        hou.ui.addSelectionCallback(self.selectionupdatecam)
        self.addcamcallback()

    ## Callbacks

    def get_viewers_on_load(self,event_type):
        if event_type == hou.hipFileEventType.AfterLoad:
            self.cam = None
            self.curport = None
            self.addcamcallback()

    def camChange(self,**kwargs):
        if kwargs['event_type'].name() == 'CameraSwitched':
            self.curport = kwargs['viewport']
            self.cam = self.curport.camera()
            self.camfootprints()

    def selectionupdatecam(self,selection):
        self.camfootprints()

    def footprintcam_parmChange(self,**kwargs):
        try:
            p = kwargs["parm_tuple"]
            if p.name() == 'camswitch':
                self.camfootprints()
        except:
            return

    def addparmcallback(self,n):
        if not [e[1] for e in n.eventCallbacks() if e[1].func_name.startswith('footprintcam_')]:
            n.addEventCallback((hou.nodeEventType.ParmTupleChanged,), self.footprintcam_parmChange)

    def addcamcallback(self):
        desktop = hou.ui.curDesktop()
        sceneViewer = [p.curViewport() 
            for p in desktop.paneTabs() 
                if p.type().name() == 'SceneViewer']
        
        for view in sceneViewer:
            if view.camera():
                self.curport = view
                self.cam = view.camera()

            if not [e for e in view.eventCallbacks() if e.func_name == 'camChange']:
                view.addEventCallback(self.camChange)

    def switcherresolve(self):
        cam = self.cam
        inputs = cam.inputs()

        if inputs:
            self.addparmcallback(self.cam)

            ind = cam.evalParm('camswitch')
            if len(inputs) > ind:
                return inputs[ind]

            else:
                return inputs[-1]

        return cam

    def valid(self,n):
        if n.type().name() in ['cam','switcher']:
            return True
        return False

    def camfootprints(self):

        if debug:
            print 'Setting camera footprints.'

        ## Resolve active camera

        switch = False

        if self.cam:
            try:# if cam deleted
                if self.cam != self.curport.camera():
                    self.cam = self.curport.camera()

                nodetype = self.cam.type().name()

                # resolve switch camera

                if nodetype == 'switcher':
                    cam = self.switcherresolve()
                    if cam:
                        switch = True
                else:
                    cam = self.cam

            except:
                cam = None

        else:
            cam = None

        ## Get visible object pane tabs

        tabs = hou.ui.paneTabs()
        editors = [t for t in tabs 
            if t.type().name() == 'NetworkEditor' and
                t.pwd().children() and 
                    t.pwd().children()[0].type().category().name() == 'Object']

        ## Set flags

        flagtype = [hou.nodeFlag.Footprint, hou.nodeFlag.Template]

        visibleobj = [child for e in editors for child in e.pwd().children() if self.valid(child)]

        with hou.undos.disabler():
            for n in visibleobj:
                n.setGenericFlag(flagtype[1], 
                    int(switch and self.cam == n))
                n.setGenericFlag(flagtype[0], 
                    int(n == cam))

        ## Set obj footprints

        objfootprints(editors)

## Light Footprints

class lightfootprints:
    def __init__(self):
        self.lights = ['hlight::2.0', 'arnold_light', 'octane_light', 'pxrspherelight', 'rslight', 'hlight::2.0', 'pxrrectlight', 'ambient', 'octane_spectron', 'rslightdome::2.0', 'envlight', 'pxrportallight', 'indirectlight', 'octane_toonLight', 'pxrmeshlight', 'rslighties', 'envlight', 'hlight::2.0', 'pxrdistantlight', 'pxrdomelight', 'rslightportal', 'indirectlight', 'pxrdisklight', 'rslightsun']
        self.parm = 'light_enable'
        self.lightlist = []

        try:
            if pref:
                if pref and not pref[4]:
                    return
        except: pass

        if debug:
            print 'Adding light event.'
        hou.hipFile.addEventCallback(self.get_lights_on_load)
        hou.ui.addSelectionCallback(self.selectionupdatelights)
        self.startup()

    def get_lights_on_load(self,event_type):
        if event_type == hou.hipFileEventType.AfterLoad:
            self.startup()
            if debug:
                print 'light after load callback'

    def startup(self):
        if debug:
            print 'getting all scene lights'

        self.lightlist = []

        [self.valid(n) for n in hou.node('/').allSubChildren(1,0)]

        self.lightfootprints()

    def selectionupdatelights(self,selection):
        if debug:
            print 'light selection callback'

        [self.valid(n) for n in selection]

        self.lightfootprints()

    def set_parm_callback(self,n):
        if not [e[1] for e in n.eventCallbacks() if e[1].func_name.startswith('footprintlight_')]:
            n.addEventCallback((hou.nodeEventType.ParmTupleChanged,), self.footprintlight_parmChange)

    def footprintlight_parmChange(self,**kwargs):
        try:
            p = kwargs["parm_tuple"]
            if p.name() == self.parm:
               self.lightfootprints()
        except:
            return

    def valid(self,n):
        try:
            if n.type().category().name() == 'Object' and n.type().name() in self.lights:
                if n not in self.lightlist:
                    self.lightlist.append(( n,self.isenabled(n) ))
                    self.set_parm_callback(n)

                return 1
        except: pass
        return 0

    def isenabled(self,n):
        return n.evalParm(self.parm)

    def lightfootprints(self):

        if debug:
            print 'Setting light footprints.'

        ## Get visible object pane tabs

        tabs = hou.ui.paneTabs()
        editors = [t for t in tabs 
            if t.type().name() == 'NetworkEditor' and
                t.pwd().children() and 
                    t.pwd().children()[0].type().category().name() == 'Object']

        ## Set flags

        flagtype = hou.nodeFlag.XRay

        visibleobj = [child for e in editors for child in e.pwd().children()]

        #if debug:
        #    print visibleobj

        with hou.undos.disabler():
            [n.setGenericFlag(flagtype, self.isenabled(n)) 
                if self.valid(n) else n.setGenericFlag(flagtype, 0) 
                    for n in visibleobj]

        ## Set obj footprints

        objfootprints(editors)


def objfootprints(editors):
    if pref:
        try:
            if not pref[4]:
                [e.setFootprints([
                    hou.NetworkFootprint(hou.nodeFlag.Footprint,
                    hou.ui.colorFromName('GraphDisplayHighlight'), 1, False),
                    hou.NetworkFootprint(hou.nodeFlag.Template,
                    hou.ui.colorFromName('GraphDisplayHighlight'), 0, False)
                    ]) 
                    for e in editors]

                return
        except: pass

    [e.setFootprints([
        hou.NetworkFootprint(hou.nodeFlag.Footprint,
        hou.ui.colorFromName('GraphDisplayHighlight'), 1, False),
        hou.NetworkFootprint(hou.nodeFlag.Template,
        hou.ui.colorFromName('GraphDisplayHighlight'), 0, False),
        hou.NetworkFootprint(hou.nodeFlag.XRay,
        hou.Color((.5,.5,.5)), 2, False)
        ]) 
        for e in editors]

pref = getpref()

if debug:
    print repr(pref)