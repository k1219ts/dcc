import maya.cmds as cmds
import maya.OpenMaya as OpenMaya

widgets = {}

def softDeformerUI():
    """UI for the whole thing"""

    if cmds.window("softModWin", exists = True):
        cmds.deleteUI("softModWin")
    widgets["window"] = cmds.window("softModWin", t="softDeformer", w=300, h=130)
    widgets["tabLO"] = cmds.tabLayout()
    widgets["smCLO"] = cmds.columnLayout("SoftModDeformer", w=300)

    cmds.separator(h=10)
    widgets["smdTFG"] = cmds.textFieldGrp(l="Deformer Name", w=300, cw=[(1,100),(2,190)], cal=[(1,"left"), (2, "left")], tx="softMod_DEF")
    widgets["firstVertCBG"] = cmds.checkBoxGrp(l="Use only 1st vert (vs. avg pos)", v1=0, cw=[(1,200)], cal=[(1,"left"), (2,"left")])
    widgets["parentCBG"] = cmds.checkBoxGrp(l="Parent Ctrl Under Geo?", v1=0, cw=[(1,200)], cal=[(1,"left"), (2,"left")])
    widgets["incrCBG"] = cmds.checkBoxGrp(l="Increment name after creation?", v1=1, cw=[(1,200)], cal=[(1,"left"), (2,"left")])
    widgets["checkCBG"] = cmds.checkBoxGrp(l="AutoCheck if there are deformers?", v1=1, cw=[(1,200)], cal=[(1,"left"), (2,"left")])
    widgets["scaleFFG"] = cmds.floatFieldGrp(l="Control Scale", v1=1, pre=2, cw=[(1,100),(2,50)], cal=[(1,"left"),(2,"left")])

    cmds.separator(h=10, style="single")
    widgets["button"] = cmds.button(l="Create Deformer", w=300, h=40, bgc=(.6,.8,.6), c=softModDeformerDo)

    #second tab to softselect deformer
    cmds.setParent(widgets["tabLO"])
    widgets["ssCLO"] = cmds.columnLayout("SoftSelectDeformer", w=300)
    widgets["ssdTFG"] = cmds.textFieldGrp(l="Deformer Name", w=300, cw=[(1,100),(2,190)], cal=[(1,"left"), (2, "left")], tx="softSelect_DEF")
    widgets["ssCPOMCBG"] = cmds.checkBoxGrp(l="Control to closest point on mesh?", v1=1, cw=[(1,200)], cal=[(1,"left"), (2,"left")])
    widgets["ssParentCBG"] = cmds.checkBoxGrp(l="Parent Ctrl Under Geo?", v1=0, cw=[(1,200)], cal=[(1,"left"), (2,"left")])
    widgets["ssIncrCBG"] = cmds.checkBoxGrp(l="Increment name after creation?", v1=1, cw=[(1,200)], cal=[(1,"left"), (2,"left")])
    widgets["ssCheckCBG"] = cmds.checkBoxGrp(l="AutoCheck if there are deformers?", v1=1, cw=[(1,200)], cal=[(1,"left"), (2,"left")])
    widgets["ssScaleFFG"] = cmds.floatFieldGrp(l="Control Scale", v1=1, pre=2, cw=[(1,100),(2,50)], cal=[(1,"left"),(2,"left")])
    cmds.separator(h=10, style="single")
    widgets["button"] = cmds.button(l="Create Deformer", w=300, h=40, bgc=(.6,.8,.6), c=softSelectDef)


    cmds.showWindow(widgets["window"])

# --------------------------
# softMod deformer
# --------------------------

def softModDeformerDo(*args):
    """creates and sets up the softmod deformer setup"""

    check = cmds.checkBoxGrp(widgets["checkCBG"], q=True, v1=True)
    increment = cmds.checkBoxGrp(widgets["incrCBG"], q=True, v1=True)
    toParent = cmds.checkBoxGrp(widgets["parentCBG"], q=True, v1=True)
    #get deformer name
    defName = cmds.textFieldGrp(widgets["smdTFG"], tx=True, q=True)
    scaleFactor = cmds.floatFieldGrp(widgets["scaleFFG"], q=True, v1=True)

    if not (cmds.objExists(defName)):
        # choose a vert (for example)
        vertsRaw = cmds.ls(sl=True, fl=True)

        if vertsRaw == []:
            cmds.warning("Must select at least one vertex")
        else:
            if (cmds.checkBoxGrp(widgets["firstVertCBG"], q=True, v1=True)):
                vertex = [vertsRaw[0]]
            else:
                vertex = vertsRaw

        obj = vertex[0].partition(".")[0]

        #get vert position then select the geo
        positions = []
        for vert in vertex:
            positions.append(cmds.pointPosition(vert))

        numVerts = len(positions)

        x,y,z = 0,0,0
        for i in range(numVerts):
            x += positions[i][0]
            y += positions[i][1]
            z += positions[i][2]

        vertPos = [(x/numVerts), (y/numVerts), (z/numVerts)]

        #check if there are other deformers on the obj
        if check:
            deformers = []
            deformers = getDeformers(obj)
            if deformers:
                cmds.confirmDialog( title='Deformer Alert!', message='Found some deformers on %s.\nYou may want to put the softmod\n early in the input list'%obj, button=['OK'], defaultButton='OK', cancelButton='OK', dismissString='OK' )

        cmds.select(obj)

        # create a soft mod at vert position (avg)
        softMod = defName
        softModOrig = cmds.softMod(relative=False, falloffCenter = vertPos, falloffRadius=5.0, n=softMod)[0]
        cmds.rename(softModOrig, softMod)
        softModXform = cmds.listConnections(softModOrig, type="transform")[0]


        # create a control at the position of the softmod
        control = defName + "_CTRL"
        cmds.curve(n=control, d=1, p=[[0.0, 1.0, 0.0], [-0.382683, 0.92388000000000003, 0.0], [-0.70710700000000004, 0.70710700000000004, 0.0], [-0.92388000000000003, 0.382683, 0.0], [-1.0, 0.0, 0.0], [-0.92388000000000003, -0.382683, 0.0], [-0.70710700000000004, -0.70710700000000004, 0.0], [-0.382683, -0.92388000000000003, 0.0], [0.0, -1.0, 0.0], [0.382683, -0.92388000000000003, 0.0], [0.70710700000000004, -0.70710700000000004, 0.0], [0.92388000000000003, -0.382683, 0.0], [1.0, 0.0, 0.0], [0.92388000000000003, 0.382683, 0.0], [0.70710700000000004, 0.70710700000000004, 0.0], [0.382683, 0.92388000000000003, 0.0], [0.0, 1.0, 0.0], [0.0, 0.92388000000000003, 0.382683], [0.0, 0.70710700000000004, 0.70710700000000004], [0.0, 0.382683, 0.92388000000000003], [0.0, 0.0, 1.0], [0.0, -0.382683, 0.92388000000000003], [0.0, -0.70710700000000004, 0.70710700000000004], [0.0, -0.92388000000000003, 0.382683], [0.0, -1.0, 0.0], [0.0, -0.92388000000000003, -0.382683], [0.0, -0.70710700000000004, -0.70710700000000004], [0.0, -0.382683, -0.92388000000000003], [0.0, 0.0, -1.0], [0.0, 0.382683, -0.92388000000000003], [0.0, 0.70710700000000004, -0.70710700000000004], [0.0, 0.92388000000000003, -0.382683], [0.0, 1.0, 0.0], [-0.382683, 0.92388000000000003, 0.0], [-0.70710700000000004, 0.70710700000000004, 0.0], [-0.92388000000000003, 0.382683, 0.0], [-1.0, 0.0, 0.0], [-0.92388000000000003, 0.0, 0.382683], [-0.70710700000000004, 0.0, 0.70710700000000004], [-0.382683, 0.0, 0.92388000000000003], [0.0, 0.0, 1.0], [0.382683, 0.0, 0.92388000000000003], [0.70710700000000004, 0.0, 0.70710700000000004], [0.92388000000000003, 0.0, 0.382683], [1.0, 0.0, 0.0], [0.92388000000000003, 0.0, -0.382683], [0.70710700000000004, 0.0, -0.70710700000000004], [0.382683, 0.0, -0.92388000000000003], [0.0, 0.0, -1.0], [-0.382683, 0.0, -0.92388000000000003], [-0.70710700000000004, 0.0, -0.70710700000000004], [-0.92388000000000003, 0.0, -0.382683], [-1.0, 0.0, 0.0]])

        cmds.select(cl=True)
    #TO-DO----------------pull this out into separate function?? Args would be object and color
        shapes = cmds.listRelatives(control, shapes=True)
        for shape in shapes:
            cmds.setAttr("%s.overrideEnabled"%shape, 1)
            cmds.setAttr("%s.overrideColor"%shape, 14)
        controlGrp = cmds.group(control, n="%s_GRP"%control)
        cmds.xform(controlGrp, ws=True, t=vertPos)

        # connect the pos, rot, scale of the control to the softModHandle
        cmds.connectAttr("%s.translate"%control, "%s.translate"%softModXform)
        cmds.connectAttr("%s.rotate"%control, "%s.rotate"%softModXform)
        cmds.connectAttr("%s.scale"%control, "%s.scale"%softModXform)

        cmds.addAttr(control, ln="__XTRA__", at="enum", k=True)
        cmds.setAttr("%s.__XTRA__"%control, l=True)

        # cmds.addAttr(control, ln="centerCtrlVis", at="bool", min=0, max=1, k=True, dv=0)
        cmds.addAttr(control, ln="envelope", at="float", min=0, max=1, k=True, dv=1)
        cmds.addAttr(control, ln="falloff", at="float", min=0, max=100, k=True, dv=5)
        # cmds.addAttr(control, ln="centerX", at="float", dv=0, k=True)
        # cmds.addAttr(control, ln="centerY", at="float", dv=0, k=True)
        # cmds.addAttr(control, ln="centerZ", at="float", dv=0, k=True)

        # connect that attr to the softmod falloff radius
        cmds.connectAttr("%s.envelope"%control, "%s.envelope"%softMod)
        cmds.connectAttr("%s.falloff"%control, "%s.falloffRadius"%softMod)

        # inherit transforms on softModHandle are "off"
        cmds.setAttr("%s.inheritsTransform"%softModXform, 0)

        # centerName = defName + "_center_CTRL"
        # #create secondary (area of influence) control here
        # centerCtrl = cmds.curve(n=centerName, d=1, p=[[-1.137096, -1.137096, 1.137096], [-1.137096, 1.137096, 1.137096], [1.137096, 1.137096, 1.137096], [1.137096, -1.137096, 1.137096], [-1.137096, -1.137096, 1.137096], [-1.137096, -1.137096, -1.137096], [-1.137096, 1.137096, -1.137096], [-1.137096, 1.137096, 1.137096], [1.137096, 1.137096, 1.137096], [1.137096, 1.137096, -1.137096], [1.137096, -1.137096, -1.137096], [1.137096, -1.137096, 1.137096], [1.137096, -1.137096, -1.137096], [-1.137096, -1.137096, -1.137096], [-1.137096, 1.137096, -1.137096], [1.137096, 1.137096, -1.137096]])

        # centerCtrlSh = cmds.listRelatives(centerCtrl, s=True)
        # for shape in centerCtrlSh:
        # #turn on overrides
        # cmds.setAttr("%s.overrideEnabled"%shape, 1)
        # cmds.connectAttr("%s.centerCtrlVis"%control, "%s.overrideVisibility"%shape)
        # cmds.setAttr("%s.overrideColor"%shape, 13)

        # centerGrp = cmds.group(centerCtrl, n="%s_GRP"%centerName)
        # #turn off scale and rotation for the center control
        # cmds.setAttr("%s.rotate"%centerCtrl, k=False, l=True)
        # cmds.setAttr("%s.scale"%centerCtrl, k=False, l=True)
        # cmds.setAttr("%s.visibility"%centerCtrl, k=False, l=True)

        # #move the group to the location
        # cmds.xform(centerGrp, ws=True, t=vertPos)

        # plusName = defName + "_plus"
        # plusNode = cmds.shadingNode("plusMinusAverage", asUtility=True, n=plusName)

        # cmds.connectAttr("%s.translate"%centerGrp, "%s.input3D[0]"%plusNode, f=True)
        # cmds.connectAttr("%s.translate"%centerCtrl, "%s.input3D[1]"%plusNode, f=True)
        # cmds.connectAttr("%s.output3D"%plusNode, "%s.falloffCenter"%softMod, f=True)

        #hide the softmod
        cmds.setAttr("%s.visibility"%softModXform, 0)

        #group the group and the softmod xform
        # defGroup = cmds.group(softModXform, controlGrp, n=(defName + "_deform_GRP"))
        defGroup = cmds.group(empty=True, n=(defName + "_deform_GRP"))
        cmds.xform(defGroup, ws=True, t=vertPos)
        cmds.parent(softModXform, controlGrp, defGroup)
        #parent the softmod under the centerCtrl
        # cmds.parent(defGroup, centerCtrl)

        #parent that group under the obj?
        if toParent:
            cmds.parent(defGroup, obj)

        # #connect group translate to plus node
        # plusName = defName + "_plus"
        # plusNode = cmds.shadingNode("plusMinusAverage", asUtility=True, n=plusName)
        # cmds.connectAttr("%s.translate"%defGroup, "%s.input3D[0]"%plusNode)

        # #connect to falloff center
        # cmds.connectAttr("%s.centerX"%control, "%s.input3D[1].input3Dx"%plusNode)
        # cmds.connectAttr("%s.centerY"%control, "%s.input3D[1].input3Dy"%plusNode)
        # cmds.connectAttr("%s.centerZ"%control, "%s.input3D[1].input3Dz"%plusNode)

        # cmds.connectAttr("%s.output3D"%plusNode, "%s.falloffCenter"%softMod)

        #scale the controls
        scaleCtrl([control], scaleFactor)

        #increment name
        if increment == 1:
            print "trying to rename"
            split = defName.rpartition("_")
            end = split[2]
            isInt = integerTest(end)

            if isInt:
                newNum = int(end) + 1
                newName = "%s%s%02d"%(split[0], split[1], newNum)
                cmds.textFieldGrp(widgets["smdTFG"], tx=newName, e=True)
            else:
                newName = "%s_01"%defName
                cmds.textFieldGrp(widgets["smdTFG"], tx=newName, e=True)

        #select the control to wrap up
        cmds.select(control)
    else:
        cmds.warning("An object of this name, %s, already exists! Choose a new name!"%defName)


# --------------------------
# softSelection deformer
# --------------------------

#TO-DO----------------checks on whether a) something is selected b) vertices are selected
def getSoftSelection():
    """from brian ???, gets the list of softselection components and passes out the verts and the weights"""
    #Grab the soft selection
    selection = OpenMaya.MSelectionList()
    softSelection = OpenMaya.MRichSelection()
    OpenMaya.MGlobal.getRichSelection(softSelection)
    softSelection.getSelection(selection)

    dagPath = OpenMaya.MDagPath()
    component = OpenMaya.MObject()

    # Filter Defeats the purpose of the else statement
    iter = OpenMaya.MItSelectionList( selection,OpenMaya.MFn.kMeshVertComponent )
    elements, weights = [], []
    while not iter.isDone():
        iter.getDagPath( dagPath, component )
        dagPath.pop() #Grab the parent of the shape node
        node = dagPath.fullPathName()
        fnComp = OpenMaya.MFnSingleIndexedComponent(component)
        getWeight = lambda i: fnComp.weight(i).influence() if fnComp.hasWeights() else 1.0

        for i in range(fnComp.elementCount()):
            elements.append('%s.vtx[%i]' % (node, fnComp.element(i)))
            weights.append(getWeight(i))
        iter.next()

    return elements, weights


def softSelectDef(*args):
    """calls on getSoftSelection() to get the weights of the softSelect and then puts it all under a cluster and a control"""

    ssDefName = cmds.textFieldGrp(widgets["ssdTFG"], q=True, tx=True)

    if not cmds.objExists("%s_CLS"%ssDefName):
        ssScale = cmds.floatFieldGrp(widgets["ssScaleFFG"], q=True, v1=True)
        ssIncrement = cmds.checkBoxGrp(widgets["ssIncrCBG"], q=True, v1=True)
        ssCheck = cmds.checkBoxGrp(widgets["ssCheckCBG"], q=True, v1=True)
        ssParent = cmds.checkBoxGrp(widgets["ssParentCBG"], q=True, v1=True)
        ssCPOM = cmds.checkBoxGrp(widgets["ssCPOMCBG"], q=True, v1=True)

        #this gets the verts selected and their respective weights in the soft selection
        elements,weights = getSoftSelection()

        #get transform and mesh
        xform = elements[0].partition(".")[0]
        #maybe here I should check for "orig", etc and exclude them?
        mesh = cmds.listRelatives(xform, f=True, s=True)[0]

        #check if there are other deformers on the obj
        if ssCheck:
            deformers = []
            deformers = getDeformers(xform)
            if deformers:
                cmds.confirmDialog( title='Deformer Alert!', message='Found some deformers on %s.\nYou may want to put the softmod\n early in the input list'%xform, button=['OK'], defaultButton='OK', cancelButton='OK', dismissString='OK' )

        #select each of the points from the list and create a cluster
        cmds.select(cl=True)
        for elem in elements:
            cmds.select(elem, add=True)

        clus = cmds.cluster(relative=True, name="%s_CLS"%ssDefName)

        for i in range(len(elements)):
            element = elements[i]
            value = weights[i]
            #percent -v 0.5 thisCluster pSphere1.vtx[241] ;
            cmds.percent(clus[0], element, v=value, )

        #get cluster position
        clusPos = cmds.xform(clus[1], ws=True, q=True, rp=True)

        if ssCPOM:

            #create closest point on mesh (surface?) node
            cpomNode = cmds.shadingNode("closestPointOnMesh", asUtility=True, n="%s_CPOM"%ssDefName)

            #--------------------
            #inputs and outputs for "closestPointOnMesh":

            #inputs:
            #"mesh"->"inputMesh" (mesh node of transform)
            #"clusPos"->"inPosition"
            #"worldMatrix"(transform of object)->"inputMatrix"

            #outputs:
            #"position"->surfacepoint in space
            #"u"->parameter u
            #"v"->parameter v
            #"normal"->normal vector
            #---------------------

            #connect up object to cpom
            cmds.connectAttr("%s.outMesh"%mesh, "%s.inMesh"%cpomNode)
            cmds.setAttr("%s.inPosition"%cpomNode, clusPos[0], clusPos[1], clusPos[2])
            cmds.connectAttr("%s.worldMatrix"%mesh, "%s.inputMatrix"%cpomNode)

            cpomPos = cmds.getAttr("%s.position"%cpomNode)[0]

            #delete cpom node
            cmds.delete(cpomNode)

        else:
            cpomPos = avgElementPos(elements)
    #TO-DO----------------see if you can't orient things to the verts (maybe only with cpom?)

        #now create a control
        control = "%s_CTRL"%ssDefName
        cmds.curve(n=control, d=1, p=[[0.0, 1.0, 0.0], [-0.382683, 0.92388000000000003, 0.0], [-0.70710700000000004, 0.70710700000000004, 0.0], [-0.92388000000000003, 0.382683, 0.0], [-1.0, 0.0, 0.0], [-0.92388000000000003, -0.382683, 0.0], [-0.70710700000000004, -0.70710700000000004, 0.0], [-0.382683, -0.92388000000000003, 0.0], [0.0, -1.0, 0.0], [0.382683, -0.92388000000000003, 0.0], [0.70710700000000004, -0.70710700000000004, 0.0], [0.92388000000000003, -0.382683, 0.0], [1.0, 0.0, 0.0], [0.92388000000000003, 0.382683, 0.0], [0.70710700000000004, 0.70710700000000004, 0.0], [0.382683, 0.92388000000000003, 0.0], [0.0, 1.0, 0.0], [0.0, 0.92388000000000003, 0.382683], [0.0, 0.70710700000000004, 0.70710700000000004], [0.0, 0.382683, 0.92388000000000003], [0.0, 0.0, 1.0], [0.0, -0.382683, 0.92388000000000003], [0.0, -0.70710700000000004, 0.70710700000000004], [0.0, -0.92388000000000003, 0.382683], [0.0, -1.0, 0.0], [0.0, -0.92388000000000003, -0.382683], [0.0, -0.70710700000000004, -0.70710700000000004], [0.0, -0.382683, -0.92388000000000003], [0.0, 0.0, -1.0], [0.0, 0.382683, -0.92388000000000003], [0.0, 0.70710700000000004, -0.70710700000000004], [0.0, 0.92388000000000003, -0.382683], [0.0, 1.0, 0.0], [-0.382683, 0.92388000000000003, 0.0], [-0.70710700000000004, 0.70710700000000004, 0.0], [-0.92388000000000003, 0.382683, 0.0], [-1.0, 0.0, 0.0], [-0.92388000000000003, 0.0, 0.382683], [-0.70710700000000004, 0.0, 0.70710700000000004], [-0.382683, 0.0, 0.92388000000000003], [0.0, 0.0, 1.0], [0.382683, 0.0, 0.92388000000000003], [0.70710700000000004, 0.0, 0.70710700000000004], [0.92388000000000003, 0.0, 0.382683], [1.0, 0.0, 0.0], [0.92388000000000003, 0.0, -0.382683], [0.70710700000000004, 0.0, -0.70710700000000004], [0.382683, 0.0, -0.92388000000000003], [0.0, 0.0, -1.0], [-0.382683, 0.0, -0.92388000000000003], [-0.70710700000000004, 0.0, -0.70710700000000004], [-0.92388000000000003, 0.0, -0.382683], [-1.0, 0.0, 0.0]])
        cmds.select(cl=True)

        #scale the control
        scaleCtrl([control], ssScale)

        shapes = cmds.listRelatives(control, shapes=True)
        for shape in shapes:
            cmds.setAttr("%s.overrideEnabled"%shape, 1)
            cmds.setAttr("%s.overrideColor"%shape, 14)
        controlGrp = cmds.group(control, n="%s_GRP"%control)

        #put the control at the cpomPos
        cmds.xform(controlGrp, ws=True, t=(cpomPos[0],cpomPos[1],cpomPos[2]))

        clusHandShape = cmds.listRelatives(clus[1], s=True)

        # #move the cluster control to the control space (weighted node)
        cmds.cluster(clus[0], e=True, bs=1, wn=(control, control))

        cmds.setAttr("%s.originX"%clusHandShape[0], 0.0)
        cmds.setAttr("%s.originY"%clusHandShape[0], 0.0)
        cmds.setAttr("%s.originZ"%clusHandShape[0], 0.0)

        cmds.delete(clus[1])

        cmds.setAttr("%s.visibility"%clusHandShape[0], 0)

        if ssParent:
            cmds.parent(controlGrp, xform)

        cmds.select(control, r=True)

        if ssIncrement == 1:
            print "trying to rename"
            split = ssDefName.rpartition("_")
            end = split[2]
            isInt = integerTest(end)

            if isInt:
                newNum = int(end) + 1
                newName = "%s%s%02d"%(split[0], split[1], newNum)
                cmds.textFieldGrp(widgets["ssdTFG"], tx=newName, e=True)
            else:
                newName = "%s_01"%ssDefName
                cmds.textFieldGrp(widgets["ssdTFG"], tx=newName, e=True)
    else:
        cmds.warning("An object/cluster of that name already exists! Please choose another name!")

# ------------------------
# helper functions
# ------------------------

def integerTest(test, *args):
    """use to test if a variable is an integer"""
    try:
        int(test)
        return True
    except:
        return False

def getDeformers(obj, *args):
    """gets a list of deformers on the passed obj"""
    history = cmds.listHistory(obj)
    Arrdeformers = []
    for node in history:
        types = cmds.nodeType(node, inherited = True)
        if "geometryFilter" in types:
            Arrdeformers.append(types[1])
    return Arrdeformers

def scaleCtrl(objs=[], scale=1.0, *args):
    """scales all of the cvs of selected controls and scales them by 'scale' value"""
    for obj in objs:
        cmds.select("%s.cv[:]"%obj, r=True)
        cmds.scale(scale, scale, scale)
        cmds.select(cl=True)

def avgElementPos(verts, *args):
    """uses a list of verts and gets the average position"""
    #get a selection of verts and avg their position
    xVal = []
    yVal = []
    zVal = []
    xAll = 0.0
    yAll = 0.0
    zAll = 0.0

    for vert in verts:
        pos = cmds.pointPosition(vert)
        xVal.append(pos[0])
        yVal.append(pos[1])
        zVal.append(pos[2])

    for x in xVal:
        xAll += x
    for y in yVal:
        yAll += y
    for z in zVal:
        zAll += z

    avgX = xAll/len(xVal)
    avgY = yAll/len(yVal)
    avgZ = zAll/len(zVal)

    avgPos = (avgX, avgY, avgZ)

    return avgPos


def softDeformer():
    """Use this to start the script!"""
    softDeformerUI()