import sys, traceback, math

import pymel.core as pm
import pymel.core.nodetypes as nt
import pymel.core.datatypes as dt

import ch_cmn as cmn
import ch_rig as rig

import bakeRolling_prebake as prebake


BKRTAG = 'bakeRolling_targetedTransform'
BKRORGTAG = 'bakeRolling_targetedTransform_orgRadius'
PRJTITLE = 'Dx_bakeRolling'

MSGATTRS = {
    'rotate' :'bkr_rotateGrpNode',
    'ground' :'bkr_groundGrpNode',
    'cached' :'bkr_cachedGrpNode',
    'rolling':'bkr_rollingGrpNode',
    'ctr'    :'bkr_ctrNode',
    'subCtr' :'bkr_subCtrNode',
    'customCtr' :'bkr_customCtrNode',
    'exp'    :'bkr_expNode',
    'customQuat':'bkr_customQuatNode',
    'customRotate':'bkr_customRotateNode'
}
EXPMSGATTRS = {
    'expCache':'bkr_cachedRollingNode',
    'expCustom':'bkr_customRollingNode',
    'expCtr':'bkr_bkrCtrNode',
    'expCustomCtr':'bkr_customCtrNode'
}

# only for test home
#class tmpCls:
#    def __init__(self):#        pass
#nt.DxRig = tmpCls

class functions:
    def __init__(self, ui):
        self.ui = ui
        self.taggedLists = {}

        self.ui.bkr_update_btn.clicked.connect(lambda: self.updateList())
        self.ui.bkr_pickSelected_btn.clicked.connect(lambda: self.pickSelected())
        self.ui.bkr_remove_btn.clicked.connect(lambda: self.bkrRemvoe())


        self.ui.bkr_prebake_btn.clicked.connect(lambda: self.prebake())
        self.ui.bkr_finalBake_btn.clicked.connect(lambda: self.finalBake())

        self.ui.bkr_groundCtr_add_btn.clicked.connect(lambda: self.groundCtr_add())
        self.ui.bkr_fitCustomCtrAxis_btn.clicked.connect(lambda: self.fitCustomCtrAxis())
        self.ui.bkr_prebakeRemove_btn.clicked.connect(lambda: self.prebakeRemove())

        self.ui.bkr_finalbakeRemove_btn.clicked.connect(lambda: self.finalbakeRemove())

        self.ui.bkr_addTag_btn.clicked.connect(lambda: self.addTag())
        self.ui.bkr_removeTag_btn.clicked.connect(lambda: self.removeTag())
        self.ui.bkr_selectTag_btn.clicked.connect(lambda: self.selectTag())
        self.ui.bkr_setRadiusTag_btn.clicked.connect(lambda: self.setRadius())
        self.ui.bkr_connectScale_btn.clicked.connect(lambda: self.connectScale())

        self.updateList()

    # --------------------------------------------------------------------------
    # ui funcionts
    def finalbakeRemove(self):
        obj, dxRig, name = self.getTaggedObject(asList=True)
        bkrGrp    = self.getBkrGroup(obj, name, err=True)
        rotate = bkrGrp.attr(MSGATTRS['rotate']).inputs()[0]
        rolling = bkrGrp.attr(MSGATTRS['rolling']).inputs()[0]

        cmn.confirmDialog('Do you want to remove final baking(%s)?'% bkrGrp.name(), 1)

        try:
            for attr in ['rx', 'ry', 'rz']:
                inputs = rotate.attr(attr).inputs(p=True)[0]
                inputs // rotate.attr(attr)
        except:
            pass

        rolling.r >> rotate.r


    def updateList(self):
        self.taggedLists.clear()
        self.ui.bkr_selection_combox.clear()

        for obj in pm.ls(type=nt.Transform):
            if obj.hasAttr(BKRTAG):
                dxRig = self._getDxRig(obj)

                name = cmn.shortNameOf(dxRig).split('_')
                if name[-1] in ['GRP', 'NUL', 'g']:
                    name.pop(-1)

                name = '_'.join(name)

                info  = { dxRig.name():(obj, dxRig, name) }
                self.taggedLists.update(info)

        selItems = ['------ Select ------']
        selItems.extend(self.taggedLists.keys())
        self.ui.bkr_selection_combox.addItems(selItems)

    def pickSelected(self):
        pm.select(self.getTaggedObject())


    def bkrRemvoe(self):
        obj, dxRig, name = self.getTaggedObject(asList=True)
        bkrGrp    = self.getBkrGroup(obj, name, err=True)
        groundGrp = self.getGroundGroup(bkrGrp)

        cmn.confirmDialog('Do you want to remove %s?'% bkrGrp.name(), 1)

        groundCtrs = groundGrp.getChildren()
        for ctr in groundCtrs:
            ctr.setParent(world=True)

        dels = []
        for attr in MSGATTRS.values():
            inputs = bkrGrp.attr(attr).inputs()
            dels.extend(inputs)

        pm.delete(dels)
        pm.delete(bkrGrp)


    def groundCtr_add(self):
        obj, dxRig, name = self.getTaggedObject(asList=True)
        bkrGrp    = self.getBkrGroup(obj, name)
        groundGrp = self.getGroundGroup(bkrGrp)

        ctr = rig.createCtr('ground', name+'_groundCtr1', groundGrp, 'skyblue')

    def fitCustomCtrAxis(self):
        obj, dxRig, name = self.getTaggedObject(asList=True)
        bkrGrp    = self.getBkrGroup(obj, name, err=True)

        ctr = bkrGrp.attr(MSGATTRS['ctr']).inputs()[0]
        customCtr = bkrGrp.attr(MSGATTRS['customCtr']).inputs()[0]
        cachedGrp = bkrGrp.attr(MSGATTRS['cached']).inputs()[0]
        rotate = bkrGrp.attr(MSGATTRS['rotate']).inputs()[0]
        caches = cachedGrp.getChildren()

        gc = ctr.groundChange.get() - 1
        idcs = [int(gc)]
        gcws = [1-(gc-idcs[0])]

        if gcws[0] < 1:
            gcws.append(1-gcws[0])
            idcs = idcs[0] + 1

        vo = rig.XUP
        vd = rig.ZEROV
        w = 0
        for i in range(len(idcs)):
            _v = dt.Vector( caches[idcs[i]].outputQuatX.get(),
                            caches[idcs[i]].outputQuatY.get(),
                            caches[idcs[i]].outputQuatZ.get())
            _v.normalize()
            vd += _v * gcws[i]

            w += math.acos(caches[idcs[i]].outputQuatW.get()) * gcws[i]

        vd *= dt.Matrix(rotate.getParent().getMatrix(ws=True))
        q = rig.quat(vo.cross(vd), vo.angle(vd))

        r = rig.quatToEuler(q)
        customCtr.r.set(r)
        customCtr.rolling.set(math.degrees(w))


    def prebake(self):
        rotateGrp, dxRig, name = self.getTaggedObject(asList=True)
        bkrGrp      = self.getBkrGroup(rotateGrp, name, err=True)
        groundGrp   = self.getGroundGroup(bkrGrp)
        ctr, subCtr, customCtr, customQuat = self.getBkrCtr(bkrGrp, rotateGrp, name)
        cachedGrp   = self.getCachedGroup(bkrGrp)
        rollingGrp  = self.getRollingGroup(bkrGrp)
        radius      = rotateGrp.attr(BKRTAG).get()
        cachingGrps = prebake.doit(rotateGrp, groundGrp,
                                   ctr, subCtr, cachedGrp, name, radius)


        msgAttr1 = MSGATTRS['exp']
        msgAttr2 = MSGATTRS['customRotate']

        if not (bkrGrp.hasAttr(msgAttr1) and \
                bkrGrp.hasAttr(msgAttr2)):
            cmn.confirmDialog('Given bkr group node is not available.')

        # find exp and customRotate
        oldExp          = self._breakRenameInputs(bkrGrp.attr(msgAttr1))
        oldCustomRotate = self._breakRenameInputs(bkrGrp.attr(msgAttr2))

        # rig all together
        # cachedGrp > exp.bkr_cachedRollingMsg
        # customRollingQuat > exp.bkr_customRollingMsg
        exp = pm.createNode('expression', name=name+'_bkr_exp')

        for attr in EXPMSGATTRS.values():
            exp.addAttr(attr, at='message')

        exp.message >> bkrGrp.attr(msgAttr1)
        customQuat.message >> exp.attr(EXPMSGATTRS['expCustom'])
        ctr.message >> exp.attr(EXPMSGATTRS['expCtr'])
        customCtr.message >> exp.attr(EXPMSGATTRS['expCustomCtr'])

        # connect caching groups to exp to find them
        for grp in cachingGrps:
            grp.addAttr(EXPMSGATTRS['expCache'], at='message')
            exp.attr(EXPMSGATTRS['expCache']) >> grp.attr(EXPMSGATTRS['expCache'])

        # subCtrQuat * exp > quat to euler > rollingGrp
        subCtrQuat = pm.createNode('eulerToQuat', name=name+'_subCtr_quat')
        subCtr.r >> subCtrQuat.inputRotate

        finalQuat = pm.createNode('quatProd', name=name+'_final_quat')
        subCtrQuat.outputQuat >> finalQuat.input1Quat

        rollingEuler = pm.createNode('quatToEuler', name=name+'_rolling_euler')
        finalQuat.outputQuat >> rollingEuler.inputQuat
        rollingEuler.outputRotate >> rollingGrp.rotate
        rollingEuler.message >> bkrGrp.attr(msgAttr2)

        # rollingGrp.rotate > rotateGrp.rotate
        rollingGrp.r >> rotateGrp.r

        # edit exp
        import bakeRolling_expression as bkrScr
        exp.expression.set( bkrScr.exp.format(ctr=ctr.name(),
                                              customCtr=customCtr.name(),
                                              output=finalQuat.name()
                                              ))

        pm.delete(oldExp)
        pm.delete(oldCustomRotate)

    def prebakeRemove(self):
        obj, dxRig, name = self.getTaggedObject(asList=True)
        bkrGrp    = self.getBkrGroup(obj, name, err=True)

        cmn.confirmDialog('Do you want to remove prebake (%s)?'% bkrGrp.name(), 1)

        dels = []
        for attr in ['cached', 'rolling', 'exp', 'customRotate']:
            dels.extend(bkrGrp.attr(MSGATTRS[attr]).inputs())

        pm.delete(dels)


    def finalBake(self):
        obj, dxRig, name = self.getTaggedObject(asList=True)
        bkrGrp    = self.getBkrGroup(obj, name, err=True)
        rotate = bkrGrp.attr(MSGATTRS['rotate']).inputs()[0]
        rolling = bkrGrp.attr(MSGATTRS['rolling']).inputs()[0]
        ctr = bkrGrp.attr(MSGATTRS['ctr']).inputs()[0]

        pm.bakeResults( rotate,
                        at=['rotateX', 'rotateY', 'rotateZ'],
                        simulation=True,
                        t="%d:%d"%(ctr.minFrame.get(), ctr.maxFrame.get()),
                        sampleBy=ctr.stepFrame.get(),
                        oversamplingRate=1,
                        disableImplicitControl=True,
                        preserveOutsideKeys=True,
                        sparseAnimCurveBake=False,
                        removeBakedAttributeFromLayer=False,
                        removeBakedAnimFromLayer=False,
                        bakeOnOverrideLayer=False,
                        minimizeRotation=False
                        )

        rolling.r // rotate.r

        inputs = []
        for attr in ['rx', 'ry', 'rz']:
            inputs.extend(rotate.attr(attr).inputs())

        pm.filterCurve(inputs)


    def addTag(self):
        obj = rig.getSelTransformNode()

        if obj.hasAttr(BKRTAG) or self._findTag(obj, False):
            cmn.confirmDialog('The tag already exists.')

        # Ask to add the attribute. If no, it stops
        cmn.confirmDialog('Do you want to add tag to \n%s object?'% obj, 1)

        obj.addAttr(BKRTAG, at='float')
        obj.attr(BKRTAG).set(self._getRadiusFromBBox(obj))

        pm.warning('The tag added.')

        # update selection list and choose the added object
        self.updateList()

        dxRig = self._getDxRig(obj)
        idx   = self.ui.bkr_selection_combox.findText(dxRig.name())
        self.ui.bkr_selection_combox.setCurrentIndex(idx)


    def selectTag(self):
        obj = self.getTaggedObject()
        pm.select(obj)


    def removeTag(self):
        obj = self.getTaggedObject()

        cmn.confirmDialog('Remove tag?', 1)

        scaleMd = obj.attr(BKRTAG).inputs()
        if scaleMd and isinstance(scaleMd[0], nt.MultiplyDivide):
            pm.delete(scaleMd[0])

        obj.deleteAttr(BKRTAG)

        if obj.hasAttr(BKRORGTAG):
            obj.deleteAttr(BKRORGTAG)

        pm.warning('The tag removed.')

        self.updateList()

    def setRadius(self):
        obj = self.getTaggedObject()
        msgAttr = 'bkrTrsMesh'

        # get radius from selected transform's boundingBox

        radius = obj.attr(BKRTAG).get()

        outputs = obj.message.outputs(p=True)
        trsMesh = None

        for output in outputs:
            if output.attrName(longName=True) == msgAttr:
                trsMesh = output.node()
                break

        if trsMesh:
            attr = BKRORGTAG if obj.hasAttr(BKRORGTAG) else BKRTAG
            obj.attr(attr).set(self._getRadiusFromBBox(trsMesh))

            pm.delete(trsMesh)
        else:
            trsMesh = pm.polySphere(r=radius, n='BKR_setRadius_trsMesh')[0]
            trsMesh.setParent(obj)
            rig.resetTrasform(trsMesh, s=False)

            trsMesh.addAttr(msgAttr, at='message')
            obj.message >> trsMesh.attr(msgAttr)


    def connectScale(self):
        scaleObj  = rig.getSelTransformNode()
        scaleAttr = scaleObj.attr(rig.getSelAttribute())
        tagObj    = self.getTaggedObject()

        if tagObj.attr(BKRTAG).inputs():
            cmn.confirmDialog('Alerady scale attribute connected.')

        if not tagObj.hasAttr(BKRORGTAG):
            tagObj.addAttr(BKRORGTAG, at='float')

        tagObj.attr(BKRORGTAG).set(tagObj.attr(BKRTAG).get())

        md = pm.createNode('multiplyDivide', name='BKR_scale_md')
        scaleAttr >> md.input1X
        tagObj.attr(BKRORGTAG) >> md.input2X

        md.outputX >> tagObj.attr(BKRTAG)


    # --------------------------------------------------------------------------

    def _findTag(self, obj, err=True):
        if obj.hasAttr(BKRTAG):
            return obj

        dxRig = self._getDxRig(obj)

        for node in dxRig.getChildren(ad=True):
            if node.hasAttr(BKRTAG):
                return node
        else:
            if err:
                cmn.confirmDialog('The tag does not exist.')
            else:
                return None

    def _getDxRig(self, obj, orRoot=True):
        for node in obj.getAllParents():
            if isinstance(node, nt.DxRig):
                return node
        else:
            return obj.root() if orRoot else None

    def _getRadiusFromBBox(self, obj):
        bbox = obj.getBoundingBox()
        radius = 0
        for i in range(3):
            _r = dt.abs(bbox[0][i] - bbox[1][i]) * 0.5
            radius = _r if radius < _r else radius

        return radius

    def _breakRenameInputs(self, attr, rename=True):
        inputs = attr.inputs()
        if inputs:
            for p in inputs[0].inputs(p=True, c=True):
                p[1] // p[0]
            if rename:
                inputs[0].rename('_tmp_')

        return inputs


    def getBkrGroup(self, rotateGrp, name, err=False):
        outputs = rotateGrp.message.outputs(p=True)
        msgAttr = MSGATTRS['rotate']
        grp     = None

        for output in outputs:
            if output.attrName() == msgAttr:
                grp = output.node()
                break
        else:
            if err:
                cmn.confirmDialog('No BKR GROUP')

            name += '_bkr_GRP'
            grp = pm.group(em=True, name=name)

            for attr in MSGATTRS.values():
                grp.addAttr(attr, at='message')

            rotateGrp.message >> grp.attr(msgAttr)

        return grp


    def getGroundGroup(self, bkrGrp):
        msgAttr = MSGATTRS['ground']
        grpName  = 'ground_GRP'

        if not bkrGrp.hasAttr(msgAttr):
            cmn.confirmDialog('Given bkr group node is not available.')

        inputs = bkrGrp.attr(msgAttr).inputs()
        if inputs:
            return inputs[0]

        grp = pm.group(em=True, name=grpName, parent=bkrGrp)
        grp.message >> bkrGrp.attr(msgAttr)

        return grp


    def getRollingGroup(self, bkrGrp):
        msgAttr = MSGATTRS['rolling']
        grpName = 'rolling_GRP'

        if not bkrGrp.hasAttr(msgAttr):
            cmn.confirmDialog('Given bkr group node is not available.')

        inputs = bkrGrp.attr(msgAttr).inputs()
        if inputs:
            pm.delete(inputs)

        grp = pm.group(em=True, name=grpName, parent=bkrGrp)
        grp.message >> bkrGrp.attr(msgAttr)

        return grp


    def getBkrCtr(self, bkrGrp, rotateGrp, name):
        msgAttr1 = MSGATTRS['ctr']
        msgAttr2 = MSGATTRS['subCtr']
        msgAttr3 = MSGATTRS['customCtr']
        msgAttr4 = MSGATTRS['customQuat']

        ctrName = name + '_bkrCtr'
        subCtrName = name + '_bkrSubCtr'
        customCtrName = name + '_customRollingCtr'

        if not (bkrGrp.hasAttr(msgAttr1) and \
                bkrGrp.hasAttr(msgAttr2) and \
                bkrGrp.hasAttr(msgAttr3) and \
                bkrGrp.hasAttr(msgAttr4)):
            cmn.confirmDialog('Given bkr group node is not available.')

        inputs1 = bkrGrp.attr(msgAttr1).inputs()
        inputs2 = bkrGrp.attr(msgAttr2).inputs()
        inputs3 = bkrGrp.attr(msgAttr3).inputs()
        inputs4 = bkrGrp.attr(msgAttr4).inputs()

        if inputs1 and inputs2 and inputs3 and inputs4:
            return inputs1[0], inputs2[0], inputs3[0], inputs4[0]

        ctrGrp = 'ctr_GRP'
        try:
            ctrGrp = pm.PyNode(bkrGrp.name()+'|'+ctrGrp)
        except:
            ctrGrp = pm.group(em=True, name=ctrGrp, parent=bkrGrp)

        # bkr ctr
        ctrFg = pm.group(em=True, name=ctrName+'Fg', parent=ctrGrp)
        ctr = rig.createCtr('halfSphere', ctrName, ctrFg, 'red')
        rig.lockHideAttributes(ctr)
        ctr.addAttr('cachedWeight', at='float', min=0, dv=1)
        ctr.cachedWeight.setKeyable(True)
        ctr.addAttr('groundChange', at='float', min=1, max=2, dv=1)
        ctr.groundChange.setKeyable(True)
        ctr.addAttr('subCtrVisibility', at='bool', dv=0)
        ctr.subCtrVisibility.setKeyable(True)
        ctr.addAttr('customRollingCtrVisibility', at='bool', dv=0)
        ctr.customRollingCtrVisibility.setKeyable(True)
        ctr.addAttr('minFrame', at='float')
        ctr.minFrame.setKeyable(True)
        ctr.addAttr('maxFrame', at='float')
        ctr.maxFrame.setKeyable(True)
        ctr.addAttr('stepFrame', at='float')

        # sub ctr
        subCtrFg = pm.group(em=True, name=subCtrName+'Fg', parent=ctr)
        subCtr   = rig.createCtr('crossRoundArrow', subCtrName, subCtrFg, 'yellow')
        rig.lockHideAttributes(subCtr, r=False)

        # custom rolling ctr
        customCtrFg = pm.group(em=True, name=customCtrName+'Fg', parent=ctr)
        customCtr   = rig.createCtr('rolling', customCtrName, customCtrFg, 'blue')
        rig.lockHideAttributes(customCtr, r=False)
        customCtr.addAttr('rolling', at='float')
        customCtr.rolling.setKeyable(True)
        customCtr.addAttr('cachedToCustom', at='float', min=0, max=1, dv=0)
        customCtr.cachedToCustom.setKeyable(True)

        # connect attributes
        pm.pointConstraint(rotateGrp, ctrFg, mo=False)
        ctr.message >> bkrGrp.attr(msgAttr1)
        subCtr.message >> bkrGrp.attr(msgAttr2)
        customCtr.message >> bkrGrp.attr(msgAttr3)
        ctr.subCtrVisibility >> subCtrFg.visibility
        ctr.customRollingCtrVisibility >> customCtrFg.visibility

        # fit the size
        ctrFg.s.set([rotateGrp.attr(BKRTAG).get()*1.2]*3)

        # custom rolling ctr rig
        # customCtr.r > eulerToQuat > quatInvert > quatProd
        # customCtr.rx + rolling > eulerToQuat   >
        # customCtr.ry, rz       >
        customCtrOrgQuat = pm.createNode('eulerToQuat', name=name+'_customRollingOrg_quat')
        customCtrIvtQuat = pm.createNode('quatInvert', name=name+'_customRollingIvt_quat')
        customCtr.r >> customCtrOrgQuat.inputRotate
        customCtrOrgQuat.outputQuat >> customCtrIvtQuat.inputQuat

        customRollingSumPma  = pm.createNode('plusMinusAverage', name=name+'_customRollingSum_pma')
        customRollingSumQuat = pm.createNode('eulerToQuat', name=name+'_customRollingSum_quat')
        customCtr.rx >> customRollingSumPma.input1D[0]
        customCtr.rolling >> customRollingSumPma.input1D[1]
        customRollingSumPma.output1D >> customRollingSumQuat.inputRotateX
        customCtr.ry >> customRollingSumQuat.inputRotateY
        customCtr.rz >> customRollingSumQuat.inputRotateZ
        customRollingSumPma.operation.set(2)

        customRollingQuat = pm.createNode('quatProd', name=name+'_customRolling_quat')
        customCtrIvtQuat.outputQuat >> customRollingQuat.input1Quat
        customRollingSumQuat.outputQuat >> customRollingQuat.input2Quat
        customRollingQuat.message >> bkrGrp.attr(msgAttr4)

        return ctr, subCtr, customCtr, customRollingQuat


    def getCachedGroup(self, bkrGrp):
        msgAttr = MSGATTRS['cached']

        if not bkrGrp.hasAttr(msgAttr):
            cmn.confirmDialog('Given bkr group node is not available.')

        inputs = bkrGrp.attr(msgAttr).inputs()
        if inputs:
            pm.delete(inputs)

        grp = pm.group(em=True, name='cached_GRP', parent=bkrGrp)
        grp.message >> bkrGrp.attr(msgAttr)

        return grp


    def getTaggedObject(self, asList=False):
        sel = self.ui.bkr_selection_combox.currentText()
        if not sel or sel[0] == '-':
            cmn.confirmDialog('Select an item in the list')

        res = self.taggedLists[sel]
        return res if asList else res[0]















  #
