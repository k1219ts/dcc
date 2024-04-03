import maya.api.OpenMaya as om
import pymel.core as pm
import pymel.core.nodetypes as nt


def getShellFaces(mIter):
    curIdx = mIter.index()
    searchFids = [curIdx]
    foundFids = [curIdx]

    while len(searchFids):
        mIter.setIndex(searchFids.pop(0))
        m_cntFids = mIter.getConnectedFaces()

        for i in list(m_cntFids):
            if not i in foundFids:
                foundFids.append(i)
                searchFids.append(i)

    return foundFids

def run():
    # get selection
    sels = pm.ls(sl=True)

    # attribute names
    primVarName = 'shell_id'
    primVarAttr = 'rmanafF%s' % primVarName
    primVarScopeAttr = '%s_AbcGeomScope' % primVarAttr

    for sel in sels:

        # get shape from sels
        if isinstance(sel, nt.Mesh):
            shape = sel
        else:
            shape = sel.getShape()

            if not (shape and isinstance(shape, nt.Mesh)):
                continue

        # check given object has multiple shells
        numShell = pm.polyEvaluate(sel, shell = True)
        if not numShell > 1:
            pm.warning('%s skipped. (It has one poly-shell)' % shape)
            continue

        # check attributes if exists, confirm it
        if shape.hasAttr(primVarAttr) and shape.hasAttr(primVarScopeAttr):
            confirm = pm.confirmDialog(title = 'Add shell_id',
                             message = '%s already has the attr. \n Do you want to skip?',
                             button = ['Skip', 'Keep going'],
                             defaultButton = 'Skip')
            if confirm == 'Skip':
                pm.warning('%s skipped. (It has shell_id attributes.)' % shape)
                continue

        # set api object
        mSel = om.MSelectionList()
        mSel.add(shape.name())
        mObj = mSel.getDependNode(0)
        mIter = om.MItMeshPolygon(mObj)

        fids = list(range(mIter.count()))
        fVaryIds = []

        # get shell's polygon count that set facevary ids
        id = 0
        while len(fids):
            mIter.setIndex(fids[0])

            for i in getShellFaces(mIter):
                mIter.setIndex(i)
                fVaryIds.extend([id] * mIter.polygonVertexCount())

                fids.remove(i)
            id += 1

        # add and set attributes
        if not shape.hasAttr(primVarAttr):
            shape.addAttr(primVarAttr, dt='doubleArray')

        shape.attr(primVarAttr).set(fVaryIds)

        if not shape.hasAttr(primVarScopeAttr):
            shape.addAttr(primVarScopeAttr, dt='string')
            shape.attr(primVarScopeAttr).set('fvr')
            shape.attr(primVarScopeAttr).setLocked(True)

    if sels:
        pm.confirmDialog(title = 'Add shell_id', message = 'Completed', button='OK')
    else:
        pm.error('Select polygons')
