'''
 * dexter Render Setup Common Files
 * author : Dexter Sup Sanghun.kim, daeseok.chae in Dexter RND
 * date : 2017.11.03
'''
# import maya.cmds as cmds
# import maya.mel as mel

# import os
# import sys
import json

import maya.app.renderSetup.model.selector as selector
import maya.app.renderSetup.model.override as override
import maya.app.renderSetup.model.renderSetup as renderSetup

def rs_export(filename):
    # renderSetup.instance()
    renderSetup.instance().getDefaultRenderLayer().makeVisible()
    data = renderSetup.instance().encode()
    userData = {}
    userData["masterLayer"] = renderSetup.instance().getDefaultRenderLayer().isRenderable()
    for layer in renderSetup.instance().getRenderLayers():
        userData[layer.name()] = layer.isRenderable()
    data['sceneSettings']['renderManRIS']['userData'] = userData
    f = open(filename, 'w')
    json.dump(data, f, indent=2, sort_keys=True)
    f.close()

def rs_import(filename):
    f = open(filename, 'r')
    rs = renderSetup.instance()
    data = json.load(f)
    rs.decode(data, renderSetup.DECODE_AND_OVERWRITE, None)
    rs.acceptImport()
    f.close()

    userData = data['sceneSettings']['renderManRIS']['userData']
    renderSetup.instance().getDefaultRenderLayer().setRenderable(userData["masterLayer"])
    for layer in renderSetup.instance().getRenderLayers():
        layer.setRenderable(userData[layer.name()])


def rsInit():
    rs = renderSetup.instance()
    rs.clearAll()

def create_el():
    rs = renderSetup.instance()

    # create layer
    el = rs.createRenderLayer('EL')

    # object collection
    col = el.createCollection('objects')

    # simple selector
    sel = col.getSelector()
    sel.setFilterType(selector.Filters.kTransforms) # Collection Filters
    sel.setPattern('normalSpider*') # include expression

    # assigned objects
    objects = list(sel.getAbsoluteNames())

    # Absolute Override
    ovr = col.createOverride('Visibility', override.AbsOverride.kTypeId)
    plug = '%s.visibility' % objects[0]
    ovr.setAttributeName(plug)
    ovr.finalize(plug)
    ovr.setAttrValue(False)

    # stupidAOV
    col = el.createCollection('aov_object')
    sel = col.getSelector()
    sel.setFilterType(selector.Filters.kTransforms)
    sel.setStaticSelection('|stupidAOV1')

    return col
