# -*- coding: utf-8 -*-

import json
import string
import maya.cmds as cmds
import sgCommon


def addNamespace(ns, node):
    if ns:
        name = string.join([ns, node], ":")
    else:
        name = node
    return name


def writeJson(data, filepath):
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)
        f.close()


def readJson(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data


def getAssetpath(node):
    """Get referenced asset's file path

    :param node: dxRig Node
    :return: A string of reference file path
    """
    filename = cmds.referenceQuery(node, f=True)
    return filename


def getNodesReferenced(object, type):
    """

    :param object: Reference node name
    :param type: Object type to find
    :return: Node list
    """
    nodes = list()
    refNodes = cmds.referenceQuery(object, nodes=True)
    for rNode in refNodes:
        try:
            rType = cmds.objectType(rNode)
            if rType == type:
                nodes.append(rNode)
        except:
            pass
    return nodes


def loadReference(file, namespace, type):
    """Load asset as reference

    :param file: rig filename
    :param namespace: asset name
    :return A string of referenced dxRig node name
    """
    if cmds.namespace(ex=namespace):
        namespace = namespace + "#"
    ref_fileName = cmds.file(file, r=True, type='mayaBinary', gl=True,
                             mergeNamespacesOnClash=False,
                             namespace=namespace, options='v=0;')
    referencedNode = getNodesReferenced(ref_fileName, type=type)[0]
    return referencedNode


def getAttrs(nodes):
    """Get attribute list from dxRig nodes

    :param nodes: list of dxRig
    :return: A dictionary of controlers attributes
    """
    attrs = dict()
    for node in nodes:
        ns_name, node_name = sgCommon.getNameSpace(node)
        attrs[node_name] = dict()
        cons = cmds.getAttr(node + '.controlers')
        for con in cons:
            con_name = addNamespace(ns_name, con)
            attrs[node_name][con] = cmds.listAttr(con_name, k=True)

            # find null group of controler ( dexter rig only )
            null_name = con.replace("_CON", "_NUL")
            null_node = addNamespace(ns_name, null_name)
            if cmds.objExists(null_node):
                attrs[node_name][null_name] = cmds.listAttr(null_node, k=True)
    return attrs


def write(nodes, filename=None, namespace=True, assetpath=True):
    """Save rig controlers key data to dictionary

    :param nodes: dxRig node
    :param filename: json filename
    :param namespace: If true, save dxRig's namespace
    :param assetpath: If true, save asset file path
    :return if jsonOut is True, json file name. else, A dictionary of key data
    """
    attrdata = getAttrs(nodes)
    for node in nodes:
        ns_name, node_name = sgCommon.getNameSpace(node)
        assetFileName = getAssetpath(node=node)
        condata = attrdata[node_name]
        for m_con in condata:
            con = addNamespace(ns_name, m_con)
            if condata[m_con] is None:
                continue
            attrdata[node_name][m_con] = sgCommon.attributesKeyDump(node=con, attrs=condata[m_con])
        if namespace:
            attrdata[node_name].update({'_namespace': ns_name})
        if assetpath:
            attrdata[node_name].update({'_assetpath': assetFileName})
    attrdata.iterkeys()
    writeJson(data=attrdata, filepath=filename)
    return filename


def read(jsonfile, loadFile=True, ns_fromfile=True, namespace=None, returnType='dxRig'):
    """Load and apply keyframe data from json file

    :param jsonfile: json keydata file
    :param loadFile: If True, reference asset file
    :param ns_fromfile: True if namespace 'from file'
    :param namespace: Namespace of reference node
    :param returnType: Maya nodeType of referenced nodes to return
    :return referencedNode:
    """
    referencedNode = None
    keydata = readJson(jsonfile)
    for dxRigNode in keydata:
        condata = keydata[dxRigNode]
        if ns_fromfile and condata.has_key('_namespace'):
            namespace = condata['_namespace']
        assetfile = condata['_assetpath']
        if loadFile:
            referencedNode = loadReference(file=assetfile, namespace=namespace, type=returnType)
            namespace = referencedNode.split(":")[0]
        for m_con in condata:
            con = addNamespace(namespace, m_con)
            if (m_con != '_namespace') and (m_con != '_assetpath'):
                try:
                    con_attrs = cmds.listAttr(con, k=True)
                    attrs = condata[m_con]
                    if attrs is None:
                        continue
                    newAttrs = {k: v for k, v in attrs.iteritems() if k in con_attrs}

                    sgCommon.attributesKeyLoad(node=con, keyData=newAttrs)
                except Exception, e:
                    print e
                    continue
    return referencedNode
