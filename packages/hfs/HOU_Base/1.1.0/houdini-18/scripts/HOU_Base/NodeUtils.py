#coding:utf-8
from __future__ import print_function

import re
import hou

def RetrieveByNodeType(node, type, res=[], output=False, firstMatch=True,
                       init=True, checked=[]):
    if init:
        checked = list()

    streamNodes = []
    if node.type().name() == 'object_merge' and not output:
        for i in range(node.parm('numobj').evalAsInt()):
            _node = node.parm('objpath%d'%(i+1)).evalAsNode()
            if _node and _node.type().name() != 'root':
                streamNodes.append(_node)
    ## elif node.type().name() == 'geo' and not output:
    elif node.isNetwork() and not node.isSubNetwork():
        if node.parm('skinsop'):
            GetStreamNode(node, 'skinsop', streamNodes)

        elif node.parm('sourcegroomobject'):
            GetStreamNode(node, 'sourcegroomobject', streamNodes)

        elif node.parm('sourcegroomobjects'):
            GetStreamNode(node, 'sourcegroomobjects', streamNodes)

        else:
            for child in node.children():
                if child.type().name() == 'output':
                    streamNodes.append(child)
    else:
        streamNodes = node.outputs() if output else node.inputs()

    thisres = []
    next = []
    for n in streamNodes:
        if n and n not in checked:
            checked.append(n)
            next.append(n)
            if n.type().name() == type:
                thisres.append(n.path())

    res.extend(thisres)
    if not thisres or not firstMatch:
        for n in next:
            RetrieveByNodeType(n, type, res, output, firstMatch, False, checked)


def FindParmInExpression(parm):
    if parm and parm.keyframes():
        found = re.search('"[a-zA-Z0-9./_]*"', parm.expression())
        if found:
            return hou.parm('%s/%s'%(parm.node().path(), found.group()[1:-1]))
    return parm


def GetStreamNode(node, parm, res):
    if node.parm(parm).evalAsString():
        _node = node.parm(parm).evalAsNode()
        res.append(_node)
    else:
        _node = node.inputs()[1]
        res.append(_node)

    if node.parm('animskin'):
        if node.parm('animskin').evalAsString():
            _node = node.parm('animskin').evalAsNode()
            res.append(_node)
        else:
            _node = node.inputs()[2]
            res.append(_node)


def GetGroomDependency(node, type, res=[], output=False, firstMatch=True,
                       init=True, checked=[]):
    if init:
        checked = list()

    streamNodes = []
    if node.type().name() == 'object_merge' and not output:
        for i in range(node.parm('numobj').evalAsInt()):
            _node = node.parm('objpath%d'%(i+1)).evalAsNode()
            if _node and _node.type().name() != 'root':
                streamNodes.append(_node)
        if node.parm('skinsop'):
            GetStreamNode(node, 'skinsop', streamNodes)

        elif node.parm('sourcegroomobject'):
            GetStreamNode(node, 'sourcegroomobject', streamNodes)

        elif node.parm('sourcegroomobjects'):
            GetStreamNode(node, 'sourcegroomobjects', streamNodes)

        else:
            for child in node.children():
                if child.type().name() == 'output':
                    streamNodes.append(child)
    else:
        streamNodes = node.outputs() if output else node.inputs()

    thisres = []
    next = []
    for n in streamNodes:
        if n and n not in checked:
            checked.append(n)
            next.append(n)
            if n.type().name() == type:
                thisres.append(n.path())

    res.extend(thisres)
    if not thisres or not firstMatch:
        for n in next:
            GetGroomDependency(n, type, res, output, firstMatch, False, checked)

