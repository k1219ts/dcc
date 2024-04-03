from __future__ import print_function
import os
import xml.etree.ElementTree as ET

from Katana import RenderingAPI
from fnpxr import Sdf

import DXUSD_KAT.Utils as utl


_KAT_TYPE_MAP = {}
for n in dir(RenderingAPI.RendererInfo):
    if n.startswith('k'):
        _KAT_TYPE_MAP[eval('RenderingAPI.RendererInfo.%s' % n)] = n


def DebugParams(shaderType):
    slAttr = utl.GetShaderFnAttr(shaderType)
    types  = list()
    for n, attr in slAttr.getChildByName('params').childList():
        idx = attr.getChildByName('type').getValue()
        print('# {0} : {1}'.format(n, _KAT_TYPE_MAP[idx]))
        print('')
        types.append(_KAT_TYPE_MAP[idx])
    types = list(set(types))
    print("# TYPES :")
    for n in types:
        print('>', n)
    return types


def DebugOutputs(shaderType):
    slAttr = utl.GetShaderFnAttr(shaderType)
    for n, attr in slAttr.getChildByName('outputTags').childList():
        print('# {0} : {1}'.format(n, attr.getValue()))


def GetAllShaders():
    def getShaderType(filename):
        tree = ET.parse(filename)
        root = tree.getroot()
        types= root.findall('shaderType')
        if types:
            return types[0].getchildren()[0].attrib['value']
    result = list()
    shaderTypes = list()
    for dir in os.getenv('RMAN_RIXPLUGINPATH').split(':'):
        if dir and os.path.exists(dir):
            for f in os.listdir(dir):
                fn = os.path.join(dir, f)
                if os.path.isfile(fn) and f.split('.')[-1] == 'so':
                    argsfile = os.path.join(dir, 'Args', f.replace('.so', '.args'))
                    if os.path.exists(argsfile):
                        stype = getShaderType(argsfile)
                        if not stype in ['displaydriver', 'integrator']:
                            shaderTypes.append(stype)
                            result.append(f.split('.')[0])
    shaderTypes = list(set(shaderTypes))
    # print('>', shaderTypes)
    return result


def DebugAllParamTypes():
    shaders = GetAllShaders()
    types   = list()
    for s in shaders:
        print('>>>>', s)
        types += DebugParams(s)
    types = list(set(types))
    print('')
    print("# ALL TYPES :")
    for n in types:
        print('>', n)
