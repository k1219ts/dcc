import sys
import os
import glob
import re
import string
import datetime
import time
import json
import shutil
import getpass
import configobj
import pprint


configpath = os.getenv('BACKSTAGE_PATH')
if os.getenv('LOCALCONFIG'):
    configpath = os.getenv('LOCALCONFIG')
config_fn = '{PATH}/config/rfm.config'.format(PATH=configpath)
getConfig = configobj.ConfigObj(config_fn)


def GetEngineList():
    result = list()
    data = getConfig['TractorEngine']
    for i in range(len(data)/2):
        result.append(data[i*2])

    data = getConfig['CloudEngine']
    for i in range(len(data) / 5):
        result.append(data[i * 5])

    return result

def GetEnginePort(engine):
    if ":" in engine:
        return int(engine.split(":")[-1])

    data = list()
    for k in getConfig.keys():
        if k.find('Engine') > -1:
            data += getConfig[k]
    port = 80
    if engine in data:
        id = data.index(engine)
        port = int(data[id+1])
    return port

def GetCloudDenoiseIP(engine):
    data = list()
    for k in getConfig.keys():
        if k.find('CloudEngine') > -1:
            data += getConfig[k]
    ip = "10.0.1.2"
    if engine in data:
        id = data.index(engine)
        ip = data[id + 2]
    return ip

def GetCloudCopyIP(engine):
    data = list()
    for k in getConfig.keys():
        if k.find('CloudEngine') > -1:
            data += getConfig[k]
    ip = "10.0.1.2"
    if engine in data:
        id = data.index(engine)
        ip = data[id + 4]
    return ip

def GetProcessEngine(process):
    if getConfig.has_key(process):
        hostname = getConfig[process][0]
    else:
        hostname = getConfig['TractorEngine'][0]
    return hostname


def GetFormatList(renderer):
    map = {
        'renderManRIS': ['OpenEXR', 'Tiff32', 'Tiff16', 'Tiff8', 'Targa'],
        'mayaSoftware': ['tif', 'jpg', 'taga'],
        'mentalRay': ['tif', 'jpg', 'taga']
    }
    if map.has_key(renderer):
        return map[renderer]
    else:
        return None

try:
    sys.path.append(getConfig['TractorAPI'])
    import tractor.api.author as author
    import tractor.api.query as query
    import tractor.base.EngineClient as EngineClient
    from tractor.base.TrHttpRPC import TrHttpRPC
except Exception as e:
    print e

# scriptpath = os.path.dirname(os.path.abspath(__file__))
if os.getenv('LOCALCONFIG'):
    # DeNoiseCmd = '%s/denoiseRender.py' % scriptpath
    DeNoiseCmd = '%s/denoiseRender.py' % os.getenv('LOCALCONFIG')
else:
    DeNoiseCmd = '%s/apps/maya2/global/RfM_Submitter/denoiseRender.py' % os.getenv('BACKSTAGE_PATH')

PYTHONCMD  = '/netapp/backstage/pub/bin/python'

__all__ = [
    'sys', 'os', 'glob', 're', 'string', 'datetime', 'time',
    'json', 'shutil', 'getpass', 'configobj', 'pprint', 'getConfig',
    'GetEngineList', 'GetEnginePort', 'GetProcessEngine', 'GetFormatList',
    'author', 'query', 'EngineClient', 'TrHttpRPC',
    'DeNoiseCmd', 'PYTHONCMD'
]