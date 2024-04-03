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


config_fn = '{PATH}/config/rfm.config'.format(PATH=os.getenv('BACKSTAGE_PATH'))
if not os.path.exists(config_fn):
    config_fn = '/netapp/backstage/pub/config/rfm.config'
getConfig = configobj.ConfigObj(config_fn)


def GetEngineList():
    result = list()
    data = getConfig['TractorEngine']
    for i in range(len(data)/2):
        result.append(data[i*2])
    return result


def GetEnginePort(engine):
    data = list()
    for k in getConfig.keys():
        if k.find('Engine') > -1:
            data += getConfig[k]
    port = 80
    if engine in data:
        id = data.index(engine)
        port = int(data[id+1])
    return port


try:
    sys.path.append(getConfig['TractorAPI'])
    import tractor.api.author as author
    import tractor.api.query as query
    import tractor.base.EngineClient as EngineClient
    from tractor.base.TrHttpRPC import TrHttpRPC
except Exception as e:
    print e


__all__ = [
    'sys', 'os', 'glob', 're', 'string', 'datetime', 'time',
    'json', 'shutil', 'getpass', 'configobj', 'pprint', 'getConfig',
    'GetEngineList', 'GetEnginePort',
    'author', 'query', 'EngineClient', 'TrHttpRPC',
]
