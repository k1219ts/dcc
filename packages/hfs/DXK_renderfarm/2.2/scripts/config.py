"""
Main config file for the farm
Little messy as we didn't have proper backstage
"""

import sys
import os
import datetime
import json
import math
import shutil
import subprocess
import time
import site
import argparse
import configobj
import re
import fnmatch
import pprint

server_root = os.getenv('BACKSTAGE_PATH')
server_dev_root = os.getenv('BACKSTAGE_DEV_PATH')
if server_dev_root:
    server_root = server_dev_root
# temporary use hou.config
# config_fn = '{BACKSTAGE_PATH}/config/hou.config'.format(BACKSTAGE_PATH=server_root)
config_fn = '%s/hou.config'%os.getenv('HOU_CONFIG_PATH')
getConfig = configobj.ConfigObj(config_fn)

# Slot Menu for DXC_config
def slotList():
    data = getConfig['SLOT_MENU']
    keys = data.keys()
    keys.sort()
    result = list()
    for i in keys:
        result.append( i )
        result.append( data[i] )
    return result

try:
    sys.path.append(getConfig['TractorAPI'])
    import tractor.api.author as author
    import tractor.api.query as query
    import tractor.base.EngineClient as EngineClient
    from tractor.base.TrHttpRPC import TrHttpRPC
except Exception as e:
    print e

__all__ = ['sys', 'os', 'datetime', 'json', 'math', 'shutil',
           'subprocess', 'time', 'site', 'argparse', 'configobj',
           're', 'fnmatch', 'pprint',
           'author', 'query', 'EngineClient', 'TrHttpRPC'
           ]
