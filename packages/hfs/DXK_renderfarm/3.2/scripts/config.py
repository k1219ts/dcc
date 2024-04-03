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
import dxConfig



tr_api = dxConfig.getConf('TRACTOR_API')

slot_menu = {'1':'Large job (32 cores, 128GB) - for heavy sim'}

# Slot Menu for DXC_config
def slotList():
    keys = slot_menu.keys()
    keys.sort()
    result = list()
    for i in keys:
        result.append(i)
        result.append(slot_menu[i])
    # print TRACTOR_API
    return result


try:
    sys.path.append(tr_api)
    import tractor.api.author as author
    import tractor.api.query as query
    import tractor.base.EngineClient as EngineClient
    from tractor.base.TrHttpRPC import TrHttpRPC
except Exception as e:
    print e


__all__ = ['sys', 'os', 'datetime', 'json', 'math', 'shutil',
           'subprocess', 'time', 'site', 'argparse', 'configobj',
           're', 'fnmatch', 'pprint', 'dxConfig',
           'author', 'query', 'EngineClient', 'TrHttpRPC'
           ]