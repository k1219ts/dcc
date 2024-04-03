# -*- coding: utf-8 -*-
import sys
import os
import platform

BACKSTAGE_PATH='/backstage'
if os.getenv('BACKSTAGE_PATH'):
    BACKSTAGE_PATH=os.getenv('BACKSTAGE_PATH')

dxConfig = {'KOR':{'DB_IP'     : '10.0.0.12:27017',
                   #'DB_IP'     : '10.0.0.12:27017, 10.0.0.13:27017',
                   'TACTIC_IP' : '10.0.0.51',
                   'TRACTOR_IP': '10.0.0.30',
                   'TRACTOR_CACHE_IP': '10.0.0.25',
                   'TRACTOR_REDSHIFT_IP': '10.0.0.25',
                   'TRACTOR_PORT': 80,
                   'TRACTOR_API': {
                       'Linux': '{}/apps/Tractor/applications/linux/Tractor-2.2/lib/python2.7/site-packages'.format(BACKSTAGE_PATH),
                       'Windows': '{}/apps/Tractor/applications/win64/Tractor-2.2/lib/python2.7/Lib/site-packages'.format(BACKSTAGE_PATH),
                       'Darwin': '{}/apps/Tractor/applications/linux/Tractor-2.2/lib/python2.7/site-packages'.format(BACKSTAGE_PATH),
                       },
                   'ASSETLIB_PATH' : '/assetlib/reference'
                   },

            'CHN':{'DB_IP'     : '11.0.0.12:27017',
                   'TACTIC_IP' : '220.73.45.241',
                   'TRACTOR_IP': '11.0.2.20',
                   'TRACTOR_CACHE_IP': '11.0.2.20',
                   'TRACTOR_REDSHIFT_IP': '11.0.2.20',
                   'TRACTOR_PORT': 80,
                   'TRACTOR_API': {
                            'Linux': '{}/apps/tractor/linux/Tractor-2.2/lib/python2.7/site-packages'.format(BACKSTAGE_PATH),
                            'Windows': '{}/apps/tractor/win64/Tractor-2.2/lib/python2.7/Lib/site-packages'.format(BACKSTAGE_PATH)
                        },
                   'ASSETLIB_PATH' : '/netapp/assetlib'
                   }
            }

def getHouse():
    try:
        house = os.environ['SITE']
        return house

    except KeyError:
        # IF NO ENV THEN GET IP ADDRESS TO CHECK SITE
        import socket

        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))  # GOOGLE PUBLIC DNS SERVER
        ip_addr = s.getsockname()[0]

        if ip_addr.startswith('10.'):
            house = 'KOR'
        elif ip_addr.startswith('11.'):
            house = 'CHN'
        else:
            try:
                # HARD CODED PING TO KOR DATABASE LOCAL IP ADDRESS TO CHECK SITE
                if sys.platform == 'darwin':
                    cmd = 'ping -c 1 -t 1 10.0.0.12'
                else:
                    cmd = 'timeout 0.1 ping -c 1 10.0.0.12'
                res = os.system(cmd)

                if res == 0:
                    house = 'KOR'
                else:
                    house= 'CHN'
            except:
                house=None

        return house

def getConf(key):
    house = getHouse()
    if key in dxConfig[house]:
        val = dxConfig[house][key]
        if type(val).__name__ == 'dict':
            if platform.system() in val:
                return val[platform.system()]
            else:
                return val
        else:
            return val
    else:
        print("no key")

def getKeyList():
    house = getHouse()
    return dxConfig[house].keys()
