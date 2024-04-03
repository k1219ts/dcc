from __future__ import print_function
import os
from pprint import pprint

import DXRulebook.Interface as rb

T = rb.Tags()
print(T.MODEL) # model
print(T.USD.ATTR_CHANNELS) # userProperties:Texture:channels
print(T.MAP.DISF) # disF

T = rb.Tags('USD')
print(T.MODEL) # model
print(T.tag.MODEL) # model
print(T.ATTR_CHANNELS) # userProperties:Texture:channels
print(T.MAP.DISF) # disF

Coder = rb.Coder()
print(Coder.D.Decode('/show/pipe/_3d/shot/S09/S09_0120/asset/houseA/branch/burned/model/v012'))
# {'task': 'model', 'shot': '0120', 'seq': 'S09', 'show': 'pipe',
#  'pub': '_3d', 'asset': 'houseA', 'branch': 'burned', 'ver': 'v012', 'root': '/show'}
print(Coder.D.Encode(show='pipe', pub='_3d', seq='S09', shot='0120', task='ani', ver='v003'))
# /show/pipe/_3d/shot/S09/S09_0120/ani
print(Coder.D.SEQ.Encode(show='pipe', pub='_3d', seq='S09', shot='0120', task='ani'))
# /show/pipe/_3d/shot/S09
print(Coder.D.ASSETS.Encode(show='pipe', pub='_3d', seq='S09', shot='0120', task='ani'))
# /show/pipe/_3d/shot/S09/S09_0120/asset
res = Coder.F.USD.ani.FINAL.Decode('upRobot1.usd')
print(res)
# {'ext': 'usd', 'nslyr': 'upRobot1'}
print(Coder.F.USD.ani.PAYLOAD.Encode(**res))
# upRobot1.payload.usd

D = rb.Coder('D', pub='_3d')
F = rb.Coder('F', dcc='USD')
N = rb.Coder('N', dcc='USD')
print(D.Encode(show='pipe', seq='S09', shot='0120', task='ani', ver='v003'))
# /show/pipe/_3d/shot/S09/S09_0120/ani
res = F.ani.FINAL.Decode('upRobot1.usd')
print(res)
# {'ext': 'usd', 'nslyr': 'upRobot1'}
print(F['ani'].PAYLOAD.Encode(**res))
# upRobot1.payload.usd


class Arguments(rb.Flags):
    def __init__(self, **kwargs):
        self.pub = '_3d'

        self.myAttr = 'hahaha'

        rb.Flags.__init__(self, 'USD', **kwargs)

arg = Arguments(show='pipe')
arg.asset = 'houseA'
arg.branch = 'broken'
arg.ver   = 'v003'
arg.task = 'model'

print(arg)
# {'task': 'model', 'ver': 'v003', 'show': 'pipe', 'pub': '_3d',
#  'asset': 'houseA', 'branch': 'broken'}
print('rig.PAYLOAD   >>>', arg.F.rig.PAYLOAD)
print('ASSETS        >>>', arg.F.ASSETS)
print('ABNAME_PAY    >>>', arg.F.ABNAME_PAY)
print('ASSET         >>>', arg.F.ASSET)
print('BRANCH        >>>', arg.F.BRANCH)
print('FINAL         >>>', arg.F.FINAL)
# rig.PAYLOAD   >>> broken_model.payload.usd
# ASSETS        >>> asset.usd
# ABNAME_PAY    >>> broken.payload.usd
# ASSET         >>> houseA.usd
# BRANCH        >>> broken.usd
# FINAL         >>> broken_model.usd
