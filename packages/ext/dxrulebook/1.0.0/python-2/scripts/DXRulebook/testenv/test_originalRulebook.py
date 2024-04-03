# from __future__ import print_function
import DXRulebook.Rulebook as rb
import pprint, os

# yaml = os.path.dirname(os.path.realpath(__file__))
# yaml = os.path.split(yaml)[0]
# yaml = os.path.join(yaml, 'testenv/test.yaml')
#
# pprint.pprint('>>> yaml : %s'%yaml)
#
# rulebook = rb.Coder()
# rulebook.load_rulebook(yaml)


# # set flags and encode
# root.flag['name'] = 'Charles'
# # root.flag['sex'] = 'woman'
# root.flag['age'] = '13'
# res = root.encode('PERSON')
# res = root.product['PERSON']
# print(res)


# decode and get flags
# val = 'Charles_mans_39'
# res = root.decode(val, 'PERSON')
# print(res)

# root.flag['date'] = '20200328'
# root.flag['location'] = '0028'
# print(root.Chair.Office.product['niceOfficeChair'])
#
#
# val = '20200702-2803N0010'
# res = root.Table.Caffe.decode(val, 'terraceTable')
# print(res)


# flag = root.flag['date']
# print(flag.name, flag.default, flag.pattern, flag.value)

# print(root.tag['DEPARTMENT'])
# print(root.AssetTeam.tag['DEPARTMENT'])
# print(root.AssetTeam.TEAM1.tag['DEPARTMENT'])
#
# print(root.flag['sex'].pattern)
#
# root.flag['name'] = 'Charles'
# root.flag['sex']  = 'man'
# print(root.AssetTeam.product['ARTIST'])

# rulebook.flag['asset'] = 'houseA'
# rulebook.flag['ver']   = 'v003'
# print '>>>', rulebook.product['ASSETUSD']

# val = '/show/pipe/_3d/asset/bear/rig/bear_rig_v003'
# print '>>>', rulebook.decode(val)
#
# rulebook.flag['root']  = '/show'
# rulebook.flag['show']  = 'pipe'
# rulebook.flag['pub']   = '_3d'
# rulebook.flag['seq']   = 'S03'
# rulebook.flag['shot']  = '0220'
# rulebook.flag['task']  = 'ani'
# rulebook.flag['nslyr'] = 'bear'
# rulebook.flag['nsver'] = 'v002'
# print '>>>', rulebook.product['SHOT']
# print '>>>', rulebook.product['TASKNV']
# print '>>>', rulebook.combine()

# val = '20200821-2107J0020'
# print '>>>', rulebook.Chair.Office.decode(val)

# rulebook.flag['kind']  = 'CH'
# rulebook.flag['usage'] = 'ST'
# rulebook.flag['date']  = '20200821'
# rulebook.flag['location'] = '0020'
# print '>>>', rulebook.product['backgoodChair']

# rulebook.flag['kind']  = 'CH'
# rulebook.flag['usage'] = 'ST'
# print '>>>', rulebook.decode('20200821-CHSTJ0020')





# yaml = os.path.dirname(os.path.realpath(__file__))
# yaml = os.path.split(yaml)[0]
# yaml = os.path.join(yaml, 'resources/DXRulebook.yaml')
#
# pprint.pprint('>>> yaml : %s'%yaml)
#
# rulebook = rb.Coder()
# rulebook.load_rulebook(yaml)
#
#
# rulebook.flag['show']  = 'pipe'
# rulebook.flag['pub']  = rulebook.tag['PUB3'].value
# rulebook.flag['asset'] = 'bear'
# print rulebook.D.product['ASSET']
#
# rulebook.resetFlags()
#
# rulebook.flag['abname'] = 'bear'
# rulebook.flag['task']   = 'model'
# print rulebook.F.USD.model.product['PAYLOAD']


import DXRulebook.Interface as rb

# print rb._RBROOT.name
# print rb._CATEGORY
# print rb._DCCS
#
# for flag in rb._FLAGS:
#     print '# >>>', flag
#
# for tag in rb._TAGS:
#     print '# >>>', tag

# import DXRulebook.Interface as rb
#
# tag = rb.Tags('USD')
#
# print '>>>', tag.tag['MODEL']
# print '>>>', tag.PAYLOAD
# print '>>>', tag.MAP.NORM




# import DXRulebook.Interface as rb
#
# Coder = rb.Coder()
#
# res   = Coder.D.Decode('/show/pipe/_3d/shot/S09/S09_0120/asset/houseA/branch/burned/model/v012')
# print '>>>', res
# # >>> {'task': 'model', 'shot': '0120', 'seq': 'S09', 'show': 'pipe',
# #      'pub': '_3d', 'asset': 'houseA', 'branch': 'burned', 'ver': 'v012', 'root': '/show'}
#
# print '>>>', Coder.D.Encode(show='pipe', pub='_3d', seq='S09', shot='0120', task='ani', ver='v003')
# # >>> /show/pipe/_3d/shot/S09/S09_0120/ani
#
# print '>>>', Coder.D.BRANCH.Encode(**res)
# # >>> /show/pipe/_3d/shot/S09/S09_0120/asset/houseA/branch/burned/model/v012
#
# print '>>>', Coder.N.USD.VAR_MODELVER.Encode(ver='v003')



#
# import DXRulebook.Interface as rb
#
# T = rb.Tags()
# D = rb.Coder('D', pub=T.PUB3)
# N = rb.Coder('N', 'USD')
#
# print '>>>', D.Encode(show='pipe', seq='S09', shot='0120', task='ani', ver='v003')
# # >>> /show/pipe/_3d/shot/S09/S09_0120/ani/v003
#
# print '>>>', N.VAR_MODELVER.Encode(ver='v003')
# # {modelVer=v003}



# import DXRulebook.Interface as rb
# T = rb.Tags()
# D = rb.Coder('D', pub=T.PUB3)
#
# res = D.Decode('/show/pipe/_3d/asset/houseA/branch/burned/model')
# print '>>>', res
# # >>> {'task': 'model', 'asset': 'houseA', 'branch': 'burned', 'show': 'pipe', 'root': '/show', 'pub': '_3d'}
#
# res.asset = 'houseB'
# print '>>>', res.asset, res.branch
# # >>> houseB burned
#
# print '>>>', res.product
# # >>> PUB/ASSET/BRANCH/TASK
#
# print '>>>', res.IsBranch()
# # >>> True


import DXRulebook.Interface as rb
#
# T = rb.Tags()
#
# arg = rb.Flags(show='pipe', pup=T.PUB3)
#
# arg.asset  = 'houseA'
# arg.branch = 'broken'
# arg.ver    = 'v003'
# arg.task   = 'model'
#
# print(arg)
# # {'task': 'model', 'ver': 'v003', 'show': 'pipe', 'pub': '_3d',
# #  'asset': 'houseA', 'branch': 'broken'}
# print('rig.PAYLOAD   >>>', arg.F.USD.rig.PAYLOAD)
# print('ASSETS        >>>', arg.F.USD.ASSETS)
# print('ABNAME_PAY    >>>', arg.F.USD.ABNAME_PAY)
# print('ASSET         >>>', arg.F.USD.ASSET)
# print('BRANCH        >>>', arg.F.USD.BRANCH)
# print('MASTER        >>>', arg.F.USD.MASTER)
# # rig.PAYLOAD   >>> broken_model.payload.usd
# # ASSETS        >>> asset.usd
# # ABNAME_PAY    >>> broken.payload.usd
# # ASSET         >>> houseA.usd
# # BRANCH        >>> broken.usd
# # MASTER        >>> broken_model.usd
#
# arg.Reset()
# arg.F.USD.model.SetDecode('bear_model.payload.usd')
# print arg

# N = rb.Coder('N', 'MAYA')
# res = N.TOP.Decode('OriginalAgent_Sdfsdf')
#
# print res.IsBranch()




import DXRulebook.Interface as rb
class Arguments(rb.Flags):
    def __init__(self, **kwargs):
        self.pub = '_3d'

        self.myAttr = 'hahaha'

        rb.Flags.__init__(self, 'USD', **kwargs)

arg = Arguments(show='pipe')
# arg.asset  = 'houseA'
# arg.branch = 'broken'
# arg.ver    = 'v003'
# arg.task   = 'model'

# res = arg.N.MAYA.SetDecode('S03_0030_main_sdf_cam')
# print '>>>', arg
#
# coder = rb.Coder()
# res = coder.N.MAYA.Decode('S03_0030_main_cam')
# print '>>>', res

# print arg.D.Decode('/show/pipe/_3d/shot/PKL/PKL_0290/cam/v001')
print arg.N.MAYA.Decode('town_burning_set_asb')


# print(arg)
# # {'task': 'model', 'ver': 'v003', 'show': 'pipe', 'pub': '_3d',
# #  'asset': 'houseA', 'branch': 'broken'}
# print('rig.PAYLOAD   >>>', arg.F.rig.PAYLOAD)
# print('ASSETS        >>>', arg.F.ASSETS)
# print('ABNAME_PAY    >>>', arg.F.ABNAME_PAY)
# print('ASSET         >>>', arg.F.ASSET)
# print('BRANCH        >>>', arg.F.BRANCH)
# print('MASTER        >>>', arg.F.MASTER)
# # rig.PAYLOAD   >>> broken_model.payload.usd
# # ASSETS        >>> asset.usd
# # ABNAME_PAY    >>> broken.payload.usd
# # ASSET         >>> houseA.usd
# # BRANCH        >>> broken.usd
# # MASTER        >>> broken_model.usd
