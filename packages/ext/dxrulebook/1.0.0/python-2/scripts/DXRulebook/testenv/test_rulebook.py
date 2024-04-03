from __future__ import print_function
import DXRulebook.Rulebook as rb
import pprint, os

yaml = os.path.dirname(os.path.realpath(__file__))
yaml = os.path.split(yaml)[0]
yaml = os.path.join(yaml, 'resources/DXRulebook.yaml')

pprint.pprint('>>> yaml : %s'%yaml)

root = rb.Coder()
root.load_rulebook(yaml)

print('>>>>>>>>>>>>>>', root.N.USD.tag['MATERIALSET'])


# root.load_rulebook('/Users/wonchulkang/works/dexter/pylibs/DXUSD/ruleBook.yaml')
# print var.F.RULEBOOK
# a = '/show/pipe/_3d/asset/baer'
# pprint.pprint(root.decode(a, 'USD_DIR'))


# root.load_rulebook('/works/test/rulebookTest.yaml')
# a = 'abc/show'
# pprint.pprint(root.decode(a, 'TEST'))

# root.N.USD.flag['abname'] = 'haha'
# root.N.USD.flag['txlyrname'] = 'hoho'
# pprint.pprint(root.N.USD.product['TXATTR_CLASS'])

# b = '_sesdf_txAttr'
# pprint.pprint(root.N.USD.decode(b, 'TXATTR_CLASS'))

# path = '/mach/show/asdf/_3d/asset/houseB/model/v002/subdir'
path = '/show/asdf/_3d/shot/S03/S03_0030/asset/houseB/branch/burned/model/v002/subdir'
# path = '/show/asdf/_3d/shot/S03/S03_0030/asset/houseB/branch/burned/model/v002'
# path = '/show/asdf/_3d/shot/S03/S03_0030/asset/houseB/branch/burned/model'
# path = '/show/asdf/_3d/shot/S03/S03_0030/asset/houseB/branch/burned'
# path = '/show/asdf/_3d/shot/S03/S03_0030/asset/houseB/branch'
# path = '/show/asdf/_3d/shot/S03/S03_0030/asset/houseB'
# path = '/show/asdf/_3d/shot/S03/S03_0030/asset'
# path = '/show/asdf/_3d/shot/S03/S03_0030'
# path = '/show/asdf/_3d/shot/S03'
# path = '/show/asdf/_3d/shot'
# path = '/show/asdf/_3d'
# path = '/show/asdf'

# path = '/show/asdf/_3d/asset/houseB/branch/burned/model/v002/subdir'
# path = '/show/asdf/_3d/asset/houseB/branch/burned/texture/proxy/v002'
# path = '/show/asdf/_3d/shot/S02/S02_1234/ani/v008'

path = '/sdfa/asda/sd23/asset/houseB/branch/burned/model/v002/subdir'
pprint.pprint(root.D.decode(path))

# root.D.flag['show'] = 'pipe'
# root.D.flag['pub']  = '_3d'
root.D.flag['customdir'] = '/assetlib/3D'
# root.D.flag['seq']  = 's02'
# root.D.flag['shot'] = '0020'
root.D.flag['asset'] = 'houseA'
root.D.flag['branch'] = 'broken'
root.D.flag['task'] = 'model'
root.D.flag['nslyr'] = 'fjfjfjfj'
root.D.flag['ver'] = 'v002'
pprint.pprint(root.D.combine('ROOTS'))
pprint.pprint(root.D.product['TASKS'])
# pprint.pprint(root.N.combiner)

root.N.USD.flag['txlyrname'] = 'hahahaha'
root.N.USD.flag['abname'] = 'llll'
pprint.pprint(root.N.USD.product['TXATTR_CLASS'])

root.F.USD.flag['task'] = 'model'
root.F.USD.flag['abname'] = 'houseA'
# pprint.pprint(root.F.USD.model.product['PAYLOAD'])
# pprint.pprint(root.F.USD.product['PAYLOAD'])
pprint.pprint(root.child['F'].child['USD'].product['PAYLOAD'])

root.F.USD.flag['task'] = 'rig'
f = 'assetName_rig.payload.lgt.usd'
pprint.pprint(root.F.USD.rig.decode(f, 'LIGHT'))
pprint.pprint(root.F.USD.decode(f))
#
#

root.D.resetFlags()
root.D.flag['show'] = 'pipe'
root.D.flag['pub']  = '_3d'
root.D.flag['shot'] = '0030'
root.D.flag['seq']  = 'S06'
root.D.flag['abname'] = 'houseA'
root.D.flag['task'] = 'model'
root.D.flag['ver']  = 'v003'
pprint.pprint(root.D.product['TASKV'])
