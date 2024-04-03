from __future__ import print_function
import DXUSD.Vars as var

print(var.Ver(3))
print(var.VerAsInt('v012'))

# test tags
print(var.T.CGSUP)
print(var.T.USD.ATTR)
print(var.T.ATTR)
print(var.T.TASKS)
print(var.T.LODS)
print(var.T.HIGH)
#
print(var.T.ATTR_MATERIALSET)
print(var.T.tag.MATERIALSET)

print(var.IsVer('v003w'))
print(var.Ver(1))
