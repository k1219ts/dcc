#coding:utf-8
from __future__ import print_function

import DXUSD.moduleloader as mdl
import DXUSD.Vars as var
mdl.importModule(__name__, var)


NULL  = '__NULL__'
ORDER = '__ORDER__'
USDPATH = '__USDPATH__'

PADDING4 = '`padzero(4, $F)`'
PADDING5 = '`padzero(5, $F)`'

TYPEGEOM = 'geom'
TYPEINST = 'inst'
TYPEGROOM = 'groom'
TYPECROWD = 'crowd'
TYPEFEATHER = 'feather'

LYRTYPES = [
    TYPEGEOM,
    TYPEINST,
    TYPEGROOM,
    TYPECROWD,
    TYPEFEATHER
]

DEPENDPATH = 'depend_%s:path'
DEPENDVARS = 'depend_%s:vars'
