#coding:utf-8
from __future__ import print_function

import DXUSD.moduleloader as mdl
import DXUSD.Vars as var
DEV = mdl.importModule(__name__, var)
if DEV:
    rb.Reload()

NULL  = '__NULL__'
ORDER = '__ORDER__'
USDPATH = '__USDPATH__'
NSLYR   = '__NSLYR__'

PADDING4 = '`padzero(4, $F)`'
PADDING5 = '`padzero(5, $F)`'

LYRGEOM = 'geom'
LYRINST = 'inst'
LYRGROOM = 'groom'
LYRCROWD = 'crowd'
LYRFEATHER = 'feather'

LYRTYPES = [
    LYRGEOM,
    LYRINST,
    LYRGROOM,
    LYRCROWD,
    LYRFEATHER
]

PRCNONE = 'none'
PRCCLIP = 'clip'
PRCSIM  = 'sim'
PRCFX   = 'fx'

PRCTYPES = [
    PRCNONE,
    PRCCLIP,
    PRCSIM,
    PRCFX
]

DEPEND = 'depend'
PATH = 'path'
VARS = 'vars'
DEPENDPATH = DEPEND + ':%s:%s:' + PATH
DEPENDVARS = DEPEND + ':%s:%s:' + VARS

UNKNOWN = '*** UNKOWN ***'
