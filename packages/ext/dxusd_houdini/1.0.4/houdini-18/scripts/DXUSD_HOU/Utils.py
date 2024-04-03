#coding:utf-8
from __future__ import print_function

import DXUSD.moduleloader as mdl
import DXUSD.Utils as utl
mdl.importModule(__name__, utl)

import DXUSD_HOU.Vars as var




def DependInfoToDict(lyrdata):
    '''
    [Return] - dict
    {
        (kind) : {
            (task) : {
                (vset) : (variant),
                (var.USDPATH) : (path),
                (var.ORDER) : [(vset), ...]
            }
        }
    }
    '''
    res = dict()
    for key, data in lyrdata.items():
        if not key.startswith(var.DEPEND):
            continue

        elms = key.split(':')
        dkind = elms[1]
        dtask = elms[2]
        dtype = elms[3]

        if not res.has_key(dkind):
            res[dkind] = {}

        if not res[dkind].has_key(dtask):
            res[dkind][dtask] = {var.USDPATH:'', var.ORDER:[]}

        if dtype == var.PATH:
            res[dkind][dtask][var.USDPATH] = data
        else:
            for v in data.split(', '):
                vset, variant = v.split('=')
                res[dkind][dtask][vset] = variant
                res[dkind][dtask][var.ORDER].append(vset)
    return res


def GetTxBasePath(**kwargs):
    '''
    [Arguments]
    kwargs : rulebook flags (asset, branch)
    '''
    if not kwargs.has_key('asset'):
        msg.error('Need "asset" argument')

    flags = {'asset':kwargs['asset']}
    if kwargs.has_key('branch'):
        path = 'asset/{asset}/branch/{branch}/texture'
        flags['branch'] = kwargs['branch']
    else:
        path = 'asset/{asset}/texture'

    return path.format(**flags)





#
