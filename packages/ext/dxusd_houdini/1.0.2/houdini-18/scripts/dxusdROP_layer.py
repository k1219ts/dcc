import hou
import DXUSD.Message as msg
import DXUSD_HOU.Vars as var
import DXUSD_HOU.Utils as utl

import HOU_Base.NodeUtils as ntl

LYRPARMS = ['nslyr', 'lyrname', 'dprim', 'sublyr']

def ShowLayerResParms(node, *args):
    for parmname in LYRPARMS:
        hide = parmname not in args
        node.parm(parmname).hide(hide)
        node.parm(parmname+'tgl').hide(hide)


def GetVariants(node, i, kind):
    res = {var.ORDER:[]}
    for j in range(node.parm('%svariantfolder%d'%(kind, i)).evalAsInt()):
        key = node.parm('%svariantset%d_%d'%(kind, i, j)).evalAsString()
        val = node.parm('%svariants%d_%d'%(kind, i, j)).evalAsString()
        res[key] = val
        res[var.ORDER].append(key)

    return res


def GetDependency(node):
    '''
    [Return] - dict
    {
        'shot':
            { variantSet:variant, ... __order__:[variantSet ...]},
        'asset:(task)':
            { variantSet:variant, ... __order__:[variantSet ...]}
    }
    '''
    res = {}
    checked = []

    for i in range(node.parm('dependencyfolder').evalAsInt()):
        soppath = node.parm('dependencysoppath%d'%i).evalAsNode()
        if soppath:
            if soppath in checked:
                continue
            else:
                checked.append(soppath)
        else:
            continue

        # check shot
        if node.parm('shotvariantfolder%d'%i).evalAsInt():
            path = utl.DirName(node.parm('usdpath%d'%i).evalAsString())
            try:
                args = var.D.Decode(path)
            except Exception as e:
                msg.errmsg(e)
                warnmsg = 'Failed decoding usdpath%d of %s node'
                msg.warning(warnmsg%(i, soppath))
                continue

            args.nslyr = node.parm('shotnamespace%d'%i).evalAsString()

            vlist = GetVariants(node, i, var.T.SHOT)
            # reverse order becuase the last task version is used for this layer
            order = list(vlist[var.ORDER])
            order.reverse()

            for vset in order:
                try:
                    vargs = var.N.VAR_TASKVER.Decode(vset)
                    if vargs.has_key(var.T.TASK):
                        vargs.nsver = vlist[vset]
                        args.update(vargs)
                except:
                    continue


            path = var.D.TASKNV.Encode(**args)
            file = var.F[args.task].MASTER.Encode(**args)
            vlist[var.USDPATH] = utl.SJoin(path, file)
            vlist[var.NSLYR] = args.nslyr
            res['%s:%s'%(var.T.SHOT, args.task)] = vlist

        # check asset and branch
        if node.parm('assetvariantfolder%d'%i).evalAsInt():
            path = utl.DirName(node.parm('assetusdpath%d'%i).evalAsString())
            try:
                args = var.D.Decode(path)
            except Exception as e:
                msg.errmsg(e)
                warnmsg = 'Failed decoding assetusdpath%d of %s node'
                msg.warning(warnmsg%(i, soppath))
                continue

            for kind in [var.T.ASSET, var.T.BRANCH]:
                vlist = GetVariants(node, i, kind)

                # check task
                if not vlist.has_key(var.T.TASK):
                    continue

                # check key
                key = '%s:%s'%(kind, vlist[var.T.TASK])
                if res.has_key(key):
                    msg.warning('Layer node has same kind:task dependency.')
                    continue

                # check branch
                if vlist[var.T.TASK] == var.T.BRANCH:
                    args.branch = vlist[var.T.BRANCH]
                    continue
                else:
                    args.task = vlist[var.T.TASK]

                # get task version
                taskver = var.N.VAR_TASKVER.Encode(**args)
                if not vlist.has_key(taskver):
                    continue

                if utl.IsVer(vlist[taskver]):
                    if args.task == var.T.CLIP:
                        args.nslyr = vlist[var.T.CLIP]
                        args.nsver = vlist[taskver]
                    else:
                        args.ver = vlist[taskver]
                else:
                    args.nslyr = vlist[taskver]

                # set usdpath
                path = var.D.Encode(**args)
                file = var.F[args.task].MASTER.Encode(**args)
                vlist[var.USDPATH] = utl.SJoin(path, file)
                res[key] = vlist

    # set override variants
    for i in range(node.parm('overvariantsfolder').evalAsInt()):
        vset = node.parm('overvariantset%d'%i).evalAsString()
        val  = node.parm('overvariant%d'%i).evalAsString()

        if not (vset and val):
            continue

        if ':' in vset:
            # find variantSet in asset
            vset = key.split(':')
            key  = 'asset:%s'%vset[0]
            vset = vset[1]
            if not res[key].has_key(vset):
                res[key][var.ORDER].append(vset)
            res[key][vset] = val
        else:
            # find variantSet in shot
            if res.has_key(var.T.SHOT):
                if not res[var.T.SHOT].has_key(vset):
                    res[var.T.SHOT][var.ORDER].append(vset)
                res[var.T.SHOT][vset] = val
            else:
                res[var.T.SHOT] = {vset:val, var.ORDER:[vset]}

    return res


def ResolveLayer(kwargs):
    node = kwargs['node']
    type = node.parm('lyrtype').evalAsString()

    prctype = node.parm('processtype').evalAsString()
    prcname = node.parm('processname').evalAsString()
    subprcname = node.parm('subprocessname').evalAsString()

    info = {}
    for k in LYRPARMS:
        info.update({k:None})

    obj = node.parm('%s_objpath'%type)
    obj = obj.evalAsNode() if obj else None

    # --------------------------------------------------------------------------
    # check shot or not
    if prctype in [var.PRCCLIP, var.PRCSIM]:
        isShot = True
    else:
        pubnode = []
        ntl.RetrieveByNodeType(node, 'dxusdROP_publish', pubnode, output=True)

        try:
            if pubnode:
                pubnode = hou.node(pubnode[0])
                respath = pubnode.parm('resultpath').evalAsString()
                if respath == var.UNKNOWN:
                    raise ValueError('Unknown result path')
                else:
                    isShot = var.D.Decode(respath).IsShot()
            else:
                raise Exception('No publish node')
        except Exception as e:
            isShot = node.parm('trange').evalAsInt() > 0
            msg.errmsg(e)
            msg.warning('Failed checking shot or not (assumes from trange)')

    # --------------------------------------------------------------------------
    # get dependency
    dependency = GetDependency(node)

    # --------------------------------------------------------------------------
    if   type == var.LYRGEOM:
        if   prctype == var.PRCCLIP:
            pass
        elif prctype == var.PRCSIM:
            pass
        elif prctype == var.PRCFX:
            pass
        else:
            ShowLayerResParms(node, 'dprim')
            info['dprim'] = 'Geom'

    # --------------------------------------------------------------------------
    elif type == var.LYRINST:
        if   prctype == var.PRCCLIP:
            pass
        elif prctype == var.PRCSIM:
            pass
        elif prctype == var.PRCFX:
            pass
        else:
            ShowLayerResParms(node, 'nslyr', 'dprim')
            info['dprim'] = 'World'
            if obj:
                nslyr = obj.name()

    # --------------------------------------------------------------------------
    elif type == var.LYRGROOM:
        if   prctype == var.PRCCLIP:
            pass
        elif prctype == var.PRCSIM:
            pass
        elif prctype == var.PRCFX:
            pass
        else:
            pass

    # --------------------------------------------------------------------------
    elif type == var.LYRCROWD:
        if   prctype == var.PRCCLIP:
            pass
        elif prctype == var.PRCSIM:
            pass
        elif prctype == var.PRCFX:
            pass
        else:
            pass

    # --------------------------------------------------------------------------
    elif type == var.LYRFEATHER:
        typeparm = node.parm('feather_exporttype')
        soppath  = typeparm.menuLabels()[typeparm.evalAsInt()].lower()
        soppath  = node.parm('feather_%spath'%soppath).evalAsString()
        soppath  = soppath.split('/')[-1]

        ShowLayerResParms(node, 'nslyr', 'lyrname', 'dprim')

        info['dprim'] = 'Feather'
        info['lyrname'] = soppath if soppath else var.UNKNOWN

        if   prctype == var.PRCCLIP:
            info['nslyr'] = node.parm('processname').evalAsString()
        elif prctype == var.PRCFX:
            info['nslyr']  = node.parm('processname').evalAsString()
            info['sublyr'] = node.parm('subprocessname').evalAsString()
        else: # none or sim
            if isShot:
                if dependency.has_key(var.T.SHOT):
                    info['nslyr'] = dependency[var.T.SHOT][var.NSLYR]
                else:
                    info['nslyr'] = obj.name() if obj else ''
            else:
                info['nslyr'] = obj.name() if obj else ''


    # --------------------------------------------------------------------------
    # set
    for k, v in info.items():
        if not v == None:
            node.parm(k).set(v)
