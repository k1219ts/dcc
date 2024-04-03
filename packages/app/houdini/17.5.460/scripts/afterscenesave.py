import hou

FX_CACHE = '/netapp/fx_cache/extra'
FX_CACHE2 = '/fx_cache'
# FX_CACHE2 = '/space/fx_cache/extra'

def setVariable():
    # FX_CACHE
    hou.hscript('set -g FX_CACHE = %s' % FX_CACHE)
    # FX_CACHE2
    hou.hscript('set -g FX_CACHE2 = %s' % FX_CACHE2)

    hipfile = hou.hipFile.path()
    src = hipfile.split('/')

    # SHOW & SEQ & SHOT
    if 'show' in src:
        show = src[src.index('show') + 1]
        hou.hscript('set -g SHOW = %s' % show)
        if 'shot' in src:
            seq = src[src.index('shot') + 1]
            shot = src[src.index('shot') + 2]
            hou.hscript('set -g SEQ = %s' % seq)
            hou.hscript('set -g SHOT = %s' % shot)
        else:
            hou.hscript('set -u SEQ')
            hou.hscript('set -u SHOT')
    else:
        hou.hscript('set -u SHOW')

setVariable()

# print 'afterscenesave.py'