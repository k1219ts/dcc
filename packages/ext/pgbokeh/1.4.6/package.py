# -*- coding: utf-8 -*-
name = 'pgbokeh'

def commands():
    import os

    nukeVer = getenv("REZ_NUKE_MAJOR_VERSION") + '.' + getenv("REZ_NUKE_MINOR_VERSION")
    pluginDir = '/opt/plugins/bokeh/Bokeh-v{VERSION}_Nuke{NUKE_VER}-linux'.format(VERSION=version,
                                                                                  NUKE_VER=nukeVer)
    if not os.path.exists(pluginDir):
        stop('Not installed pgBokeh-1.4.6 in Nuke-{NUKE_VER}'.format(NUKE_VER=nukeVer))

    try:
        if env.SITE == "KOR":
            env.peregrinel_LICENSE.set('5053@10.10.10.109')
        else:
            env.peregrinel_LICENSE.set('5053@11.0.2.43')
    except:
        env.peregrinel_LICENSE.set('5053@10.10.10.109')

    env.PG_BOKEH_ABORT_ON_LICENSE_FAIL.set(1) # if license error, abort error

    env.NUKE_PATH.prepend(pluginDir)
