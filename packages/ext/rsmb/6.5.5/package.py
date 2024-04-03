name = 'rsmb'

def commands():
    import os

    appdir = '/opt/plugins/rsmb/{}'.format(version)
    if not os.path.exists(appdir):
        stop('Not installed ReelSmart MotionBlur 6.5.5.')

    env.OFX_PLUGIN_PATH.append(appdir)
    try:
        if env.SITE == "KOR":
            env.RVL_SERVER.set('10.10.10.108')
        else:
            env.RVL_SERVER.set('11.0.2.44')
    except:
        env.RVL_SERVER.set('10.10.10.108')
