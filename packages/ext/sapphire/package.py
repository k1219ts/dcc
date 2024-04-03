name = 'sapphire'

def commands():
    import os

    appdir = '/usr/genarts'
    if not os.path.exists(appdir):
        stop('Not installed Sapphire')

    env.OFX_PLUGIN_PATH.append('{}/SapphireOFX'.format(appdir))
    env.SAPPHIRE_OFX_DIR.set('{}/SapphireOFX'.format(appdir))

    try:
        if env.SITE == "KOR":
            env.genarts_LICENSE.set('5053@10.10.10.109')
        else:
            env.genarts_LICENSE.set('5053@11.0.2.43')
    except:
        env.genarts_LICENSE.set('5053@10.10.10.109')
