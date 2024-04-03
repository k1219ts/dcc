name = 'neatvideo'

def commands():
    import os

    appdir = '/opt/plugins/neatvideo/{}'.format(version)
    if not os.path.exists(appdir):
        stop('Not installed Neatvideo 4.8.8')

    env.OFX_PLUGIN_PATH.append(appdir)
