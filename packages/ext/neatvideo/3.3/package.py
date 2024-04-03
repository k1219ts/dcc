name = 'neatvideo'
version = '3.3'

def commands():
    import os

    appdir = '/opt/plugins/neatvideo/{}'.format(version)
    if not os.path.exists(appdir):
        stop('Not installed Neatvideo 3.3')

    env.OFX_PLUGIN_PATH.append(appdir)
