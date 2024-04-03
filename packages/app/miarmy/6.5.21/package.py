name = 'miarmy'
version = '6.5.21'

variants = [
    ['maya-2018']
]

def commands():
    import os

    INSTALL_PATH='/opt/plugins/miarmy/6.5.21'
    if not os.path.exists(INSTALL_PATH):
        stop('Not installed Miarmy 6.5.21.')

    env.MAYA_ENABLE_LEGACY_VIEWPORT.set('1')
    env.MAYA_MODULE_PATH.append('{}/maya'.format(INSTALL_PATH))
    env.LD_LIBRARY_PATH.append('{}/maya/plug-ins'.format(INSTALL_PATH))

    env.MAYA_MODULE_PATH.append('{base}/dxarmy')
