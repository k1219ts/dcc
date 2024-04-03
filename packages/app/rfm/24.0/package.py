name = 'rfm'
version = '24.0'

requires = [
    'maya',
    'renderman-24.0'
]

def commands():
    import os

    env.RFMTREE.set('/opt/pixar/RenderManForMaya-24.0')
    if not os.path.exists(str(env.RFMTREE)):
        stop('Not installed RenderManForMaya-24.0')

    env.RFM_DO_NOT_CREATE_MODULE_FILE.set('1')

    env.MAYA_MODULE_PATH.append('{}/etc'.format(env.RFMTREE))

    env.RFM_SITE_PATH.set(getenv('REZ_RENDERMAN_ROOT'))
    env.XBMLANGPATH.append('{}/resources/icons/%B'.format(env.RFM_SITE_PATH))
