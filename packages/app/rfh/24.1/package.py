name = 'rfh'
version = '24.1'

requires = [
    'renderman-24.1'
]

variants = [
    ['houdini-17.5.460'],
    ['houdini-18.0.597'],
    ['houdini-18.5.563'],
    ['houdini-18.5.596'],
    ['houdini-18.5.633']
]

def commands():
    import os

    env.RFHTREE.set('/opt/pixar/RenderManForHoudini-{}'.format(version))
    if not os.path.exists(str(env.RFHTREE)):
        stop('Not installed RenderManForHoudini-{}'.format(version))

    env.RFH_ARGS2HDA.set('1')
    env.HOUDINI_PATH.append('{0}/{1}'.format(env.RFHTREE, getenv('REZ_HOUDINI_VERSION')))
