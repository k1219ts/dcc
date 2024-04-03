name = 'rfh'
version = '23.5'

requires = [
    'renderman-23.5'
]

variants = [
    ['houdini-17.5.460'],
    ['houdini-18.0.460'],
    ['houdini-18.0.499'],
    ['houdini-18.0.532'],
    ['houdini-18.5.351'],
    ['houdini-18.5.596']
]

def commands():
    import os

    env.RFHTREE.set('/opt/pixar/RenderManForHoudini-{}'.format(version))
    if not os.path.exists(str(env.RFHTREE)):
        stop('Not installed RenderManForHoudini-23.5')

    env.RFH_ARGS2HDA.set('1')
    env.HOUDINI_PATH.append('{0}/{1}'.format(env.RFHTREE, getenv('REZ_HOUDINI_VERSION')))
