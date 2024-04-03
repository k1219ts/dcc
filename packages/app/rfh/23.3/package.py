name = 'rfh'
version = '23.3'

requires = [
    'renderman-23.3'
]

variants = [
    ['houdini-17.5'],
    ['houdini-18.0']
]

def commands():
    import os

    env.RFHTREE.set('/opt/pixar/RenderManForHoudini-{}'.format(version))
    if not os.path.exists(str(env.RFHTREE)):
        stop('Not installed RenderManForHoudini-23.3')

    env.RFH_ARGS2HDA.set('1')
    env.HOUDINI_PATH.append('%s/%s' % (env.RFHTREE, env.MAIN_HVER))
