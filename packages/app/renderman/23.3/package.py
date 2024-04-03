name = 'renderman'
version = '23.3'

def commands():
    import os

    env.RMAN_VER.set(version)
    env.RMAN_MAIN_VER.set('23')
    env.RMANTREE.set('/opt/pixar/RenderManProServer-23.3')
    if not os.path.exists(str(env.RMANTREE)):
        stop('Not installed RenderManProServer-23.3')

    env.RMAN_RIXPLUGINPATH.prepend('{}/lib/plugins'.format(env.RMANTREE))
    env.RMAN_SHADERPATH.prepend('{}/lib/shaders'.format(env.RMANTREE))
    env.LD_LIBRARY_PATH.prepend('{}/lib'.format(env.RMANTREE))
    env.PATH.prepend('{}/bin'.format(env.RMANTREE))
    env.PYTHONPATH.append('{}/lib/python2.7/site-packages'.format(env.RMANTREE))

    env.RMAN_RIXPLUGINPATH.append('{root}/lib/plugins')
    env.RMAN_SHADERPATH.append('{root}/lib/shaders')
    env.DENOISE_CONFIG_PATH.set('{root}/resources/denoise')
    env.PATH.prepend('{root}/bin')
