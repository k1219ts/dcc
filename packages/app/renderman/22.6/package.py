name = 'renderman'
version = '22.6'

def commands():
    import os

    env.RMAN_VER.set(version)
    env.RMAN_MAIN_VER.set('22')
    env.RMANTREE.set('/opt/pixar/RenderManProServer-22.6')
    if not os.path.exists(str(env.RMANTREE)):
        stop('Not installed RenderManProServer-22.6')

    env.RMAN_RIXPLUGINPATH.prepend('{}/lib/plugins'.format(env.RMANTREE))
    env.RMAN_SHADERPATH.prepend('{}/lib/shaders'.format(env.RMANTREE))
    env.LD_LIBRARY_PATH.prepend('{}/lib'.format(env.RMANTREE))
    env.PATH.prepend('{}/bin'.format(env.RMANTREE))
    env.PYTHONPATH.append('{}/lib/python2.7/site-packages'.format(env.RMANTREE))

    env.RMAN_RIXPLUGINPATH.append('{root}/lib/plugins')
    env.RMAN_SHADERPATH.append('{root}/lib/shaders')
    env.DENOISE_CONFIG_PATH.set('{root}/resources/denoise')
    env.PATH.prepend('{root}/bin')
