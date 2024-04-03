name = 'renderman'
version = '25.2'

def commands():
    import os

    env.RMAN_VER.set(version)
    env.RMAN_MAIN_VER.set('25')
    env.RMANTREE.set('/opt/pixar/RenderManProServer-{}'.format(version))
    if not os.path.exists(str(env.RMANTREE)):
        stop('Not installed RenderManProServer-{}'.format(version))

    env.RMAN_RIXPLUGINPATH.prepend('{}/lib/plugins'.format(env.RMANTREE))

    env.RMAN_SHADERPATH.prepend('{}/lib/shaders'.format(env.RMANTREE))

    env.LD_LIBRARY_PATH.prepend('{}/lib'.format(env.RMANTREE))
    env.PATH.prepend('{}/bin'.format(env.RMANTREE))
    env.PYTHONPATH.append('{}/bin'.format(env.RMANTREE))

    env.PYTHONPATH.append('{}/lib/python3.7/site-packages'.format(env.RMANTREE))


    env.RMAN_RIXPLUGINPATH.prepend('{root}/lib/plugins')
    env.RMAN_SHADERPATH.append('{root}/lib/shaders')
    env.RMAN_ASSET_LIBRARY.append('{root}/resources/RenderManAssetLibrary')
    cfg = os.getenv('PROJECTCONFIG')
    if cfg:
        env.RMAN_ASSET_LIBRARY.append('/show/{}/works/LNR/01_hdri/RenderManAssetLibrary'.format(os.getenv('SHOW')))
    env.DENOISE_CONFIG_PATH.set('{root}/resources/denoise')
    env.PATH.prepend('{root}/bin')


    # AI Denoise Core
    if 'katana-6.0' not in str(env.REZ_USED_RESOLVE):
        env.PYTHONPATH.append('{root}/AI_denoise')
        env.PYTHONPATH.append('{root}/AI_denoise/python')
        env.LD_LIBRARY_PATH.append('{root}/lib64')

