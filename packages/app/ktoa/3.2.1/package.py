name = 'ktoa'
version = '3.2.1'

requires = [
    'katana-4.0'
]

def commands():
    import os

    appdir = '/opt/arnold/ktoa-3.2.1.1-kat4.0'
    if not os.path.exists(appdir):
        stop('Not installed Katana to Arnold %s' % version)

    try:
        if env.SITE == "KOR":
            env.solidangle_LICENSE.set('5053@10.10.10.130')
        else:
            env.solidangle_LICENSE.set('5053@11.0.2.41')
    except:
        env.solidangle_LICENSE.set('5053@10.10.10.130')

    env.KTOA_ROOT.set(appdir)
    env.DEFAULT_RENDERER.set('arnold')
    env.KATANA_TAGLINE.set('With KtoA 3.2.1.1 and Arnold 6.2.0.1')

    env.USD_KATANA_ALLOW_CUSTOM_MATERIAL_SCOPES.set('true')
    env.PATH.prepend('{}/bin'.format(env.KTOA_ROOT))
    env.LD_LIBRARY_PATH.prepend('{KTOA_ROOT}/USD/KatanaUsdPlugins/lib:{KTOA_ROOT}/USD/KatanaUsdPlugins/plugin/Libs'.format(KTOA_ROOT=env.KTOA_ROOT))
    env.KATANA_RESOURCES.prepend('{KTOA_ROOT}:{KTOA_ROOT}/USD/KatanaUsdPlugins/plugin:{KTOA_ROOT}/USD/KatanaUsdArnold'.format(KTOA_ROOT=env.KTOA_ROOT))
    env.FNPXR_PLUGINPATH.prepend('{}/USD/Viewport'.format(env.KTOA_ROOT))
    env.PYTHONPATH.prepend('{}/USD/KatanaUsdPlugins/lib/python'.format(env.KTOA_ROOT))

    alias('ktoa', 'katana')
