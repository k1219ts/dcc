name = 'usdtoolkit'

requires = [
    'usd_core',
    'dxusd',
    'renderman-23.5',
    'python-2.7',
    'pylibs-2.7',
    'pyside2-5.12.6',
    'pyalembic',
    'ocio_configs',
]

variants = [
    ['usd_core-21.05'],
    ['usd_core-20.08'],
    ['usd_core-20.05'],
    ['usd_core-20.02']
]

def commands():
    import os

    env.PATH.prepend('{base}/bin')
    env.PATH.prepend('{root}/bin')

    # ocio
    env.OCIO.set('{}/katana_config.ocio'.format(getenv('REZ_OCIO_CONFIGS_ROOT')))

    # motionblur
    env.HDX_PRMAN_ENABLE_MOTIONBLUR.set('1')

    # viewer plugins
    def add_view_plugin(dirpath):
        if os.path.exists(dirpath):
            env.PYTHONPATH.append(dirpath)
            for d in os.listdir(dirpath):
                if os.path.exists(os.path.join(dirpath, d, 'plugInfo.json')):
                    env.PXR_PLUGINPATH_NAME.append(os.path.join(dirpath, d))

    add_view_plugin('{}/plugins/viewer'.format(getenv('REZ_USDTOOLKIT_BASE')))
    add_view_plugin('{}/plugins/viewer'.format(getenv('REZ_USDTOOLKIT_ROOT')))
