name = 'katana'
version = '5.0'

requires = [
    'python-3',
    'baselib-3.0',
    'ocio_configs'

]

def commands():
    import os
    # Project configuration
    cfg = os.getenv('PROJECTCONFIG')
    if cfg:
        env.KATANA_RESOURCES.prepend(cfg)

    # application version
    app_version = '5.0v4'

    env.KATANA_ROOT.set('/opt/Katana%s' % app_version)
    if not os.path.exists(str(env.KATANA_ROOT)):
        stop('Not installed Katana%s' % app_version)

    env.PATH.prepend(env.KATANA_ROOT)

    # USD Plugin
#    env.USD_KATANA_ALLOW_CUSTOM_MATERIAL_SCOPES.set('1')
#    env.LD_LIBRARY_PATH.append('{}/plugins/Resources/Usd/lib'.format(env.KATANA_ROOT))
#    env.PYTHONPATH.append('{}/plugins/Resources/Usd/lib/python'.format(env.KATANA_ROOT))
#    env.KATANA_RESOURCES.append('{}/plugins/Resources/Usd/plugin'.format(env.KATANA_ROOT))

    env.KATANA_USD_PLUGINS_DISABLED.set(1)
    env.KATANA_RESOURCES.append('{root}/Resources')
    env.PYTHONPATH.append('{root}/scripts')
    env.PYTHONPATH.append('/backstage/libs/usd_katana/21.05/katana-5.0/lib/python')
    env.LUA_PATH.append('{root}/lua/?.lua')

    # ocio
    env.OCIO.set('{}/katana_config.ocio'.format(getenv('REZ_OCIO_CONFIGS_ROOT')))

def post_commands():
    if defined('LUA_PATH'):
        env.LUA_PATH.set(str(env.LUA_PATH).replace(':', ';'))

    print 'resolved packages'
    print '>>', env.REZ_USED_RESOLVE
