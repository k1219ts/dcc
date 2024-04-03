name = 'katana'
version = '6.0'

requires = [
    'python-3.9.15',
    'baselib-3.9',
    'ocio_configs-1.2'


]

def commands():
    import os
    # Project configuration
    cfg = os.getenv('PROJECTCONFIG')
    if cfg:
        env.KATANA_RESOURCES.prepend(cfg)

    # application version
    app_version = '6.0v1'

    env.KATANA_ROOT.set('/opt/Katana%s' % app_version)
    if not os.path.exists(str(env.KATANA_ROOT)):
        stop('Not installed Katana%s' % app_version)

    env.PATH.prepend(env.KATANA_ROOT)


    env.KATANA_USD_PLUGINS_DISABLED.set(1)
    env.KATANA_RESOURCES.append('{root}/Resources')
    env.PYTHONPATH.append('{root}/scripts')
    env.PYTHONPATH.append('/backstage/libs/usd_katana/22.05/katana-6.0/lib/python')

    env.LUA_PATH.append('{root}/lua/?.lua')

    env.KATANA_NUKE_EXECUTABLE.append('{root}/nuke/Nuke13.2')
    env.KATANA_DISABLE_FORESIGHT_PLUS.set(1)


    env.OCIO.set('{}/katana_config.ocio'.format(getenv('REZ_OCIO_CONFIGS_ROOT')))

def post_commands():
    if defined('LUA_PATH'):
        env.LUA_PATH.set(str(env.LUA_PATH).replace(':', ';'))

    print 'resolved packages'
    print '>>', env.REZ_USED_RESOLVE
    print 'hello'






