name = 'katana'
version = '3.2'

requires = [
    'baselib-2.5',
    'ocio_configs-1.0.3'
]



def commands():
    import os
    # Project configuration
    cfg = os.getenv('PROJECTCONFIG')
    if cfg:
        env.KATANA_RESOURCES.prepend(cfg)

    # application version
    app_version = '3.2v3'

    env.KATANA_ROOT.set('/opt/Katana%s' % app_version)
    if not os.path.exists(str(env.KATANA_ROOT)):
        stop('Not installed Katana%s' % app_version)

    env.PATH.prepend(env.KATANA_ROOT)

    env.KATANA_RESOURCES.append('{root}/Resources')
    env.PYTHONPATH.append('{root}/scripts')
    env.LUA_PATH.append('{root}/lua/?.lua')

    # ocio
    env.OCIO.set('{}/it.ocio'.format(getenv('REZ_OCIO_CONFIGS_ROOT')))


def post_commands():
    if defined('LUA_PATH'):
        env.LUA_PATH.set(str(env.LUA_PATH).replace(':', ';'))

    print 'resolved packages'
    print '>>', env.REZ_USED_RESOLVE
