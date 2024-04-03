name = 'mari'

requires = [
    'baselib-3.0',
    'dxusd',
    'dxrulebook',
    # 'mari_extpack-5.2',
    'python-3.7',
    'pylibs-2.7',
    # 'pyalembic'
]

def commands():
    import os

    version = '5.0v4'
    env.MARI_LOCATION.set('/opt/Mari{}'.format(version))
    if not os.path.exists(str(env.MARI_LOCATION)):
        stop('Not installed Mari 5.0v4')

    env.PATH.prepend(env.MARI_LOCATION)
    if building:
        env.MARI_SDK_INCLUDE_DIR.set('{}/SDK/include'.format(getenv('MARI_LOCATION')))
    env.MARI_SCRIPT_PATH.append('{root}/scripts')


    env.PATH.append('{root}/bin')


def post_commands():
    # Debug
    print 'resolved packages:'
    print '>>', env.REZ_USED_RESOLVE
