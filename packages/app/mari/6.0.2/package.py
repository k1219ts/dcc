name = 'mari'

requires = [
    'baselib-3.9',
    'dxusd',
    'dxrulebook',
    'mari_extpack-6.1',
    'python-3.9',
    'pylibs-3.7',
    # 'pyalembic'
]

def commands():
    import os

    version = '6.0v2'
    env.MARI_LOCATION.set('/opt/Mari{}'.format(version))
    if not os.path.exists(str(env.MARI_LOCATION)):
        stop('Not installed Mari 6.0v2')

    env.PATH.prepend(env.MARI_LOCATION)
    if building:
        env.MARI_SDK_INCLUDE_DIR.set('{}/SDK/include'.format(getenv('MARI_LOCATION')))
    env.MARI_SCRIPT_PATH.append('{root}/scripts')
    env.PYTHONPATH.append('{root}/scripts')
    env.LD_PRELOAD.append('/usr/lib64/libcrypto.so.1.1')

    env.PATH.append('{root}/bin')


def post_commands():
    # Debug
    print 'resolved packages:'
    print '>>', env.REZ_USED_RESOLVE
