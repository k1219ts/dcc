name = 'mari'

requires = [
    'baselib-2.5',
    'dxusd',
    'mari_extpack-5.2',
    'pylibs-2.7',
    'python-2'
]

def commands():
    import os

    version = '4.6v1'
    env.MARI_LOCATION.set('/opt/Mari{}'.format(version))
    if not os.path.exists(str(env.MARI_LOCATION)):
        stop('Not installed Mari 4.6v1')

    env.PATH.prepend(env.MARI_LOCATION)
    env.MARI_SCRIPT_PATH.append('{root}/scripts')


    env.PATH.append('{root}/bin')


def post_commands():
    # Debug
    print 'resolved packages:'
    print '>>', env.REZ_USED_RESOLVE
