name = '3dequalizer'
version = '4.5'

requires = [
    'baselib-2.5'
]

def commands():
    env.PATH.append('{root}/bin')
    env.PYTHON_CUSTOM_SCRIPTS_3DE4.append('/backstage/libs/python/2.7.16/lib/python2.7/site-packages')
    env.PYTHON_CUSTOM_SCRIPTS_3DE4.append('{}/lib/python2.7/site-packages'.format(getenv('REZ_PYLIBS_ROOT')))
    env.PYTHON_CUSTOM_SCRIPTS_3DE4.append('{root}/scripts')
    env.PYTHON_CUSTOM_SCRIPTS_3DE4.append('{root}/scripts/distortion')
    env.PYTHON_CUSTOM_SCRIPTS_3DE4.append('{root}/scripts/export')
    env.PYTHON_CUSTOM_SCRIPTS_3DE4.append('{root}/scripts/import')
    env.PYTHON_CUSTOM_SCRIPTS_3DE4.append('{root}/scripts/metadata')
    env.PYTHON_CUSTOM_SCRIPTS_3DE4.append('{root}/scripts/pftrack')
    env.PYTHON_CUSTOM_SCRIPTS_3DE4.append('{root}/scripts/setup')
    env.PYTHON_CUSTOM_SCRIPTS_3DE4.append('{root}/scripts/utils')

    # alias('3de4', '3DE4')


def post_commands():
    import sys

    if defined('DEV_LOCATION'):
        sys.stdout.write('\033[1;31m')
        print '[[ IS DEVELOPER MODE ]]'
        print '>>', env.DEV_LOCATION
        sys.stdout.write('\033[0;0m')

    if not defined('ISFARM'):
        sys.stdout.write('\033[1;94m')
    print 'resolved packages:'
    print '>>', env.REZ_USED_RESOLVE
    if not defined('ISFARM'):
        sys.stdout.write('\033[0;0m')
    print ''
