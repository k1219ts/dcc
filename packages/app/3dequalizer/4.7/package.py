name = '3dequalizer'
version = '4.7'

requires = [
    'pylibs-2.7',
    'python-3.7.7',
    # 'baselib-2.5',
    # 'pyalembic',
    'dxrulebook'
]

def commands():
    # env.PYTHONPATH.prepend('/backstage/dcc/packages/ext/pylibs/2.7/lib/python2.7/site-packages')
    env.PYTHONPATH.prepend('/backstage/libs/python/3.7.7/lib/python3.7/site-packages')
    env.PYTHONPATH.append('/backstage/libs/tactic')

    env.PATH.append('{root}/bin')
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
