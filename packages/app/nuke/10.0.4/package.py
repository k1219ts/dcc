name = 'nuke'

requires = [
    'baselib-2.5',
    'openexr',
    'pgbokeh-1.4.6',
    'neatvideo-4.8.8',
    'rsmb-5.2.4', # Real Smart Motion Blur
    'opticalflares-1.0.86',
    'sapphire',
    'ldpk',
    'ocio_configs',
    'nuke_extend',
    'nuke_scripts-1.0'
]

def commands():
    import os
    env.PYTHONPATH.prepend('{root}/scripts/python')

    cfg = os.getenv('PROJECTCONFIG')
    if cfg:
        env.NUKE_PATH.prepend(cfg)

    version = '10.0v4'
    env.NUKE_VER.set(version)

    env.NUKE_LOCATION.set('/opt/Nuke{}'.format(version))
    if not os.path.exists(str(env.NUKE_LOCATION)):
        stop('Not installed Nuke 10.0v4')

    # env.NUKE_PATH.append('{root}/scripts')
    env.PATH.append(env.NUKE_LOCATION)

    alias('nuke', 'Nuke10.0')
    alias('nukeX', 'Nuke10.0 --nukex')
    alias('nukeS', 'Nuke10.0 --studio')
    alias('nukeP', 'Nuke10.0 --player')


def post_commands():
    print 'resolved packages:'
    print '>>', env.REZ_USED_RESOLVE
