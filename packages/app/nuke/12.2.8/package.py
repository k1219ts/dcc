name = 'nuke'

requires = [
    'baselib-2.5',
    'python-2',
    'openexr',
    # 'usd_nuke-20.05-dev',
    'pgbokeh-1.4.7',
    'neatvideo',
    'rsmb-5.2.4', # Real Smart Motion Blur
    'sapphire',
    'ldpk-2.8',
    'ocio_configs-1.2',
    'opticalflares-1.0.86',
    'twixtor-5.2.3',
    'nuke_extend',
    'nuke_scripts-2.0'
]

def commands():
    import os

    cfg = os.getenv('PROJECTCONFIG')
    if cfg:
        env.NUKE_PATH.prepend(cfg)

    version = '12.2v8'
    env.NUKE_VER.set(version)

    env.NUKE_LOCATION.set('/opt/Nuke{}'.format(version))
    if not os.path.exists(str(env.NUKE_LOCATION)):
        stop('Not installed Nuke 12.2v8')

    env.PATH.append(env.NUKE_LOCATION)
    env.FN_NUKE_DISABLE_GPU_ACCELERATION.set('1')

    alias('nuke', 'Nuke12.2 --nukex')
    alias('nukeX', 'Nuke12.2 --nukex')
    alias('nukeS', 'Nuke12.2 --studio')
    alias('nukeP', 'Nuke12.2 --player')


def post_commands():
    print 'resolved packages:'
    print '>>', env.REZ_USED_RESOLVE
    env.NUKE_PATH.prepend(env.NUKE_LOCATION)
