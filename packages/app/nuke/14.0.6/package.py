# -*- coding: utf-8 -*-
name = 'nuke'

requires = [
    'baselib-3.9',
    'openexr',
    'neatvideo-4.8.8',
    'rsmb-5.2.4',   # 'rsmb-6.5.5',
    'sapphire',
    'ldpk-2.9.3',
    'ocio_configs-1.3',
    # 'opticalflares-1.0.9', # 14버전 지원되지 않음
    'twixtor-5.2.3',    #'twixtor-7.4.1',
    'nuke_scripts-4.0',
    'python-3.9'
]

def commands():
    import os

    cfg = os.getenv('PROJECTCONFIG')
    if cfg:
        env.NUKE_PATH.prepend(cfg)

    version = '14.0v6'
    env.NUKE_VER.set(version)

    env.NUKE_LOCATION.set('/opt/Nuke{}'.format(version))
    if not os.path.exists(str(env.NUKE_LOCATION)):
        stop('Not installed Nuke 14.0v6')

    env.PATH.append(env.NUKE_LOCATION)
#    env.FN_NUKE_DISABLE_GPU_ACCELERATION.set('1')

    alias('nuke', 'Nuke14.0 --nukex --disable-nuke-frameserver')
    alias('nukeX', 'Nuke14.0 --nukex --disable-nuke-frameserver')
    alias('nukeS', 'Nuke14.0 --studio --disable-nuke-frameserver')
    alias('nukeP', 'Nuke14.0 --player --disable-nuke-frameserver')

########## Katana Bridge #############
#    if os.path.exists(str('/opt/Katana6.0v1')):
#	    env.NUKE_PATH.append('/opt/Katana6.0v1/plugins/Resources/Nuke/13.2')

def post_commands():
    print ('resolved packages:')
    print ('>>', env.REZ_USED_RESOLVE)
    env.NUKE_PATH.prepend(env.NUKE_LOCATION)


    # 'pgbokeh-1.4.8',  14버전부터 'bokeh로 들어옴'
