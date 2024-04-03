# -*- coding: utf-8 -*-
import getpass

name = 'nuke'

requires = [
    'baselib-3.7',
    'openexr-2.4',
    'pgbokeh-1.4.8',
    'neatvideo-4.8.8',
    'rsmb-5.2.4',   # 'rsmb-6.5.5',
    'sapphire',
    'ldpk-2.8',
#    'ocio_configs-1.2',
    'opticalflares-1.0.9',
    'twixtor-5.2.3',    #'twixtor-7.4.1',
    'nuke_scripts-3.0',
    'python-3.7'
]
### comp ACES-1.3 test ### -- 테스트 위해 멈춤 / 금일 재수정 예정
user = getpass.getuser()
if user == 'jinhee.leec' :
    requires.append('ocio_configs-1.2')
else :
    requires.append('ocio_configs-1.2')


def commands():
    import os
    import platform
    cfg = os.getenv('PROJECTCONFIG')
    if cfg:
        env.NUKE_PATH.prepend(cfg)

    version = '13.2v5'
    env.NUKE_VER.set(version)
    env.NUKE_LOCATION.set('/opt/Nuke{}'.format(version))
    if not os.path.exists(str(env.NUKE_LOCATION)):
        stop('Not installed Nuke 13.2v5')

    env.PATH.append(env.NUKE_LOCATION)
    env.FN_NUKE_DISABLE_GPU_ACCELERATION.set('1')

    alias('nuke', 'Nuke13.2 --nukex --disable-nuke-frameserver')
    alias('nukeX', 'Nuke13.2 --nukex --disable-nuke-frameserver')
    alias('nukeS', 'Nuke13.2 --studio --disable-nuke-frameserver')
    alias('nukeP', 'Nuke13.2 --player --disable-nuke-frameserver')

########## Katana Bridge #############
    info = platform.uname()
    for pc in info:
        if 'ast' in pc or 'lnr' in pc :
            if os.path.exists(str('/opt/Katana6.0v1')):
    	        env.NUKE_PATH.append('/opt/Katana6.0v1/plugins/Resources/Nuke/13.2')
##################################
def post_commands():
    print ('resolved packages:')
    print ('>>', env.REZ_USED_RESOLVE)
    env.NUKE_PATH.prepend(env.NUKE_LOCATION)
