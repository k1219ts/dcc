name = 'rv'
version = '1.0.0'


variants = [
    ['centos-7.5'],
    ['centos-8.5']

]

requires = [
    'webp',
    'pylibs-3.7',
    'ocio_configs-1.2',
    'otio-0.13.2'
]

def commands():
    import os
    import platform

    rv_home = '/opt/RV/OpenRV-{}/app'.format(version)
    if not os.path.exists(rv_home):
        stop('Not installed rv 1.0.0.')

    env.RV_HOME.set(rv_home)
    env.PATH.prepend('{root}/bin')

    env.PYTHONPATH.prepend('{}/lib/python3.9/site-packages'.format(env.RV_HOME))
    env.PYTHONPATH.prepend('/backstage/libs/python/3.7.7/lib/python3.7/site-packages')
    env.PATH.append('{}/bin'.format(env.RV_HOME))

    env.RV_SUPPORT_PATH.set('{root}')
    env.LD_PRELOAD.append('{}/lib/libcrypto.so'.format(env.RV_HOME))

###### Centos7 library hooking #######
    if platform.platform() in 'centos-8':
        env.LD_PRELOAD.append('{root}/lib64/libk5cryptoso.3.1')


######################################

    env.PYTHONPATH.append('/backstage/libs/tactic/')

    env.LD_LIBRARY_PATH.append('{}/lib'.format(env.RV_HOME))
    env.LD_LIBRARY_PATH.append('{root}/lib')
    env.LD_LIBRARY_PATH.append('{root}/lib64')


###### prores codec hooking #######
    env.LD_PRELOAD.append('{root}/MovieFormats/mio_ffmpeg.so')
###################################


    # ocio
    env.OCIO.set('{}/config.ocio'.format(getenv('REZ_OCIO_CONFIGS_ROOT')))
