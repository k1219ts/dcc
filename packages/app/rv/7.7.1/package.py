name = 'rv'
version = '7.7.1'

requires = [
    'pylibs-2.7',
    'ocio_configs',
    'otio-0.13.2'
]

def commands():
    import os

    env.PYTHONPATH.prepend('/backstage/libs/python/2.7.16/lib/python2.7/site-packages')
    env.PYTHONPATH.append('/backstage/libs/tactic/')

    rv_home = '/opt/RV/rv-centos7-x86-64-{}'.format(version)
    if not os.path.exists(rv_home):
        stop('Not installed rv 7.7.1.')

    env.PATH.prepend('{root}/bin')

    env.RV_HOME.set(rv_home)
    env.PATH.append('{}/bin'.format(env.RV_HOME))
    env.LD_LIBRARY_PATH.append('{}/lib'.format(env.RV_HOME))
    env.MAGICK_CONFIGURE_PATH.set('{}/etc/config'.format(env.RV_HOME))

    env.RV_SUPPORT_PATH.set('{root}')

    # ocio
    env.OCIO.set('{}/config.ocio'.format(getenv('REZ_OCIO_CONFIGS_ROOT')))
