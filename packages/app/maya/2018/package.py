name = 'maya'
version = '2018'

requires = [
    'baselib-2.5',
    'assetbrowser'
]

def commands():
    import os.path
    cfg = os.getenv('PROJECTCONFIG')
    if cfg:
        env.MAYA_SCRIPT_PATH.prepend(cfg)
        env.PYTHONPATH.prepend(cfg)

    # default
    env.MAYA_NO_WARNING_FOR_MISSING_DEFAULT_RENDERER.set('1')
    env.MAYA_VP2_DEVICE_OVERRIDE.set('VirtualDeviceGL')
    env.MAYA_OPENCL_IGNORE_DRIVER_VERSION.set(1)
    env.MAYA_IGNORE_OPENCL_VERSION.set(1)
    env.MAYA_ENABLE_LEGACY_VIEWPORT.set(1)

    env.MAYA_VER.set(version)
    env.MAYA_LOCATION.set('/usr/autodesk/maya%s' % version)
    if not os.path.exists(str(env.MAYA_LOCATION)):
        stop('Not installed Maya 2018')

    env.PATH.append('%s/bin/' % env.MAYA_LOCATION)

    env.MAYA_MODULE_PATH.append('{root}')

def post_commands():
    # Debug
    print 'resolved packages:'
    print '>>', env.REZ_USED_RESOLVE
