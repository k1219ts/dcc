name = 'houdini'

requires = [
    'baselib-2.5'
]

def commands():
    import os.path

    env.HOUDINI_VERSION.set(version)
    env.HVER.set(version)
    env.MAIN_HVER.set(str(version)[:-4])
    env.HFS.set('/opt/hfs{VER}'.format(VER=version))
    if not os.path.exists(str(env.HFS)):
        stop("Not installed Houdini 17.5.229")

    env.H.set(env.HFS)
    env.HB.set('{}/bin'.format(env.H))
    env.HDSO.set('{}/dsolib'.format(env.H))
    env.HH.set('{}/houdini'.format(env.H))
    env.HHC.set('{}/config'.format(env.HH))
    env.HT.set('{}/toolkit'.format(env.H))
    env.HSB.set('{}/sbin'.format(env.HH))

    env.LD_LIBRARY_PATH.prepend(env.HDSO)

    env.PATH.prepend(env.HSB)
    env.PATH.prepend(env.HB)

    env.HOUDINI_PATH.append('{root}')


def post_commands():
    env.HOUDINI_PATH.append('&')
    env.HOUDINI_DSO_PATH.append('&')
    env.HOUDINI_SCRIPT_PATH.append('&')
    env.HOUDINI_OTLSCAN_PATH.append('&')
    env.HOUDINI_TOOLBAR_PATH.append('&')

    # Debug
    if defined('BUNDLE_NAME'):
        print '[[ BUNDLE_NAME ]]'
        print '>>', env.BUNDLE_NAME
    print 'resolved packages:'
    print '>>', env.REZ_USED_RESOLVE
