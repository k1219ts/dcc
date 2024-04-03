name = 'redshift_houdini'

variants = [
    ['houdini-16.5.405']
]

def commands():
    env.REDSHIFT_LICENSEPATH.set('~/redshift/license/verification')
    env.REDSHIFT_DISABLELICENSECHECKOUTONINIT.set('1')
    env.REDSHIFT_DISABLEOUTPUTLOCKFILES.set('1')
    redshift_root = '/netapp/backstage/pub/apps/redshift2/applications/linux/{}'.format(version)
    env.REDSHIFT_COMMON_ROOT.set(redshift_root)
    env.REDSHIFT_COREDATAPATH.set(redshift_root)
    env.REDSHIFT_LOCALDATAPATH.set(redshift_root)
    env.REDSHIFT_PLUG_IN_PATH.set('{ROOT}/redshift4houdini/{VER}/dso'.format(ROOT=redshift_root, VER=env.HVER))
    env.REDSHIFT_PROCEDURALSPATH.set('{}/Procedurals'.format(env.REDSHIFT_COREDATAPATH))
    env.REDSHIFT_TEXTURECACHEBUDGET.set('128')
    env.HOUDINI_PATH.append('{ROOT}/redshift4houdini/{VER}'.format(ROOT=redshift_root, VER=env.HVER))
    env.HOUDINI_DSO_ERROR.set('2')
    env.LD_LIBRARY_PATH.append('{}/bin'.format(env.REDSHIFT_COREDATAPATH))
    env.LD_LIBRARY_PATH.append(env.REDSHIFT_COREDATAPATH)
    env.PATH.append('{}/bin'.format(env.REDSHIFT_COREDATAPATH))
