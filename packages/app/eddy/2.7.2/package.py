name = 'eddy'

requires = [
    'nuke-12'
]

def commands():
    import os

    nukeVer = getenv("REZ_NUKE_MAJOR_VERSION") + '.' + getenv("REZ_NUKE_MINOR_VERSION")
    INSTALL_PATH='/opt/plugins/eddy/Eddy_for_Nuke-{VERSION}-nuke{NUKE_VER}-Linux'.format(VERSION=version, NUKE_VER=nukeVer)
    if not os.path.exists(INSTALL_PATH):
        stop('Not installed Eddy 2.7.2.')

    try:
        if env.SITE == "KOR":
            EDDY_LICENSE = '6200@10.10.10.161'
            if not 'VORTECHS_LICENSE_PATH' in env.keys():
                env.VORTECHS_LICENSE_PATH.set(EDDY_LICENSE)
        else:
            EDDY_LICENSE = '6200@11.0.2.32'
            if not 'VORTECHS_LICENSE_PATH' in env.keys():
                env.VORTECHS_LICENSE_PATH.set(EDDY_LICENSE)
    except:
        pass

    env.EDDY_VER.set(version)
    env.NUKE_PATH.append(INSTALL_PATH)
