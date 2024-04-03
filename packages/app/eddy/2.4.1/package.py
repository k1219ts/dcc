name = 'eddy'

requires = [
    'nuke-10'
]

def commands():
    import os

    INSTALL_PATH='/opt/plugins/eddy/Eddy_for_Nuke-2.4.1-nuke10.0-Linux'
    if not os.path.exists(INSTALL_PATH):
        stop('Not installed Eddy 2.4.1.')

    try:
        if env.SITE == "KOR":
            EDDY_LICENSE = '6200@10.10.10.161'
        else:
            EDDY_LICENSE = '6200@11.0.2.32'
    except:
        EDDY_LICENSE = '6200@10.10.10.161'

    if not 'VORTECHS_LICENSE_PATH' in env.keys():
        env.VORTECHS_LICENSE_PATH.set(EDDY_LICENSE)

    env.EDDY_VER.set(version)
    env.NUKE_PATH.append(INSTALL_PATH)
