name = 'ziva'
version = '2.1'

variants = [
    ['maya-2022']
]

def commands():
    import os

    INSTALL_PATH='/opt/plugins/ziva/ziva-{}-v2.1'.format(resolve.maya.version)
    if not os.path.exists(INSTALL_PATH):
        stop('Not installed ZivaVFX 2.1')

    env.zivadyn_LICENSE.set('5053@10.10.10.113')
    env.ZIVA_VER.set(version)
    env.MAYA_MODULE_PATH.append(INSTALL_PATH + '/Ziva-VFX-Maya-Module')
