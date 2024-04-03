name = 'ziva'
version = '1.9'

variants = [
    ['maya-2018'],
    ['maya-2019']
]

def commands():
    import os

    INSTALL_PATH='/opt/plugins/ziva/ziva-{}-v1.9'.format(resolve.maya.version)
    if not os.path.exists(INSTALL_PATH):
        stop('Not installed ZivaVFX 1.9')

    env.zivadyn_LICENSE.set('5053@10.10.10.113')
    env.ZIVA_VER.set(version)
    env.MAYA_MODULE_PATH.append(INSTALL_PATH + '/Ziva-VFX-Maya-Module')