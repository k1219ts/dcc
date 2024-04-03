name = 'bora_houdini'

variants = [
    ['houdini-16.5.405'],
    ['houdini-16.5.571'],
    ['houdini-17.0.352']
]

def commands():
    env.BORA_VER.set(version)
    bora_root = '{BSP}/libs/Bora/{BVER}/houdini/{HVER}'.format(
        BSP=env.BACKSTAGE_PATH, BVER=version, HVER=env.HVER
    )
    env.HOUDINI_OTLSCAN_PATH.append('{}/otls'.format(bora_root))
    env.HOUDINI_DSO_PATH.append('{}/plugins'.format(bora_root))
    env.HOUDINI_VEX_DSO_PATH.append('{}/plugins'.format(bora_root))
    env.LD_LIBRARY_PATH.append('{BSP}/libs/Bora/{BVER}/lib'.format(BSP=env.BACKSTAGE_PATH, BVER=version))
