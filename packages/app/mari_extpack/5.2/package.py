name = 'mari_extpack'
version = '5.2'

def commands():
    import os

    extpack_root = '/opt/plugins/mari_extpack/5.2'
    if not os.path.exists(extpack_root):
        stop('Not installed Mari ExtensionPack 5.2')

    env.MARI_SCRIPT_PATH.append(extpack_root)
    env.MARI_MODO_BAKE_PRESETS.append('{}/BakePresets'.format(extpack_root))
