name = 'mari_extpack'
version = '6.1'

def commands():
    import os

    extpack_root = '/opt/plugins/mari_extpack/6.1'
    if not os.path.exists(extpack_root):
        stop('Not installed Mari ExtensionPack 6.1')
    
    env.MARI_EP_LICENSE_FILE_PATH.append('{}/MariExtensionPack_6R1v2_forMari6/Resources/License/'.format(extpack_root))
    env.MARI_SCRIPT_PATH.append(extpack_root)
    env.MARI_MODO_BAKE_PRESETS.append('{}/BakePresets'.format(extpack_root))
